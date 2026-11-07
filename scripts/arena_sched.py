"""A greedy two-stage work pipeline for scripts/build_arena.py: stage 1 (`produce`, the neutral develop: GPU-heavy) turns a chunk of pictures into ready items, stage 2 (`consume`, the variants: CPU-heavy) makes the output of one ready item. One pool of worker threads does both: a worker that finds a ready item makes it; otherwise it takes the next chunk to produce, provided fewer than `gpu_workers` chunks are in the produce stage already and fewer than `max_ready` items wait (back-pressure: a ready item holds a 6000 px picture, about 70 MB); otherwise it waits. So the card develops the next pictures while the CPUs make the variants of the last, and neither idles while the other has work. The pool is `gpu_workers + cpu_workers` threads (the cv2 and torch calls release the GIL); with `serial=True` everything runs in the calling thread, one chunk after the other, the way the script worked before (the reference the scheduled run is compared with).
A failure of a chunk (produce) or of an item (consume) is reported through `on_error(kind, what, exception)` and the rest goes on. Pure Python: tested with fake jobs (tests/unit/test_arena_sched.py)."""
import os, threading, time


def auto_workers(cpus=None, gpu_slots=None):
    """(gpu_workers, cpu_workers) for this machine unless STRATASTILLS_ARENA_GPU_WORKERS / STRATASTILLS_ARENA_CPU_WORKERS say: two chunks in the develop stage at a time (measured on the RTX 3060, whose memory holds one whole development: 12 pictures took 102 s with one and 81 s with two or three, because the develop has CPU parts (raw decode, resize, JPEG encode) that overlap the other's GPU work, and gpu.slot() keeps the card's memory in bounds whatever the number), and for the variants about two cores each (cv2 + the registration): half the cores, at most 6. `gpu_slots` is accepted for callers that know the card; one slot still takes two workers."""
    cpus = cpus or os.cpu_count() or 2; env = lambda k: int(os.environ[k]) if os.environ.get(k, '').strip() else None
    g = env('STRATASTILLS_ARENA_GPU_WORKERS'); c = env('STRATASTILLS_ARENA_CPU_WORKERS')
    return (g if g is not None else 2), (c if c is not None else max(1, min(6, cpus // 2)))


def run(chunks, produce, consume, gpu_workers=1, cpu_workers=2, max_ready=None, serial=False, on_error=None):
    """Run `produce(chunk) -> [item, ...]` over `chunks` and `consume(item)` over every item produced. Returns dict(chunks=, items=, failed_chunks=, failed_items=, seconds=). `chunks` is a list, taken in order (greedily: whichever worker is free takes the next); items of one chunk are consumed in order of production when one worker does them, otherwise as workers get to them."""
    on_error = on_error or (lambda kind, what, e: None); t0 = time.time(); stats = dict(chunks=0, items=0, failed_chunks=0, failed_items=0); cv = threading.Condition()
    def make(chunk):
        try: out = list(produce(chunk)); ok = True
        except Exception as e: out = []; ok = False; on_error('chunk', chunk, e)
        with cv: stats['chunks' if ok else 'failed_chunks'] += 1
        return out
    def eat(item):
        try: consume(item); ok = True
        except Exception as e: ok = False; on_error('item', item, e)
        with cv: stats['items' if ok else 'failed_items'] += 1
    if serial or (gpu_workers < 1 and cpu_workers < 1):
        for chunk in chunks:
            for item in make(chunk): eat(item)
        return dict(stats, seconds=time.time() - t0)
    gpu_workers = max(1, gpu_workers); cpu_workers = max(0, cpu_workers); n = gpu_workers + cpu_workers; max_ready = max_ready or 2 * n
    pending = list(chunks); ready = []; busy = dict(make=0, eat=0)
    def worker():
        while True:
            with cv:
                while True:
                    if ready: job = ('eat', ready.pop(0)); break
                    if pending and busy['make'] < gpu_workers and len(ready) < max_ready: job = ('make', pending.pop(0)); break
                    if not pending and not ready and not busy['make']: cv.notify_all(); return      # nothing is waiting, nothing is being made that could produce more: whatever is still being eaten is another worker's
                    cv.wait(0.05)
                busy[job[0]] += 1
            if job[0] == 'make':
                items = make(job[1])
                with cv: ready.extend(items); busy['make'] -= 1; cv.notify_all()
            else:
                eat(job[1])
                with cv: busy['eat'] -= 1; cv.notify_all()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
    for t in threads: t.start()
    for t in threads: t.join()
    return dict(stats, seconds=time.time() - t0)
