"""Benchmark: how much faster eight different shots are made when several are in flight at once.
    python -u scripts/bench_parallel.py [project-id] [stem ...]
Each configuration makes the same pictures with a different sharpen amount (so none is found 'already generated'); a warm-up run first so the file cache, the detail measurements and the GPU kernels are the same for all."""
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from stratastills import gpu, pipeline as PL, projects as P

pid = sys.argv[1] if len(sys.argv) > 1 else 'hazelmere'
stems = sys.argv[2:] or ['DJI_20250402190111_0037_D', 'DJI_20250402190158_0043_D', 'DJI_20250406151728_0101_D', 'DJI_20250406151600_0092_D', 'DJI_20250406151839_0104_D', 'DJI_20250402190256_0049_D', 'DJI_20250406151720_0100_D', 'DJI_20250402190126_0040_D']
sid = P.source_summaries(pid)[0]['id']; iids = [P.item_id(sid, s) for s in stems]
torch = gpu.torch_() if gpu.available() else None
print('GPU slots:', gpu.slots() if gpu.available() else 'no GPU', flush=True)


def run(workers, amount):
    if torch: torch.cuda.reset_peak_memory_stats()
    t = time.time(); times = []

    def one(i):
        t1 = time.time(); PL.generate(pid, i, 'raw', options=dict(sharpen_amount=amount)); times.append(time.time() - t1)
    with ThreadPoolExecutor(workers) as ex: list(ex.map(one, iids))
    return time.time() - t, float(np.mean(times)), (torch.cuda.max_memory_allocated() / 1e9 if torch else 0.0)


amounts = iter(np.arange(float(os.environ.get("BENCH_START", "1.31")), 2.99, 0.01)); WORKERS = [int(x) for x in os.environ.get('BENCH_WORKERS', '1,2,3,4').split(',')]; run(1, round(next(amounts), 2)); res = {}
for w, a in [(w, round(next(amounts), 2)) for w in WORKERS]:
    el, mean, peak = run(w, a); res.setdefault(w, []).append(el); print('workers %d: %5.1f s for %d pictures = %4.1f s each overall (one picture takes %5.1f s end to end), peak VRAM %.1f GB' % (w, el, len(iids), el / len(iids), mean, peak), flush=True)
base = min(res[1]); print('\nbest of two runs each:')
for w in sorted(res): b = min(res[w]); print('  %d at once: %5.1f s (%4.1f s per picture) speed-up x%.2f' % (w, b, b / len(iids), base / b), flush=True)
