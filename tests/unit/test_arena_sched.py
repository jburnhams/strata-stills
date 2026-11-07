"""scripts/arena_sched.py: the greedy produce/consume pipeline of build_arena.py, with fake jobs (no GPU, no pictures)."""
import os, sys, threading, time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
import arena_sched as AS


def chunks(n=6, size=3): return [list(range(k * size, (k + 1) * size)) for k in range(n)]


def test_every_item_is_made_once_in_every_mode():
    for kw in (dict(serial=True), dict(gpu_workers=1, cpu_workers=0), dict(gpu_workers=1, cpu_workers=3), dict(gpu_workers=2, cpu_workers=4)):
        done, lock = [], threading.Lock()
        def consume(i):
            with lock: done.append(i)
        st = AS.run(chunks(), lambda c: c, consume, **kw)
        assert sorted(done) == list(range(18)) and st['items'] == 18 and st['chunks'] == 6 and not st['failed_chunks'] and not st['failed_items'], kw


def test_serial_mode_keeps_the_order():
    order = []; AS.run(chunks(3, 2), lambda c: c, order.append, serial=True); assert order == list(range(6))


def test_the_next_chunk_is_produced_while_items_are_consumed():
    started = threading.Event(); overlapped = []
    def produce(c):
        if c[0] > 0: overlapped.append(started.is_set())                      # the second chunk is taken while the first one's item is still being consumed
        return c
    def consume(i):
        if i == 0: started.set(); time.sleep(0.3)
    AS.run([[0], [1], [2]], produce, consume, gpu_workers=1, cpu_workers=1); assert overlapped and overlapped[0]


def test_at_most_gpu_workers_chunks_are_produced_at_a_time_and_ready_items_are_bounded():
    now = [0]; peak = [0]; lock = threading.Lock(); ready_peak = [0]; made = [0]
    def produce(c):
        with lock: now[0] += 1; peak[0] = max(peak[0], now[0])
        time.sleep(0.02)
        with lock: now[0] -= 1; made[0] += len(c)
        return c
    def consume(i):
        with lock: ready_peak[0] = max(ready_peak[0], made[0] - consumed[0]); consumed[0] += 1
        time.sleep(0.03)
    consumed = [0]
    AS.run(chunks(8, 2), produce, consume, gpu_workers=2, cpu_workers=2, max_ready=4); assert peak[0] <= 2 and ready_peak[0] <= 4 + 2 * 2


def test_a_failing_chunk_or_item_does_not_stop_the_rest():
    errors = []
    def produce(c):
        if c[0] == 3: raise RuntimeError('develop failed')
        return c
    def consume(i):
        if i == 7: raise ValueError('bad picture')
    done = []
    for mode in (dict(serial=True), dict(gpu_workers=1, cpu_workers=2)):
        errors.clear(); done.clear(); st = AS.run(chunks(), produce, lambda i: (consume(i), done.append(i)), on_error=lambda k, w, e: errors.append((k, type(e).__name__)), **mode)
        assert sorted(done) == [i for i in range(18) if i not in (3, 4, 5, 7)] and st['failed_chunks'] == 1 and st['failed_items'] == 1 and sorted(errors) == [('chunk', 'RuntimeError'), ('item', 'ValueError')]


def test_nothing_to_do_returns_at_once():
    assert AS.run([], lambda c: c, lambda i: None, gpu_workers=2, cpu_workers=2)['items'] == 0


def test_auto_workers_follow_the_machine_and_the_environment(monkeypatch):
    for k in ('STRATASTILLS_ARENA_GPU_WORKERS', 'STRATASTILLS_ARENA_CPU_WORKERS'): monkeypatch.delenv(k, raising=False)
    assert AS.auto_workers(cpus=16, gpu_slots=1) == (2, 6) and AS.auto_workers(cpus=16, gpu_slots=3) == (2, 6) and AS.auto_workers(cpus=2, gpu_slots=1) == (2, 1) and AS.auto_workers(cpus=1)[1] == 1
    monkeypatch.setenv('STRATASTILLS_ARENA_GPU_WORKERS', '3'); monkeypatch.setenv('STRATASTILLS_ARENA_CPU_WORKERS', '0'); assert AS.auto_workers(cpus=16, gpu_slots=1) == (3, 0)
