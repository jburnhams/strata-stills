"""Apply a pipeline preset to every source of a project (sets the project's auto rule, then runs it). Usage: python scripts/run_rules.py <project id> [preset]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import pipeline as PL


def main(pid, preset='best'):
    PL.set_rules(pid, [dict(source='*', pipeline=preset)]); t = time.time(); print('pending', len(PL.pending(pid)), flush=True)
    last = [0]
    def progress(done, total, text):
        if done != last[0]: last[0] = done; print(f'{done}/{total}  {time.time() - t:.0f} s  {text}', flush=True)
    r = PL.apply_rules(pid, progress=progress); print('done', r['done'], 'failed', r['failed'], 'workers', r['workers'], f'{time.time() - t:.0f} s', flush=True)


if __name__ == '__main__': main(*sys.argv[1:3])
