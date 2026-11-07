"""Hash the raw files under the source roots that photos.file does not know yet (same root, path, size and mtime_ns) and write them straight to schema `photos` in batches. Resumable: a batch is committed before the next starts, and a rerun skips what is in the database. Runs at nice 15 and idle I/O priority; the files are only read.
  python scripts/hash_new.py [--root ~/raw ...] [--ext .arw,.cr2,...] [--batch 50] [--limit N] [--url ...]
Needs requirements-db.txt."""
import argparse, hashlib, os, shutil, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import settings

EXT = '.arw,.cr2,.cr3,.nef,.dng,.raf,.orf,.rw2,.srw,.pef,.nrw'


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(1 << 22): h.update(b)
    return h.hexdigest()


def lower_priority():
    """Nice 15 and idle I/O class where the platform has them (not Windows' os.nice, not macOS' ionice); whatever is missing or refused is skipped."""
    try:
        if hasattr(os, 'nice'): os.nice(15)
    except (OSError, AttributeError): pass
    try:
        if shutil.which('ionice'): subprocess.run(['ionice', '-c3', '-p', str(os.getpid())], check=False, capture_output=True)
    except (OSError, AttributeError): pass


def candidates(roots, exts):
    for root in roots:
        for d, _, names in os.walk(root):
            for n in sorted(names):
                if os.path.splitext(n)[1].lower() in exts: yield os.path.join(d, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--root', action='append'); ap.add_argument('--ext', default=EXT); ap.add_argument('--batch', type=int, default=50); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--url')
    a = ap.parse_args(); roots = a.root or settings.source_roots(); exts = {e if e.startswith('.') else '.' + e for e in a.ext.lower().split(',')}
    lower_priority()
    from omarchy_db import connect, session                               # the database packages are imported here, so sha256_of and candidates work (and are tested) without them
    from sqlalchemy import select
    from stratastills.db import hashes
    from stratastills.db.models import File
    eng = connect(a.url or 'photos'); labels = hashes.root_labels(roots); done = batch_n = 0; todo = []
    with session(eng) as s:
        known = {(r, p, z, m) for r, p, z, m in s.execute(select(File.root, File.relpath, File.size, File.mtime_ns))}
        def flush():
            nonlocal todo, done
            if not todo: return
            p = hashes.plan(todo, roots); print('batch', hashes.write(s, p), flush=True); done += len(todo); todo = []
        for path in candidates(roots, exts):
            ap_ = os.path.normpath(path); hit = next(((n, os.path.relpath(ap_, r)) for n, r in labels if ap_.startswith(r + os.sep)), None)
            try: st = os.stat(path)
            except OSError as e: print('skip', path, e, file=sys.stderr); continue
            if hit and (hit[0], hit[1].replace(os.sep, '/'), st.st_size, st.st_mtime_ns) in known: continue
            try: todo.append(dict(path=path, sha256=sha256_of(path), size=st.st_size, mtime_ns=st.st_mtime_ns))
            except OSError as e: print('skip', path, e, file=sys.stderr); continue
            if len(todo) >= a.batch: flush()
            if a.limit and done + len(todo) >= a.limit: break
        flush()
    print('hashed', done); return 0


if __name__ == '__main__': sys.exit(main())
