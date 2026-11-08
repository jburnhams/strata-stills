"""SHA-256 of files into a resumable JSONL (the format of scripts/load_hashes.py: path, sha256, size, mtime_ns), read-only, at nice 15 and idle I/O. For files that are not the raws of the catalogue (videos and their sidecars, say); the raws' own job was scripts/hash_new.py.
  python scripts/hash_to_jsonl.py OUT.jsonl FOLDER [FOLDER ...] [--ext .mp4,.lrf,.srt] [--recursive]
Resumable: paths already in OUT are skipped. The extension test is case-insensitive."""
import argparse, hashlib, json, os, shutil, subprocess, sys


def lower_priority():
    try:
        if hasattr(os, 'nice'): os.nice(15)
    except (OSError, AttributeError): pass
    try:
        if shutil.which('ionice'): subprocess.run(['ionice', '-c3', '-p', str(os.getpid())], check=False, capture_output=True)
    except OSError: pass


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(1 << 22): h.update(b)
    return h.hexdigest()


def files(folders, exts, recursive):
    for folder in folders:
        for d, dirs, names in os.walk(folder):
            if not recursive: dirs[:] = []
            for n in sorted(names):
                if os.path.splitext(n)[1].lower() in exts: yield os.path.join(d, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('out'); ap.add_argument('folders', nargs='+'); ap.add_argument('--ext', default='.mp4,.lrf,.srt'); ap.add_argument('--recursive', action='store_true')
    a = ap.parse_args(); exts = {e if e.startswith('.') else '.' + e for e in a.ext.lower().split(',')}; done = set()
    if os.path.exists(a.out):
        for ln in open(a.out, encoding='utf8'):
            try: done.add(json.loads(ln)['path'])
            except (ValueError, KeyError): pass
    lower_priority(); n = 0
    with open(a.out, 'a', encoding='utf8') as o:
        for p in files(a.folders, exts, a.recursive):
            if p in done: continue
            try: st = os.stat(p); sha = sha256_of(p)
            except OSError as e: print('skip', p, e, file=sys.stderr); continue
            o.write(json.dumps(dict(path=p, sha256=sha, size=st.st_size, mtime_ns=st.st_mtime_ns)) + '\n'); o.flush(); n += 1
    print(f'hashed {n}, already done {len(done)}'); return 0


if __name__ == '__main__': sys.exit(main())
