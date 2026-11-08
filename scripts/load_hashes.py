"""Load the raw files' SHA-256 (the JSONL of the hashing job: path, sha256, size, mtime_ns) into schema `photos` (src/stratastills/db/hashes.py). Default is a DRY RUN: parses the file and prints the counts, duplicates and problems, touching no database. `--write` writes them in one verified transaction; run again and it adds nothing.
  python scripts/load_hashes.py JSONL [--root ~/raw --root ~/drone]
  python scripts/load_hashes.py JSONL --write [--url postgresql+psycopg:///omarchy?options=-csearch_path%3Dphotos,public]
The roots default to the source roots of the settings file. The JSONL is only read."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import settings
from stratastills.db import hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('jsonl'); ap.add_argument('--root', action='append', help='a source root (repeatable; default: the settings file\'s)')
    ap.add_argument('--write', action='store_true', help='write to the database (default: dry run)'); ap.add_argument('--url', help='database URL (default: omarchy_db.connect("photos"))')
    a = ap.parse_args(); problems = []; recs = hashes.read_jsonl(a.jsonl, problems); p = hashes.plan(recs, a.root or settings.source_roots(), problems)
    dup = {s: v for s, v in list(p['duplicates'].items())[:5]}
    print(json.dumps(dict(counts=p['counts'], duplicates_sample=dup, problems=p['problems'][:20], problem_count=len(p['problems'])), indent=1))
    if not a.write: print('dry run: nothing written (use --write)'); return 0
    from omarchy_db import connect, session
    eng = connect(a.url or 'photos')
    with session(eng) as s: print('written:', hashes.write(s, p))
    return 0


if __name__ == '__main__': sys.exit(main())
