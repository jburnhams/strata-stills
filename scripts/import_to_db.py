"""Import the arena's files into schema `photos` (src/stratastills/db/importer.py). Default is a DRY RUN: parses the files and prints the counts and problems, touching no database. `--write` writes them in one verified transaction (rolled back on a mismatch); run it again and it adds nothing.
  python scripts/import_to_db.py [--arena out/arena] [--flags out/flags]
  python scripts/import_to_db.py --write [--url postgresql+psycopg:///omarchy?options=-csearch_path%3Dphotos,public]
The files are only read. Needs requirements-db.txt."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import settings
from stratastills.db import importer


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--arena', default=settings.arena_dir()); ap.add_argument('--flags', default=settings.report_flags_dir())
    ap.add_argument('--write', action='store_true', help='write to the database (default: dry run)'); ap.add_argument('--url', help='database URL (default: omarchy_db.connect("photos"), i.e. PHOTOS_DATABASE_URL / OMARCHY_DB_URL / the shared server)')
    a = ap.parse_args(); p = importer.plan(a.arena, a.flags)
    print(json.dumps(dict(counts=p['counts'], problems=p['problems']), indent=1))
    if not a.write: print('dry run: nothing written (use --write)'); return 0
    from omarchy_db import connect, session
    eng = connect(a.url or 'photos')
    with session(eng) as s: print('written:', importer.write(s, p))
    return 0


if __name__ == '__main__': sys.exit(main())
