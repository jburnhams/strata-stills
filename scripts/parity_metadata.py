"""Read-only parity check of the DJI fields on the real data (src/stratastills/meta_store.py): for every DJI JPEG that the database knows (a `file` of role jpeg with a photo that has `photo_meta.dji`), the fields the file store reads from its XMP against what the database store gives (every field `meta.dji_fields` reports; numbers within 1e-6), and the nadir verdict. Nothing is written. Exit status 0 only when everything is equal and at least one file was checked.
  python scripts/parity_metadata.py [--url postgresql+psycopg:///omarchy?options=-csearch_path%3Dphotos,public]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from sqlalchemy import select
from stratastills import meta, meta_store, settings
from stratastills.db.meta import DbStore
from stratastills.db.models import File, PhotoMeta


def same(a, b):
    return set(a) == set(b) and all((abs(a[k] - b[k]) < 1e-6) if isinstance(a[k], float) and isinstance(b[k], float) else a[k] == b[k] for k in a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--url', default=settings.database_url() or 'photos'); a = ap.parse_args()
    files, db = meta_store.FileStore(), DbStore(a.url); n = eq = nadir_eq = missing = db_empty = 0; diffs = []
    with db_session(db) as s:
        rows = [(f.root, f.relpath) for f in s.scalars(select(File).where(File.role == 'jpeg', File.current.is_(True)).join(PhotoMeta, PhotoMeta.photo_id == File.photo_id).where(PhotoMeta.dji.is_not(None)))]       # only photos whose dji metadata is populated
    roots = dict(db.roots())
    for root, rel in rows:
        if root not in roots: missing += 1; continue
        p = os.path.join(roots[root], *rel.split('/'))
        if not os.path.exists(p): missing += 1; continue
        fd, dd = files.dji(p), db.db_fields(p)                                                                                  # db_fields is the database ONLY: no fall back to the file, so this never compares the file reader with itself
        if dd is None: db_empty += 1; continue
        n += 1; eq += same(fd, dd); nadir_eq += meta.is_nadir(fd) == meta.is_nadir(dd)
        if not same(fd, dd) and len(diffs) < 3: diffs.append(dict(path=p, file=fd, database=dd))
    ok = n > 0 and eq == n and nadir_eq == n and db_empty == 0
    print(json.dumps(dict(jpeg_files_with_dji_in_database=len(rows), compared_database_vs_file=n, fields_equal=eq, nadir_verdict_equal=nadir_eq, database_had_nothing_for_this_version=db_empty, not_on_disk_or_root_not_configured=missing, differences=diffs), indent=1, default=str)); print('PARITY OK' if ok else 'PARITY DIFFERS'); return 0 if ok else 1


def db_session(db):
    from omarchy_db import session
    return session(db.engine)


if __name__ == '__main__': sys.exit(main())
