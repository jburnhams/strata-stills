"""Read-only parity check of the nudity filter's two stores on the real data (src/stratastills/nsfw_store.py): the file store (<STRATASTILLS_HOME>/nsfw_cache.json, nsfw_allowed.json) against the database store (schema `photos`). Compares the cache entry by entry, the allowed decisions of the entries, and what the review page would list (nsfw.review(), which looks at the files on disk) under each. Nothing is written anywhere. Exit status 0 only when everything is equal.
  python scripts/parity_nsfw.py [--url postgresql+psycopg:///omarchy?options=-csearch_path%3Dphotos,public] [--extra-root ~/personal ...]
`--extra-root` is a folder that is not a source root but holds scored pictures (the Lightroom exports in ~/personal); the same list goes into `extra_roots` of settings.json for the running app."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import nsfw, nsfw_store, settings
from stratastills.db.nsfw import DbStore


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=settings.database_url() or "photos")
    ap.add_argument("--extra-root", action="append", default=[])
    a = ap.parse_args()
    roots = list(settings.source_roots()) + [os.path.expanduser(r) for r in a.extra_root]
    files = nsfw_store.FileStore()
    db = DbStore(a.url, roots=roots)
    fc, dc = files.load(), db.load_db()
    ok = True  # the database's own entries against the files'
    only_f, only_d = sorted(set(fc) - set(dc)), sorted(set(dc) - set(fc))
    diff = sorted(
        k for k in set(fc) & set(dc) if (float(fc[k][0]), str(fc[k][1] or "")) != (float(dc[k][0]), str(dc[k][1] or ""))
    )
    print(
        json.dumps(
            dict(
                cache=dict(
                    file=len(fc),
                    database=len(dc),
                    equal=len(set(fc) & set(dc)) - len(diff),
                    only_in_file=len(only_f),
                    only_in_database=len(only_d),
                    different_values=len(diff),
                    unresolved_database_rows=db.unresolved,
                ),
                samples=dict(only_in_file=only_f[:3], only_in_database=only_d[:3], different=diff[:3]),
            ),
            indent=1,
        )
    )
    ok &= not (only_f or only_d or diff)
    fa, da = files.allowed_among(list(fc)), db.allowed_among(list(dc))
    print(json.dumps(dict(allowed=dict(file=len(fa), database=len(da), equal=fa == da))))
    ok &= fa == da
    reviews = {}
    for name, st in (("file", files), ("database", db)):
        nsfw.reset()
        nsfw._store = lambda st=st: st
        reviews[name] = [(r["path"], r["score"], r["cls"], r["allowed"]) for r in nsfw.review()]
    print(
        json.dumps(
            dict(
                review=dict(
                    file=len(reviews["file"]),
                    database=len(reviews["database"]),
                    equal=reviews["file"] == reviews["database"],
                    threshold=nsfw.threshold(),
                )
            )
        )
    )
    ok &= reviews["file"] == reviews["database"]
    print("PARITY OK" if ok else "PARITY DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
