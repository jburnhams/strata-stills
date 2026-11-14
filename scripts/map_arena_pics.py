"""Fill photos.arena_pic.photo_id from the arena's picture names (src/stratastills/db/mapping.py). Default is a DRY RUN: reads the hashes JSONL, the Lightroom catalogue and the pool, prints the counts and the unmapped names, touching no database. `--write` fills the empty links in one verified transaction; run again and it adds nothing.
  python scripts/map_arena_pics.py --hashes ~/work/strata-stills/hash/raw_sha256.jsonl --catalogue ~/work/strata-stills/candidates/personal/catalogue.json [--report unmapped.json]
  python scripts/map_arena_pics.py ... --write [--url postgresql+psycopg:///omarchy?options=-csearch_path%3Dphotos,public]
Names: the arena pool (pool.json) and the report flags (out/flags/*.json). Only the arena pool's names have a row to link (report flags have no photo link); the others are counted and listed as 'report only'."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import settings
from stratastills.db import hashes, mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hashes", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--arena", default=settings.arena_dir())
    ap.add_argument("--flags", default=settings.report_flags_dir())
    ap.add_argument("--root", action="append")
    ap.add_argument("--report", help="write the unmapped names to this JSON file")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--url")
    a = ap.parse_args()
    problems = []
    recs = hashes.read_jsonl(a.hashes, problems)
    if problems:
        print("hash file problems:", problems[:5])
    cat = json.load(open(a.catalogue, encoding="utf8"))
    pool = json.load(open(os.path.join(a.arena, "pool.json"), encoding="utf8"))
    names = list(pool)
    p = mapping.plan(names, recs, cat, a.root or settings.source_roots())
    flag_names = []
    if os.path.isdir(a.flags):
        for f in sorted(os.listdir(a.flags)):
            if f.endswith(".json"):
                flag_names += [n for n in json.load(open(os.path.join(a.flags, f), encoding="utf8")) if n not in pool]
    fp = mapping.plan(dict.fromkeys(flag_names), recs, cat, a.root or settings.source_roots())
    print(
        json.dumps(
            dict(
                arena_pool=p["counts"],
                report_only=fp["counts"],
                unmapped=p["unmapped"][:20],
                report_only_unmapped=fp["unmapped"][:20],
            ),
            indent=1,
        )
    )
    if a.report:
        json.dump(
            dict(arena_pool=p["unmapped"], report_only=fp["unmapped"]), open(a.report, "w", encoding="utf8"), indent=1
        )
    if not a.write:
        print("dry run: nothing written (use --write)")
        return 0
    from strata_python.db import connect, session

    eng = connect(a.url or "photos")
    with session(eng) as s:
        print("written:", mapping.write(s, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
