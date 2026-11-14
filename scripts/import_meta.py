"""Load the metadata JSONL of scripts/read_meta.py into schema `photos.photo_meta` (src/stratastills/db/metaimport.py). Default is a DRY RUN: parses the file and prints counts (how many rows have each main column filled) and problems, touching no database. `--write` adds the missing rows in one verified transaction; run again and it adds nothing. The photos must be loaded first (scripts/load_hashes.py).
python scripts/import_meta.py [JSONL]      (default ~/work/strata-stills/hash/raw_meta.jsonl)
python scripts/import_meta.py --write [--url postgresql+psycopg:///omarchy?options=-csearch_path%3Dphotos,public]"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills.db import metaimport as M


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jsonl", nargs="?", default=os.path.expanduser("~/work/strata-stills/hash/raw_meta.jsonl"))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--url")
    a = ap.parse_args()
    problems = []
    recs = M.read_jsonl(a.jsonl, problems)
    p = M.plan(recs, problems)
    print(json.dumps(dict(counts=p["counts"], problems=p["problems"][:10]), indent=1, default=str))
    if not a.write:
        print("dry run: nothing written (use --write)")
        return 0
    from strata_python.db import connect, session

    eng = connect(a.url or "photos")
    with session(eng) as s:
        print("written:", M.write(s, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
