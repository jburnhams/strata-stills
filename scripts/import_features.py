"""Import the derived feature matrices of the Lightroom look work into schema `photos.analysis` (src/stratastills/db/featimport.py). Default is a DRY RUN: reads the npz files, PROVES their row order against the exiftool tags (every row), prints the counts and problems, touching no database. `--write` adds the missing rows in one verified transaction; run again and it adds nothing. The photos must be loaded first.
  python scripts/import_features.py [--candidates ~/work/strata-stills/candidates/personal] [--hashes raw_sha256.jsonl] [--meta raw_meta.jsonl] [--write] [--url ...]
Needs numpy. A failed row-order proof is FATAL and nothing is written."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills.db import featimport as F
from stratastills.db import hashes


def main():
    h = os.path.expanduser("~/work/strata-stills")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", default=f"{h}/candidates/personal")
    ap.add_argument("--hashes", default=f"{h}/hash/raw_sha256.jsonl")
    ap.add_argument("--meta", default=f"{h}/hash/raw_meta.jsonl")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--url")
    a = ap.parse_args()
    problems = []
    hashed = {r["path"]: r["sha256"] for r in hashes.read_jsonl(a.hashes, problems)}
    tags_of = {}
    for ln in open(a.meta, encoding="utf8"):
        if ln.strip():
            r = json.loads(ln)
            tags_of[r["path"]] = r["tags"]
    cat = json.load(open(f"{a.candidates}/catalogue.json", encoding="utf8"))
    arrays, keys = F.load_arrays(a.candidates)
    p = F.plan(cat, hashed, tags_of, arrays, keys)
    p["problems"] = problems + p["problems"]
    print(json.dumps(dict(counts=p["counts"], problems=p["problems"][:10], input_keys=keys), indent=1))
    if not a.write:
        print("dry run: nothing written (use --write)")
        return 0
    from omarchy_db import connect, session

    eng = connect(a.url or "photos")
    with session(eng) as s:
        print("written:", F.write(s, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
