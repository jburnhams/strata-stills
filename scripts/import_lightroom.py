"""Import the Lightroom-side data into schema `photos` (src/stratastills/db/lrimport.py): the catalogue (export + settings), the nsfw cache, faces.json and the clip-pca32 vectors. Default is a DRY RUN: reads the files and prints the counts and the first problems, touching no database. `--write` writes each chosen source in its own verified transaction; run again and it adds nothing. Needs the photos already loaded (scripts/load_hashes.py).
  python scripts/import_lightroom.py [--source lightroom|nsfw|faces|embeddings|all] [--hashes JSONL] [--catalogue catalogue.json] [--personal ~/personal] [--write] [--url ...]
The sources' default places: candidates/personal/{catalogue,faces}.json and clip.npz under --candidates, ~/.stratastills/nsfw_cache.json. The files are only read."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills.db import hashes
from stratastills.db import lrimport as L

SOURCES = ("lightroom", "nsfw", "faces", "embeddings")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    h = os.path.expanduser("~/work/strata-stills")
    ap.add_argument("--source", default="all", choices=SOURCES + ("all",))
    ap.add_argument("--hashes", default=f"{h}/hash/raw_sha256.jsonl")
    ap.add_argument("--candidates", default=f"{h}/candidates/personal")
    ap.add_argument("--nsfw", default=os.path.expanduser("~/.stratastills/nsfw_cache.json"))
    ap.add_argument("--personal", default=os.path.expanduser("~/personal"), help="the root the exports are under")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--url")
    a = ap.parse_args()
    problems = []
    hashed = {r["path"]: r["sha256"] for r in hashes.read_jsonl(a.hashes, problems)}
    cat = json.load(open(f"{a.candidates}/catalogue.json", encoding="utf8"))
    roots = [a.personal]
    plans = {}
    for src in SOURCES if a.source == "all" else (a.source,):
        if src == "lightroom":
            plans[src] = L.plan_lightroom(cat, hashed, roots)
        elif src == "nsfw":
            plans[src] = L.plan_nsfw(json.load(open(a.nsfw, encoding="utf8")), cat, hashed, roots)
        elif src == "faces":
            plans[src] = L.plan_faces(cat, json.load(open(f"{a.candidates}/faces.json", encoding="utf8")), hashed)
        else:
            import numpy as np

            z = np.load(f"{a.candidates}/clip.npz")
            plans[src] = L.plan_embeddings(cat, dict(idx=z["idx"], pca=z["pca"]), hashed)
    print(json.dumps({s: dict(counts=p["counts"], problems=p["problems"][:5]) for s, p in plans.items()}, indent=1))
    if not a.write:
        print("dry run: nothing written (use --write)")
        return 0
    from strata_python.db import connect, session

    eng = connect(a.url or "photos")
    for s, p in plans.items():
        with session(eng) as sess:
            print(s, "written:", getattr(L, "write_" + s)(sess, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
