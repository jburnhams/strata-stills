"""Import drone videos (MP4 + SRT + LRF) into schema `photos` as photos of kind 'video' (src/stratastills/db/videoimport.py). Default is a DRY RUN: reads the hashes JSONL, the exiftool JSONL of the MP4s and the SRT files, and prints the counts and problems, touching no database. `--write` writes in one verified transaction; run again and it adds nothing.
python scripts/hash_to_jsonl.py video_sha256.jsonl "~/drone/20250402 - Hazelmere"                        (SHA-256 of the .mp4/.lrf/.srt files)
grep -i '\\.mp4"' video_sha256.jsonl > video_mp4.jsonl; python scripts/read_meta.py --hashes video_mp4.jsonl --out video_meta.jsonl   (exiftool of the MP4s)
python scripts/import_videos.py --hashes video_sha256.jsonl --meta video_meta.jsonl --root ~/drone [--write] [--url ...]"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import settings
from stratastills.db import hashes
from stratastills.db import videoimport as V


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hashes", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--root", action="append")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--url")
    a = ap.parse_args()
    problems = []
    recs = hashes.read_jsonl(a.hashes, problems)
    meta = [json.loads(ln) for ln in open(a.meta, encoding="utf8") if ln.strip()]
    p = V.plan(recs, meta, a.root or settings.source_roots())
    p["problems"] = problems + p["problems"]
    print(json.dumps(dict(counts=p["counts"], problems=p["problems"][:10]), indent=1))
    if not a.write:
        print("dry run: nothing written (use --write)")
        return 0
    from omarchy_db import connect, session

    eng = connect(a.url or "photos")
    with session(eng) as s:
        print("written:", V.write(s, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
