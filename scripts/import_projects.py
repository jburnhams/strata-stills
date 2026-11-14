"""Import a project (project.json, rules, source, generated sidecars) into schema `photos` (src/stratastills/db/projimport.py). Default is a DRY RUN: reads the folder and the hashes of the source's DNG/JPEG files and prints the counts and problems, touching no database. `--write` writes in one verified transaction; run again and it adds nothing. The photos (the DNGs, via scripts/load_hashes.py) must be loaded first. The JPEG next to each DNG becomes a file of role 'jpeg'.
python scripts/import_projects.py PROJECT_DIR --hashes drone_hazelmere_sha256.jsonl [--root ~/drone ...] [--write] [--url ...]"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import settings
from stratastills.db import hashes
from stratastills.db import projimport as P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--hashes", required=True)
    ap.add_argument("--root", action="append")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--url")
    a = ap.parse_args()
    problems = []
    recs = hashes.read_jsonl(a.hashes, problems)
    p = P.plan(a.project, recs, a.root or settings.source_roots())
    p["problems"] = problems + p["problems"]
    print(json.dumps(dict(counts=p["counts"], problems=p["problems"][:10]), indent=1, default=str))
    if not a.write:
        print("dry run: nothing written (use --write)")
        return 0
    from strata_python.db import connect, session

    eng = connect(a.url or "photos")
    with session(eng) as s:
        print("written:", P.write(s, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
