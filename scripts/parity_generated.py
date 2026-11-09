"""Read-only parity check of the generated variants on the real data (src/stratastills/projects_store.py): for every item that has a folder under <project>/generated, the sidecar records the file store finds against the ones the database store finds (key, picture file, output size, bytes, preset, created), and how many variants `projects.generated_variants` gives under each store. Nothing is written. Exit status 0 only when everything is equal and at least one variant was seen.
python scripts/parity_generated.py [--url postgresql+psycopg:///omarchy?options=-csearch_path%3Dphotos,public]"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import projects as P
from stratastills import projects_store, settings
from stratastills.db.projects import DbStore


def view(recs):
    return sorted(
        (r["key"], r["file"], tuple(r["output"]), r.get("bytes"), r.get("preset"), r.get("created")) for r in recs
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=settings.database_url() or "photos")
    a = ap.parse_args()
    files, db = projects_store.FileStore(), DbStore(a.url)
    rows = []
    ok = True
    seen = 0
    for p in P.list_projects():
        folder = P.folder(p["id"])
        srcs = [x["path"] for x in p["sources"]]
        g = os.path.join(folder, "generated")
        if not os.path.isdir(g):
            continue
        n_items = n_equal = n_f = n_d = 0
        for sdoc in p["sources"]:
            sid = P.source_id(sdoc["path"])
            for d in sorted(os.listdir(g)):
                stem = d.partition("~")[2]
                iid = P.item_id(sid, stem)
                fv, dv = (
                    view(files.generated(p["id"], folder, iid, srcs)),
                    view(db.generated(p["id"], folder, iid, srcs)),
                )
                n_items += 1
                n_f += len(fv)
                n_d += len(dv)
                n_equal += fv == dv
                seen += len(fv)
                if fv != dv and len(rows) < 3:
                    rows.append(dict(item=iid, file=fv[:2], database=dv[:2]))
        ok &= n_items == n_equal
        rows.insert(
            0, dict(project=p["id"], items=n_items, items_equal=n_equal, variants_file=n_f, variants_database=n_d)
        )
    print(json.dumps(dict(projects=rows), indent=1, default=str))
    print("PARITY OK" if ok and seen else "PARITY DIFFERS")
    return 0 if ok and seen else 1


if __name__ == "__main__":
    sys.exit(main())
