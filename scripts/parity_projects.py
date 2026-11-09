"""Read-only parity check of the projects store on the real data (src/stratastills/projects_store.py): for every project under the projects directory, the document and the item choices as the file store gives them against what the database store gives (name, created and each source's `added` as instants, source paths, rules, choices). Nothing is written anywhere. Exit status 0 only when everything is equal.
python scripts/parity_projects.py [--url postgresql+psycopg:///omarchy?options=-csearch_path%3Dphotos,public]"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import projects as P
from stratastills import projects_store, settings
from stratastills.db.projects import DbStore


def inst(s):
    if not s:
        return None
    d = datetime.fromisoformat(s)
    return (d if d.tzinfo else d.replace(tzinfo=UTC)).timestamp()


def norm(d):
    return (
        None
        if d is None
        else dict(
            name=d.get("name"),
            created=inst(d.get("created")),
            sources=[(x["path"], inst(x.get("added"))) for x in d.get("sources", [])],
            auto=[(r.get("source"), r.get("pipeline"), r.get("variant")) for r in d.get("auto", [])],
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=settings.database_url() or "photos")
    a = ap.parse_args()
    files, db = projects_store.FileStore(), DbStore(a.url)
    rows = []
    ok = True
    for p in P.list_projects():
        folder = P.folder(p["id"])
        fd, dd = files.read_doc(p["id"], folder), db.read_doc(p["id"], folder)
        fs, ds = files.read_state(p["id"], folder), db.read_state(p["id"], folder)
        same_doc, same_state = norm(fd) == norm(dd), fs == ds
        ok &= same_doc and same_state
        rows.append(
            dict(
                project=p["id"],
                document_equal=same_doc,
                choices_file=len(fs),
                choices_database=len(ds),
                choices_equal=same_state,
                sources=len(fd.get("sources", [])),
                rules=len(fd.get("auto", [])),
            )
        )
        if not same_doc:
            rows[-1]["file"], rows[-1]["database"] = norm(fd), norm(dd)
    print(json.dumps(dict(projects=rows), indent=1, default=str))
    print("PARITY OK" if ok and rows else "PARITY DIFFERS")
    return 0 if ok and rows else 1


if __name__ == "__main__":
    sys.exit(main())
