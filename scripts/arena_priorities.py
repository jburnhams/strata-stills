"""Which arena pairs are worth a vote: writes `<arena dir>/priorities.json`, which the arena's picker reads (arena.load_priorities / pair_weight). Nothing is removed from the arena: pairs are drawn at random with a weight, so clearly different pairs of the variants we care about, on pictures with a face, come first, and near-identical pairs are rare.
  python scripts/arena_priorities.py [--focus auto4,auto5,auto6,lightroom] [--arena DIR] [--size 96] [--no-faces]
- dist: per picture and pair of variants, the mean CIELAB distance of the two pictures at --size px (the pictures are laid on the same frame; a pair that differs in size is compared after resizing). Pictures whose files are unchanged since the last run are not measured again.
- faces: the pictures with a face of score >= --min-face in schema `photos` (detection label 'face', needs the database URL and omarchy_db), mapped by arena_pic.photo_id. Statues and the like count; the faces2.pkl real-face scores are not in the database.
- focus: the variants most worth comparing now (default: the three newest automatic versions and the person's final).
Run again after a pool rebuild or new votes; it takes a minute or two for ~600 pictures x 8 versions. Read-only on the pool and the pictures."""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import settings


def lab(path, size):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def distances(pool, root, size, old):
    out = {}
    for n, (pic, vs) in enumerate(pool.items(), 1):
        names = sorted(vs)
        key = {v: vs[v]["hash"] for v in names}
        prev = old.get(pic, {})
        labs = {}
        res = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if key[a] == key[b]:
                    continue
                k = f"{a}|{b}"
                if k in prev and prev[k][1] == [key[a], key[b]]:
                    res[k] = prev[k]
                    continue
                for v in (a, b):
                    if v not in labs:
                        labs[v] = lab(os.path.join(root, vs[v]["file"]), size)
                if labs[a] is None or labs[b] is None:
                    continue
                res[k] = [float(np.sqrt(((labs[a] - labs[b]) ** 2).sum(-1)).mean()), [key[a], key[b]]]
        out[pic] = res
        if n % 100 == 0:
            print(f"{n}/{len(pool)} pictures", flush=True)
    return out


def faces_in_db(min_score):
    url = settings.database_url()
    if not url:
        return None
    from omarchy_db import connect, session
    from sqlalchemy import select

    from stratastills.db.models import Analysis, ArenaPic, Detection

    with session(connect(url)) as s:
        q = (
            select(ArenaPic.name)
            .join(Analysis, Analysis.photo_id == ArenaPic.photo_id)
            .join(Detection, Detection.analysis_id == Analysis.id)
            .where(Detection.label == "face", Detection.score >= min_score)
            .distinct()
        )
        return sorted(s.scalars(q))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arena", default=settings.arena_dir())
    ap.add_argument("--focus", default="auto4,auto5,auto6,lightroom")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--min-face", type=float, default=0.6)
    ap.add_argument("--no-faces", action="store_true")
    a = ap.parse_args()
    pool = json.load(open(os.path.join(a.arena, "pool.json"), encoding="utf8"))
    f = os.path.join(a.arena, "priorities.json")
    old = json.load(open(f, encoding="utf8")).get("dist", {}) if os.path.exists(f) else {}
    # dist values are [distance, [hash a, hash b]] so a rebuilt picture is measured again
    dist = distances(pool, a.arena, a.size, old)
    faces = [] if a.no_faces else faces_in_db(a.min_face)
    if faces is None:
        print("no database URL: faces left empty")
        faces = []
    json.dump(dict(focus=a.focus.split(","), faces=faces, dist=dist), open(f + ".tmp", "w", encoding="utf8"))
    os.replace(f + ".tmp", f)
    print(f"{f}: {len(dist)} pictures measured, {len(faces)} with a face")


if __name__ == "__main__":
    main()
