"""Which arena pairs are worth a vote: writes `<arena dir>/priorities.json`, which the arena's picker reads (arena.load_priorities / pair_weight). Nothing is removed from the arena: pairs are drawn at random with a weight, so clearly different pairs of the variants we care about, on pictures with a face, come first, and near-identical pairs are rare.
  python scripts/arena_priorities.py [--arena DIR] [--size 96] [--no-faces]
- dist: per picture and pair of variants, the mean CIELAB distance of the two pictures at --size px (the pictures are laid on the same frame; a pair that differs in size is compared after resizing). For a picture with a face the number is (whole + 1.5 x face region) / 2.5, the region being every face box of the final's frame padded by 30 % (the arena pictures are cropped a little at the edges). Pictures whose files are unchanged since the last run are not measured again.
- faces: the pictures with a face of score >= --min-face in schema `photos` (detection label 'face', needs the database URL and omarchy_db), mapped by arena_pic.photo_id; their boxes feed dist. Statues and the like count; the faces2.pkl real-face scores are not in the database.
- interest per version is `arena.INTEREST` (edit there, or add an 'interest' object to the file by hand; a rerun keeps nothing but dist and faces).
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


def region(boxes, size, pad=0.3):
    """A boolean size x size mask of the face boxes (x, y, w, h as fractions) padded by `pad` of their size."""
    m = np.zeros((size, size), bool)
    for x, y, w, h in boxes:
        x0, x1 = max(0, int((x - pad * w) * size)), min(size, int(np.ceil((x + w + pad * w) * size)))
        y0, y1 = max(0, int((y - pad * h) * size)), min(size, int(np.ceil((y + h + pad * h) * size)))
        m[y0:y1, x0:x1] = True
    return m


def distances(pool, root, size, old, faces):
    out = {}
    for n, (pic, vs) in enumerate(pool.items(), 1):
        names = sorted(vs)
        key = {v: vs[v]["hash"] for v in names}
        prev = old.get(pic, {})
        labs = {}
        res = {}
        mask = region(faces[pic], size) if faces.get(pic) else None
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if key[a] == key[b]:
                    continue
                k = f"{a}|{b}"
                if (
                    k in prev
                    and prev[k][1] == [key[a], key[b]]
                    and (len(prev[k]) > 2 and prev[k][2]) == (mask is not None)
                ):
                    res[k] = prev[k]
                    continue
                for v in (a, b):
                    if v not in labs:
                        labs[v] = lab(os.path.join(root, vs[v]["file"]), size)
                if labs[a] is None or labs[b] is None:
                    continue
                e = np.sqrt(((labs[a] - labs[b]) ** 2).sum(-1))
                d = float(e.mean())
                if mask is not None and mask.any():
                    d = (d + 1.5 * float(e[mask].mean())) / 2.5
                res[k] = [d, [key[a], key[b]], bool(mask is not None)]
        out[pic] = res
        if n % 100 == 0:
            print(f"{n}/{len(pool)} pictures", flush=True)
    return out


def faces_in_db(min_score):
    """-> {picture: [(x, y, w, h) of its faces]} or None without a database URL. Needs OMARCHY_DB_LIVE=1 on omarchy (the shared-database guard)."""
    url = settings.database_url()
    if not url:
        return None
    from omarchy_db import connect, session
    from sqlalchemy import select

    from stratastills.db.models import Analysis, ArenaPic, Detection

    with session(connect(url)) as s:
        q = (
            select(ArenaPic.name, Detection.x, Detection.y, Detection.w, Detection.h)
            .join(Analysis, Analysis.photo_id == ArenaPic.photo_id)
            .join(Detection, Detection.analysis_id == Analysis.id)
            .where(Detection.label == "face", Detection.score >= min_score)
        )
        out = {}
        for name, x, y, w, h in s.execute(q):
            out.setdefault(name, []).append((x, y, w, h))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arena", default=settings.arena_dir())
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--min-face", type=float, default=0.6)
    ap.add_argument("--no-faces", action="store_true")
    a = ap.parse_args()
    pool = json.load(open(os.path.join(a.arena, "pool.json"), encoding="utf8"))
    f = os.path.join(a.arena, "priorities.json")
    old = json.load(open(f, encoding="utf8")).get("dist", {}) if os.path.exists(f) else {}
    # dist values are [distance, [hash a, hash b], with a face region] so a rebuilt picture is measured again
    faces = {} if a.no_faces else faces_in_db(a.min_face)
    if faces is None:
        print("no database URL: faces left empty")
        faces = {}
    dist = distances(pool, a.arena, a.size, old, faces)
    json.dump(dict(faces=sorted(faces), dist=dist), open(f + ".tmp", "w", encoding="utf8"))
    os.replace(f + ".tmp", f)
    print(f"{f}: {len(dist)} pictures measured, {len(faces)} with a face")


if __name__ == "__main__":
    main()
