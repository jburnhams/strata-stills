"""Check that the look stage sees a picture as the models were trained to: for arena pictures, the feature row stratastills.look makes from
the raw (the camera's preview cut as the final was, exiftool, CLIP through strata-models) against the dataset's row (scripts/lr_look.py load),
block by block, and what the models predict from each.
  python scripts/check_look_parity.py [--n 20]
Prints the largest differences per block and the change in the predicted look (lightness percentiles, colour) between the two rows."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import build_arena as BA
import lr_dataset as D
import lr_look as LK

from stratastills import look, settings


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    rows, X, names, Y, idx, G = LK.load()
    pool = json.load(open(os.path.join(settings.arena_dir(), "pool.json")))
    by_sid = {}
    for i, k in enumerate(idx):
        r = rows[k]
        by_sid[f"{os.path.basename(os.path.dirname(r['raw']))}_{os.path.splitext(os.path.basename(r['raw']))[0]}"] = i
    pics = [s for s in pool if s in by_sid][: a.n]
    b = look.bundle()
    blocks = dict(
        base=slice(0, len(b.manifest["base_names"])),
        meta=slice(len(b.manifest["base_names"]), len(b.manifest["base_names"]) + b.manifest["n_meta"]),
        clip=slice(len(names) - b.manifest["n_clip"], len(names)),
    )
    worst = dict.fromkeys(blocks, (0.0, ""))
    dpred = []
    for sid in pics:
        i = by_sid[sid]
        r = rows[idx[i]]
        ex = D.exif_of([r["raw"]])[r["raw"]]
        x, fl, clip = BA.arena_inputs(r["raw"], r, ex.get("Orientation", 1), b)
        for k, sl in blocks.items():
            d = np.abs(np.nan_to_num(x[sl], nan=-999) - np.nan_to_num(X[i, sl], nan=-999))
            j = int(d.argmax())
            if d[j] > worst[k][0]:
                worst[k] = (float(d[j]), f"{sid} {names[sl][j]}: {x[sl][j]:.4g} vs {X[i, sl][j]:.4g}")
        p1, p0 = look.predictions(x, fl, b)["pred"], look.predictions(X[i], [], b)["pred"]
        dpred.append(np.abs(p1 - p0)[: len(D.PCTS) + 1 + 6])
        print(
            f"{sid}: faces {len(fl)}, clip {clip}, look change |dL| {np.abs(p1 - p0)[: len(D.PCTS)].mean():.2f}",
            flush=True,
        )
    for k, (v, where) in worst.items():
        print(f"{k:5s} largest difference {v:.4g}  {where}")
    print(
        f"predicted look, mean |change| over {len(pics)} pictures: lightness {np.mean([d[: len(D.PCTS)].mean() for d in dpred]):.3f}, colour {np.mean([d[len(D.PCTS) + 1 :].mean() for d in dpred]):.3f}"
    )


if __name__ == "__main__":
    main()
