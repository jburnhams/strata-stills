"""Learn to predict the fitted transform (scripts/lr_fit.py) of a picture from its features (scripts/lr_look.py `load`: preview statistics, EXIF and the rest of the metadata), and judge it on held-out shoots by applying the predicted transform to the camera previews and measuring the distance to the person's finals.
  python scripts/lr_tmodel.py eval [--hard-weight W]    held-out report: identity, the typical transform, the prediction and the fit ceiling, over all pairs and over the hard cases (indoor / warm light / dark / high ISO / skin)
The model is one XGBoost regressor per transform parameter, trained on the shoots not held out (lr_look.test_shoot), with the hard cases weighted up (they are where a wrong guess looks wrong)."""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D
import lr_fit as FIT
import lr_look as K


def data():
    """X (features, aligned with the thumbs), the fitted params, the pair indices and the group (shoot) of each."""
    z = np.load(os.path.join(D.OUT, "fit.npz"))
    rows, X, names, Y, idx, G = K.load()
    pos = {int(i): k for k, i in enumerate(idx)}
    keep = [k for k, i in enumerate(z["idx"]) if int(i) in pos]
    sel = np.array([pos[int(z["idx"][k])] for k in keep])
    return X[sel], names, z["params"][keep], np.array(keep), G[sel], Y[sel]


def hard_mask(X, names):
    """The cases where a wrong guess shows: little light (low brightness value / high ISO), warm light (red/blue gain), or a dark picture."""
    col = lambda n: X[:, names.index(n)]
    iso = col("log_iso")
    L50 = col("L_p50")
    bb = col("meta:Composite:BlueBalance") if "meta:Composite:BlueBalance" in names else np.zeros(len(X))
    return (
        (iso >= np.nanpercentile(iso, 80))
        | (np.nanpercentile(L50, 15) >= L50)
        | (np.nan_to_num(bb) >= np.nanpercentile(bb, 85))
    )


def train(X, P, w, n_est=300):
    import xgboost as xgb

    return [
        xgb.XGBRegressor(
            n_estimators=n_est,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=5,
            n_jobs=os.cpu_count(),
            tree_method="hist",
        ).fit(X, P[:, j], sample_weight=w)
        for j in range(P.shape[1])
    ]


def train_cached(hard_weight=3.0):
    """The models trained on the shoots not held out, kept in out/cache/ under a hash of the data they were trained on."""
    import hashlib
    import pickle

    X, names, P, keep, G, Y = data()
    test = np.array([K.test_shoot(g) for g in G])
    key = hashlib.sha1(repr((X.shape, float(np.nansum(X)), float(P.sum()), hard_weight, names)).encode()).hexdigest()[
        :16
    ]
    f = os.path.join(D.ROOT, "out", "cache", f"lr-tmodel-{key}.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    models = train(X[~test], P[~test], np.where(hard_mask(X, names), hard_weight, 1.0)[~test])
    os.makedirs(os.path.dirname(f), exist_ok=True)
    pickle.dump(models, open(f, "wb"))
    return models


def apply_params(bgr, p):
    """The transform with parameters `p` applied to a picture (uint8 BGR, any size) -> uint8 BGR. On the CPU when the picture is too big for the card."""
    dev = "cuda" if torch.cuda.is_available() and bgr.shape[0] * bgr.shape[1] < 4_000_000 else "cpu"
    with torch.no_grad():
        out = (
            FIT.transform(FIT.to_lab(bgr[None]).to(dev), torch.from_numpy(np.asarray(p, np.float32))[None].to(dev))[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
    import cv2

    return np.clip(cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)


def predict(models, X):
    return np.stack([m.predict(X) for m in models], 1).astype(np.float32)


def evaluate(a):
    X, names, P, keep, G, Y = data()
    test = np.array([K.test_shoot(g) for g in G])
    hard = hard_mask(X, names)
    w = np.where(hard, a.hard_weight, 1.0)[~test]
    print(
        f"{len(X)} pairs, {test.sum()} held out ({(hard & test).sum()} hard); training on {(~test).sum()}", flush=True
    )
    models = train(X[~test], P[~test], w)
    pred = predict(models, X[test])
    typical = np.median(P[~test], 0)
    cam, fin, _ = FIT.load_pairs()
    cam, fin = cam[keep][test], fin[keep][test]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {
        "camera preview as it is": None,
        "typical transform (median params)": np.tile(typical, (test.sum(), 1)),
        "predicted transform": pred,
        "fitted per pair (ceiling)": P[test],
    }
    res = {}
    for nm, p in rows.items():
        d = []
        for s in range(0, test.sum(), 128):
            c, f = FIT.to_lab(cam[s : s + 128]).to(dev), FIT.to_lab(fin[s : s + 128]).to(dev)
            with torch.no_grad():
                out = c if p is None else FIT.transform(c, torch.from_numpy(p[s : s + 128]).to(dev))
            d.append(torch.stack(FIT.distance(out, f), 1).cpu().numpy())
        res[nm] = np.concatenate(d)
    if a.old:  # the earlier method (scripts/lr_look.py: whole-picture averages and hand-set steps) on the same previews, for comparison
        import cv2

        old = K.train_cached(X[~test], Y[~test])
        po = K.predict(old, X[test])
        d = []
        for i in range(test.sum()):
            out, _ = K.apply_look(cam[i], po[i], dict(zip(names, X[test][i])))
            c = FIT.to_lab(cam[i : i + 1]).to(dev)
            with torch.no_grad():
                d.append(
                    torch.stack(FIT.distance(FIT.to_lab(out[None]).to(dev), FIT.to_lab(fin[i : i + 1]).to(dev)), 1)
                    .cpu()
                    .numpy()[0]
                )
        res["earlier method (apply_look)"] = np.array(d)
        cv2.setNumThreads(1)
    h = hard[test]
    print(f"{'':36s}{'all: dL / colour':>22s}{'hard cases':>20s}{'worst 10% (colour)':>22s}")
    for nm, d in res.items():
        print(
            f"{nm:36s}{d[:, 0].mean():10.2f} /{d[:, 1].mean():6.2f}{d[h, 0].mean():12.2f} /{d[h, 1].mean():6.2f}{np.percentile(d[:, 1], 90):18.2f}"
        )
    np.savez(os.path.join(D.OUT, "tmodel_eval.npz"), test=np.where(test)[0], pred=pred)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["eval"])
    ap.add_argument("--hard-weight", type=float, default=3.0)
    ap.add_argument(
        "--old", action="store_true", help="also run the earlier method (lr_look.apply_look) for comparison"
    )
    a = ap.parse_args()
    evaluate(a)
