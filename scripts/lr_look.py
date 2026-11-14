"""An automatic version of the person's own Lightroom look, learned from their edits (scripts/lr_dataset.py): from stats of the camera's picture, predict the lightness percentiles and colour of the final they would have made, then bend our neutral picture to them.
  python scripts/lr_look.py eval     cross-validated by shoot: how far the predicted look is from the finals (lightness units, 0-100)
Version 1 is global: one tone curve on CIELAB L through the predicted percentiles (monotone, slope held between 0.25 and 4), the colour strength and cast changed by as much as the model predicts the final differs from the camera's picture. Lightroom's Highlights/Shadows are local, so a picture with a strong local edit will not be matched; that is the next step."""

import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import nsfw
from stratastills.look.apply import (  # noqa: F401  (moved: the pipeline's look stage applies the look)
    CAST_MAX,
    CAST_TRUST,
    CHROMA_SCALE,
    CLARITY_GAIN,
    COLOUR_TRUST,
    FACE_C_RANGE,
    FACE_L_MAX,
    FACE_ROT_MAX,
    GUIDE_EPS,
    HIGHLIGHT_GAIN,
    N_LOOK,
    SHADOW_GAIN,
    SKIN_HOLD,
    SKIN_HUE,
    SKIN_TURN,
    SLIDER_TRUST,
    SLIDERS,
    SLOPE,
    TARGETS,
    VIB_KNEE,
    VIBRANCE_GAIN,
    VIGNETTE_GAIN,
    _ang,
    _apply_look,
    _base,
    _lab,
    _on_gpu,
    anchor_face,
    apply_look,
    clarity,
    curve_knots,
    local_tone,
    predict,
    skin_rotation,
    slider,
    vignette,
)


def test_shoot(shoot, share=0.1):
    """A fixed share of the shoots (folders of finals) held out from training, chosen by a hash of the name so that it never changes between runs."""
    return int(hashlib.sha1(shoot.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def _model():
    import xgboost as xgb

    return xgb.XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=5,
        n_jobs=os.cpu_count(),
        tree_method="hist",
    )


def train(X, Y):
    """One model per target (TARGETS), on the features X."""
    return [_model().fit(X, Y[:, j]) for j in range(Y.shape[1])]


def train_cached(X, Y):
    """`train`, kept in out/cache/ under a hash of the data and of the model's settings, so that the report is quick when only the way the prediction is used changed."""
    import pickle

    m = _model()
    key = hashlib.sha1(
        repr(
            (
                X.shape,
                float(np.nansum(X)),
                float(Y.sum()),
                sorted(m.get_params().items(), key=lambda kv: kv[0]),
                TARGETS,
            )
        ).encode()
    ).hexdigest()[:16]
    f = os.path.join(D.ROOT, "out", "cache", f"lr-look-{key}.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    models = train(X, Y)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    pickle.dump(models, open(f, "wb"))
    return models


_loaded = {}  # (OUT, STRATASTILLS_NO_CLIP) -> what _load read, once per process: the arena script, the colour model and the face model each call load()


def load():
    """(rows, X, names, Y, idx, G) of the dataset (copies: a caller may change them). The first call in a process reads it, the next ones copy."""
    key = (D.OUT, os.environ.get("STRATASTILLS_NO_CLIP"))
    if key not in _loaded:
        _loaded[key] = _load()
    rows, X, names, Y, idx, G = _loaded[key]
    return list(rows), X.copy(), list(names), Y.copy(), idx.copy(), G.copy()


def _flagged_mask(paths):
    """True for each picture the nudity filter flags, as `nsfw.flagged(path) is not None` says, without nsfw.flagged's two reads of the settings file per picture (21 000 pictures, three loads per run: minutes; the finals are on a CIFS share where a stat takes 5 ms)."""
    if not nsfw.enabled():
        return np.zeros(len(paths), bool)
    thr = nsfw.threshold()
    allowed = nsfw._allowed()
    nsfw._load()

    def one(p):
        s, _ = nsfw.score(p)
        return s >= thr and nsfw._key(p) not in allowed

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(32) as ex:
        return np.array(
            list(ex.map(one, paths)), bool
        )  # the finals are on a network share (a stat is about 5 ms there): many at a time


def _load():
    rows = D.load_catalogue()
    X, names, F, C, idx = D.load_features()
    G = np.array([os.path.dirname(rows[i]["final"]) for i in idx])
    f = os.path.join(D.OUT, "meta.npz")
    if os.path.exists(f):
        z = np.load(f)
        X = np.hstack([X, z["M"][idx]])
        names = (
            names + ["meta:" + str(n) for n in z["names"]]
        )  # the rest of the raw's metadata (scripts/lr_dataset.py meta): camera, lens, modes, the as-shot white balance, the time of day and year, ...
    f = os.path.join(D.OUT, "clip.npz")
    if os.path.exists(f) and not os.environ.get("STRATASTILLS_NO_CLIP"):
        z = np.load(f)
        pca, labels = z["pca"], z["labels"]
        pos = {int(i): k for k, i in enumerate(z["idx"])}
        M = np.full(
            (len(idx), 32 + labels.shape[1]), np.nan, np.float32
        )  # each z[...] reads and checks the whole member again: 42 000 reads in the loop below took a minute
        for r, i in enumerate(idx):
            k = pos.get(int(i))
            if k is not None:
                M[r] = np.concatenate([pca[k], labels[k]])
        X = np.hstack([X, M])
        names = (
            names + ["clip:" + str(n) for n in z["pca_names"]] + ["clip:" + str(n) for n in z["label_names"]]
        )  # what CLIP sees in the preview (scripts/lr_clip.py): scene labels and the principal components of the embedding
    S = np.array([[D.num(rows[i].get(k), d) for k, d in SLIDERS] for i in idx])
    S = np.where(np.isnan(S), [d for _, d in SLIDERS], S)  # a setting the file does not have is the typical one
    Y = np.hstack([F, C, S])
    ok = ~_flagged_mask(
        [rows[i]["final"] for i in idx]
    )  # the nudity filter, applied to the dataset as it is read, so a datasets built before it (or with a higher threshold) feeds no model, report or arena either
    if ok.all():
        return rows, X, names, Y, idx, G
    return rows, X[ok], names, Y[ok], idx[ok], G[ok]


def evaluate(a):
    from sklearn.model_selection import GroupKFold

    rows, X, names, Y, idx, G = load()
    P = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y[:, 0], G):
        P[te] = predict(train(X[tr], Y[tr]), X[te])
    cam = X[:, [names.index(f"L_p{p}") for p in D.PCTS]]
    print(f"{len(X)} pairs, {len(set(G))} shoots; mean error (cross-validated by shoot):")
    for j, t in enumerate(TARGETS):
        print(
            f"  {t:8s} model {np.abs(P[:, j] - Y[:, j]).mean():5.2f}"
            + (f"   camera {np.abs(cam[:, j] - Y[:, j]).mean():5.2f}" if j < len(D.PCTS) else "")
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["eval"])
    a = ap.parse_args()
    evaluate(a)
