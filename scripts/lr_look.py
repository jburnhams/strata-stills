"""An automatic version of the person's own Lightroom look, learned from their edits (scripts/lr_dataset.py): from stats of the camera's picture, predict the lightness percentiles and colour of the final they would have made, then bend our neutral picture to them.
  python scripts/lr_look.py eval     cross-validated by shoot: how far the predicted look is from the finals (lightness units, 0-100)
Version 1 is global: one tone curve on CIELAB L through the predicted percentiles (monotone, slope held between 0.25 and 4), the colour strength and cast changed by as much as the model predicts the final differs from the camera's picture. Lightroom's Highlights/Shadows are local, so a picture with a strong local edit will not be matched; that is the next step."""

import argparse
import hashlib
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import nsfw

SLIDERS = [
    ("Highlights2012", -74.0),
    ("Shadows2012", 60.0),
    ("Whites2012", 4.0),
    ("Blacks2012", -11.0),
    ("Contrast2012", 6.0),
    ("Clarity2012", 0.0),
    ("Vibrance", 20.0),
    ("PostCropVignetteAmount", -10.0),
]  # the person's Lightroom sliders that shape the picture beyond its global tone (and the typical value of each: this person's recipe is nearly the same on every picture, so a prediction is only trusted a part of the way from it)
TARGETS = D.FINAL_L + D.FINAL_C + [k for k, _ in SLIDERS]
N_LOOK = len(D.FINAL_L) + len(D.FINAL_C)
SLOPE = (0.25, 4.0)
CHROMA_SCALE = (0.8, 2.0)
CAST_MAX = 6.0
CAST_TRUST = 0.6  # tried 0.6, 0.8, 1.0 on the 31 pictures: the mean cast error from the finals is lowest at 0.6 (3.52, 3.63, 3.74): the predicted cast is noisy, following it further adds error on the usual pictures
COLOUR_TRUST = 1.15  # the saturation follows the predicted chroma (correlation 0.79 with the finals'): at 0.6 with the skin and vibrance weights the report came out 22 % less colourful than the person's finals (median C_mean ratio 0.79), and the person rated it 'too flat / dull'; 1.0 with a wider clip gives 0.9
SLIDER_TRUST = 0.7  # the sliders are only moderately predictable (R2 0.2-0.5): the prediction counts for 70 %, the person's typical value for the rest
CLARITY_GAIN = 1.0
VIBRANCE_GAIN = 1.0
FACE_L_MAX = 12.0  # bounds of the whole-picture move that puts the face where the model says the final's face is
FACE_C_RANGE = (0.7, 1.5)
FACE_ROT_MAX = 8.0
GUIDE_EPS = 0.01
SHADOW_GAIN = 25.0
HIGHLIGHT_GAIN = 25.0
VIGNETTE_GAIN = 1.0
SKIN_HUE = 54.0  # CIELAB hue (degrees) of skin-like pixels below which the picture is turned (the 10th percentile of the person's finals' medians over pictures: 53.5; their median 59.7)
SKIN_TURN = 4.0
VIB_KNEE = 55.0  # chroma at which the colour boost has fallen to half (vibrance-like: already strong colours get less)
SKIN_HOLD = 0.2  # share of the colour change withheld from skin tones


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


def predict(models, X):
    P = np.stack([m.predict(np.atleast_2d(X)) for m in models], axis=1)
    P[:, : len(D.PCTS)] = np.maximum.accumulate(P[:, : len(D.PCTS)], axis=1)  # percentiles stay in order
    return P


def _lab(bgr):
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return lab  # L 0-100, a/b about -127..127


def curve_knots(src_L, dst_L):
    """The tone curve (x, y on L 0-100) taking the picture's lightness percentiles `src_L` to the wanted `dst_L`: ends pinned at 0 and 100, made monotone and its slope held within SLOPE."""
    keep, last = [], 0.0
    for i, v in enumerate(
        src_L
    ):  # percentiles that coincide (a flat picture, clipped highlights) say nothing about the slope between them; the ends stay pinned at (0, 0) and (100, 100)
        if v > last + 0.5 and v < 99.5:
            keep.append(i)
            last = v
    x = np.concatenate([[0.0], np.asarray(src_L)[keep], [100.0]])
    y = np.concatenate([[0.0], np.asarray(dst_L)[keep], [100.0]])
    for _ in range(3):
        dy = np.clip(np.diff(y), SLOPE[0] * np.diff(x), SLOPE[1] * np.diff(x))
        y = np.concatenate([[0.0], np.cumsum(dy)])
        y = y * 100.0 / max(y[-1], 1e-6) if y[-1] > 100 else y
    return x, y


def slider(pred, name):
    """The predicted value of a Lightroom slider, a part of the way (SLIDER_TRUST) from the person's typical value to the prediction, in the -100..100 range."""
    d = dict(SLIDERS)[name]
    v = float(pred[TARGETS.index(name)]) if len(pred) > N_LOOK else d
    return float(np.clip(d + SLIDER_TRUST * (v - d), -100, 100))


def _base(L):
    """The smooth 'base' of the lightness plane (edge-preserving: a guided filter on a quarter-size copy, self-guided), for the local Shadows/Highlights: halos stay small because the base follows the edges."""
    h, w = L.shape
    k = 4
    small = cv2.resize(L / 100, (max(8, w // k), max(8, h // k)), interpolation=cv2.INTER_AREA)
    r = max(2, max(small.shape) // 30)
    mean = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mi = mean(small)
    var = mean(small * small) - mi * mi
    A = var / (var + GUIDE_EPS)
    B = mi - A * mi
    return cv2.resize(mean(A) * small + mean(B), (w, h), interpolation=cv2.INTER_LINEAR) * 100


def local_tone(L, shadows, highlights):
    """Lightroom-like Shadows and Highlights on the lightness plane `L` (0-100): the base is lifted where it is dark (shadows > 0) and pulled down where it is bright (highlights < 0), the detail on top of it is kept, so local contrast survives where the range is compressed. SHADOW_GAIN / HIGHLIGHT_GAIN are lightness units at slider 100."""
    if abs(shadows) < 1 and abs(highlights) < 1:
        return L
    base = _base(L)
    t = np.clip(base / 100, 0, 1)
    lift = SHADOW_GAIN * shadows / 100 * (1 - t) ** 2 + HIGHLIGHT_GAIN * highlights / 100 * t**2
    return np.clip(L + lift, 0, 100).astype(np.float32)


def clarity(L, amount):
    """Lightroom-like Clarity: the mid-scale detail (lightness minus its blur at 1.5 % of the long side) added back in proportion to the slider, strongest in the midtones. CLARITY_GAIN is the extra detail at slider 100."""
    if abs(amount) < 1:
        return L
    sigma = max(2.0, 0.015 * max(L.shape))
    detail = L - cv2.GaussianBlur(L, (0, 0), sigma)
    mid = 1 - ((L - 50) / 50) ** 2
    return np.clip(L + CLARITY_GAIN * amount / 100 * np.clip(mid, 0, 1) * detail, 0, 100).astype(np.float32)


def vignette(L, amount):
    """Post-crop vignette: lightness scaled by 1 + amount/100 * r^2 (r = 0 at the centre, 1 in the corners); negative darkens the corners."""
    if abs(amount) < 1:
        return L
    h, w = L.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = (((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / 2
    return (L * (1 + VIGNETTE_GAIN * amount / 100 * r2)).astype(np.float32)


def skin_rotation(L, a, b):
    """Degrees to turn the colour plane so that skin-like pixels (hue 25-75, moderate chroma, mid lightness) are not redder than SKIN_HUE (on 15 pictures with skin ours was 2-3 degrees redder than the person's finals on average): a guard against the reddish skin that a wrong guess of the light gives, the one error that looks wrong at once. Nothing when there is little skin-like area (under 3 %); bounded to SKIN_TURN and only a part (0.6) of the way."""
    h = np.degrees(np.arctan2(b, a))
    C = np.hypot(a, b)
    m = (h > 25) & (h < 75) & (C > 12) & (C < 45) & (L > 25) & (L < 90)
    if m.mean() < 0.03:
        return 0.0
    return float(
        np.clip(0.6 * (SKIN_HUE - np.median(h[m])), 0, SKIN_TURN)
    )  # one-sided: only skin redder than the person's reddest finals is turned; the finals' own spread is wide (10th-90th percentile 53.5-70.8, and a fixed target made the 65-degree pictures redder)


def _apply_look(bgr, pred, cam, colour=None):
    """`bgr` (uint8, our neutral picture) bent to the predicted look `pred` (a row of TARGETS) -> (uint8 picture, what was done). Lightness: the curve through the predicted percentiles. Colour: only the CHANGE the model predicts from OUR neutral picture (`cam`, the camera features, is no longer used for it: the neutral picture is not the camera JPEG), taken only part of the way (COLOUR_TRUST) and kept off skin and strong colours (a vibrance-like weight), because the final's mean colour is mostly the scene's (a crowd in red is not a red cast): chroma scaled by predicted / camera mean chroma, the cast moved by predicted - camera mean a/b."""
    lab = _lab(bgr)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    L = vignette(
        local_tone(L, slider(pred, "Shadows2012"), slider(pred, "Highlights2012")),
        slider(pred, "PostCropVignetteAmount"),
    )  # the vignette goes in before the percentiles are fitted: the finals' percentiles already include theirs, and the curve then makes up the global darkening it costs
    src = [float(np.percentile(L, p)) for p in D.PCTS]
    dst = list(pred[: len(D.PCTS)])
    x, y = curve_knots(np.array(src), np.array(dst))
    L2 = np.interp(L, x, y).astype(np.float32)
    L2 = clarity(L2, slider(pred, "Clarity2012"))
    if (
        colour is not None
    ):  # colour from the learned transform (scripts/lr_fit.py, predicted by scripts/lr_tmodel.py): its chroma gains by chroma (skin and the rest), cast and hue turn, with the lightness left to the steps above
        import lr_fit as FIT
        import torch

        p = torch.from_numpy(np.asarray(colour, np.float32).copy())[None]
        p[:, : len(FIT.L_KNOTS) - 1] = FIT.init_params(1, "cpu")[:, : len(FIT.L_KNOTS) - 1]
        with torch.no_grad():
            out_lab = (
                FIT.transform(torch.from_numpy(np.dstack([L2, a, b]).astype(np.float32)).permute(2, 0, 1)[None], p)[0]
                .permute(1, 2, 0)
                .numpy()
            )
        return np.clip(cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8), dict(
            curve=[(round(float(i), 1), round(float(j), 1)) for i, j in zip(x, y)],
            chroma_scale="learned",
            cast=tuple(round(float(v), 2) for v in (colour[-3] * 5, colour[-2] * 5)),
            sliders={},
            source_L=[round(v, 1) for v in src],
        )
    p = dict(zip(TARGETS, pred))
    C = np.hypot(a, b)
    n_chroma, n_a, n_b = (
        float(C.mean()),
        float(a.mean()),
        float(b.mean()),
    )  # the change is measured from OUR neutral picture's own colour, not the camera's (they differ: the camera's JPEG is not our neutral)
    scale = float(np.clip(1 + COLOUR_TRUST * (p["C_mean"] / max(n_chroma, 0.5) - 1), *CHROMA_SCALE))
    da, db = (
        float(np.clip(CAST_TRUST * (p["a_mean"] - n_a), -CAST_MAX, CAST_MAX)),
        float(np.clip(CAST_TRUST * (p["b_mean"] - n_b), -CAST_MAX, CAST_MAX)),
    )
    hue = np.degrees(np.arctan2(b, a)) % 360
    skin = np.exp(
        -0.5 * ((hue - 50) / 22) ** 2
    )  # skin sits around hue 35-65 in CIELAB: its colour is the one that looks wrong first, so it is left nearer the neutral picture
    vib = float(
        np.clip(1 + (slider(pred, "Vibrance") - dict(SLIDERS)["Vibrance"]) / 100 * VIBRANCE_GAIN, 0.6, 1.6)
    )  # the person's Vibrance slider, against their usual +20, strengthens or weakens the boost
    w = 1 / (1 + (C / VIB_KNEE) ** 2) * (1 - SKIN_HOLD * skin)
    s = 1 + (scale - 1) * w * vib  # a vibrance-like boost: strong colours (already saturated) and skin get less of it
    wl = np.clip(
        L2 / 20.0, 0, 1
    )  # no cast pushed into black (and none withheld from skin: in warm indoor light the whole picture is skin-coloured, and holding the cast there left it red)
    a2, b2 = a * s + da * wl, b * s + db * wl
    rot = skin_rotation(L2, a2, b2)
    if rot:
        c, sn = np.float32(np.cos(np.radians(rot))), np.float32(np.sin(np.radians(rot)))
        a2, b2 = a2 * c - b2 * sn, a2 * sn + b2 * c  # the hue turned a little as a whole, like a white-balance nudge
    out = cv2.cvtColor(np.dstack([L2, a2, b2]), cv2.COLOR_LAB2BGR)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8), dict(
        curve=[(round(float(i), 1), round(float(j), 1)) for i, j in zip(x, y)],
        chroma_scale=round(scale, 3),
        cast=(round(da, 2), round(db, 2)),
        sliders={
            k: round(slider(pred, k), 1)
            for k, _ in SLIDERS
            if k in ("Highlights2012", "Shadows2012", "PostCropVignetteAmount")
        },
        source_L=[round(v, 1) for v in src],
    )


def _ang(d):
    return (d + 180) % 360 - 180


def anchor_face(bgr, box, target):
    """Move the whole picture so that its face lands near `target` (L, a, b), the predicted face of the final, without a mask around the face: the three corrections are global and selective by value, as a colourist's tools are. `box` is the face (x, y, w, h as fractions). (1) Lightness: a smooth bump in the tone curve centred on the face's lightness (everything of about that lightness moves). (2) Chroma: a gain on the skin hue band (hue within about 25 degrees of the face's, not near grey). (3) Hue: a turn of that band toward the target hue. Each is bounded, and the hue only half way (it is the part the model predicts worst)."""
    lab = _lab(bgr)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    H, W = L.shape
    x, y, w, h = box
    sl = (
        slice(max(0, int((y + h / 4) * H)), int((y + 3 * h / 4) * H)),
        slice(max(0, int((x + w / 4) * W)), int((x + 3 * w / 4) * W)),
    )
    if L[sl].size == 0:
        return bgr, {}
    fL, fa, fb = float(np.median(L[sl])), float(np.median(a[sl])), float(np.median(b[sl]))
    fC = float(np.hypot(fa, fb))
    fh = float(np.degrees(np.arctan2(fb, fa)))
    tL, ta, tb = target
    tC = float(np.hypot(ta, tb))
    th = float(np.degrees(np.arctan2(tb, ta)))
    dL = float(np.clip(tL - fL, -FACE_L_MAX, FACE_L_MAX))
    gC = float(np.clip(tC / max(fC, 1.0), FACE_C_RANGE[0], FACE_C_RANGE[1]))
    rot = float(np.clip(0.5 * _ang(th - fh), -FACE_ROT_MAX, FACE_ROT_MAX))
    C = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    band = np.exp(-0.5 * (_ang(hue - fh) / 25) ** 2) * np.clip(C / 8, 0, 1)
    L2 = np.clip(L + dL * np.exp(-0.5 * ((L - fL) / 22) ** 2), 0, 100)
    C2 = C * (1 + (gC - 1) * band)
    h2 = np.radians(hue + rot * band)
    out = cv2.cvtColor(np.dstack([L2, C2 * np.cos(h2), C2 * np.sin(h2)]).astype(np.float32), cv2.COLOR_LAB2BGR)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8), dict(
        face_dL=round(dL, 1), face_chroma_gain=round(gC, 2), face_hue_turn=round(rot, 1)
    )


def _on_gpu():
    """The look runs on the device (scripts/lr_gpu.py) when there is one; STRATASTILLS_GPU=0 (or STRATASTILLS_DEVICE=cpu) keeps the NumPy/OpenCV code above, the reference."""
    import lr_gpu

    return lr_gpu.use()


def apply_look(bgr, pred, cam, colour=None, face=None, gpu=None):
    """`_apply_look`, then, when the picture has a face (`face` = dict(box=(x, y, w, h) as fractions, target=(L, a, b) predicted for the final)), the whole picture moved so that the face lands near its target (`anchor_face`). On the device when there is one (`gpu` True/False forces it)."""
    if gpu is None:
        gpu = _on_gpu()
    if gpu:
        import lr_gpu as G

        out, did = G._apply_look(bgr, pred, cam, colour)
        if face is not None:
            out, more = G.anchor_face(out, face["box"], face["target"])
            did = dict(did, **more)
        return out, did
    out, did = _apply_look(bgr, pred, cam, colour)
    if face is not None:
        out, more = anchor_face(out, face["box"], face["target"])
        did = dict(did, **more)
    return out, did


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
