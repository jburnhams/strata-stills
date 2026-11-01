"""An automatic version of the person's own Lightroom look, learned from their edits (scripts/lr_dataset.py): from stats of the camera's picture, predict the lightness percentiles and colour of the final they would have made, then bend our neutral picture to them.
  python scripts/lr_look.py eval     cross-validated by shoot: how far the predicted look is from the finals (lightness units, 0-100)
Version 1 is global: one tone curve on CIELAB L through the predicted percentiles (monotone, slope held between 0.25 and 4), the colour strength and cast changed by as much as the model predicts the final differs from the camera's picture. Lightroom's Highlights/Shadows are local, so a picture with a strong local edit will not be matched; that is the next step."""
import argparse, os, sys, hashlib
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D

TARGETS = D.FINAL_L + D.FINAL_C
SLOPE = (0.25, 4.0)
CHROMA_SCALE = (0.8, 1.4)
CAST_MAX = 4.0
COLOUR_TRUST = 0.6      # the model's colour change explains little of the spread (correlation about 0.55-0.8): go only part of the way, so that a wrong guess costs little at the extremes (indoor, tungsten, skin) where it shows
SKIN_HOLD = 0.7         # share of the colour change withheld from skin tones


def test_shoot(shoot, share=0.1):
    """A fixed share of the shoots (folders of finals) held out from training, chosen by a hash of the name so that it never changes between runs."""
    return int(hashlib.sha1(shoot.encode()).hexdigest()[:8], 16) / 0xffffffff < share


def _model():
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.6, min_child_weight=5, n_jobs=os.cpu_count(), tree_method='hist')


def train(X, Y):
    """One model per target (TARGETS), on the features X."""
    return [_model().fit(X, Y[:, j]) for j in range(Y.shape[1])]


def predict(models, X):
    P = np.stack([m.predict(np.atleast_2d(X)) for m in models], axis=1)
    P[:, :len(D.PCTS)] = np.maximum.accumulate(P[:, :len(D.PCTS)], axis=1)          # percentiles stay in order
    return P


def _lab(bgr):
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB); return lab                                                  # L 0-100, a/b about -127..127


def curve_knots(src_L, dst_L):
    """The tone curve (x, y on L 0-100) taking the picture's lightness percentiles `src_L` to the wanted `dst_L`: ends pinned at 0 and 100, made monotone and its slope held within SLOPE."""
    keep, last = [], 0.0
    for i, v in enumerate(src_L):                                                                                                  # percentiles that coincide (a flat picture, clipped highlights) say nothing about the slope between them; the ends stay pinned at (0, 0) and (100, 100)
        if v > last + 0.5 and v < 99.5: keep.append(i); last = v
    x = np.concatenate([[0.0], np.asarray(src_L)[keep], [100.0]]); y = np.concatenate([[0.0], np.asarray(dst_L)[keep], [100.0]])
    for _ in range(3):
        dy = np.clip(np.diff(y), SLOPE[0] * np.diff(x), SLOPE[1] * np.diff(x)); y = np.concatenate([[0.0], np.cumsum(dy)])
        y = y * 100.0 / max(y[-1], 1e-6) if y[-1] > 100 else y
    return x, y


def apply_look(bgr, pred, cam):
    """`bgr` (uint8, our neutral picture) bent to the predicted look `pred` (a row of TARGETS) -> (uint8 picture, what was done). Lightness: the curve through the predicted percentiles. Colour: only the CHANGE the model predicts from OUR neutral picture (`cam`, the camera features, is no longer used for it: the neutral picture is not the camera JPEG), taken only part of the way (COLOUR_TRUST) and kept off skin and strong colours (a vibrance-like weight), because the final's mean colour is mostly the scene's (a crowd in red is not a red cast): chroma scaled by predicted / camera mean chroma, the cast moved by predicted - camera mean a/b."""
    lab = _lab(bgr); L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    src = [float(np.percentile(L, p)) for p in D.PCTS]; dst = list(pred[:len(D.PCTS)])
    x, y = curve_knots(np.array(src), np.array(dst)); L2 = np.interp(L, x, y).astype(np.float32)
    p = dict(zip(TARGETS, pred)); C = np.hypot(a, b); n_chroma, n_a, n_b = float(C.mean()), float(a.mean()), float(b.mean())                          # the change is measured from OUR neutral picture's own colour, not the camera's (they differ: the camera's JPEG is not our neutral)
    scale = float(np.clip(1 + COLOUR_TRUST * (p['C_mean'] / max(n_chroma, 0.5) - 1), *CHROMA_SCALE))
    da, db = (float(np.clip(COLOUR_TRUST * (p['a_mean'] - n_a), -CAST_MAX, CAST_MAX)), float(np.clip(COLOUR_TRUST * (p['b_mean'] - n_b), -CAST_MAX, CAST_MAX)))
    hue = np.degrees(np.arctan2(b, a)) % 360; skin = np.exp(-0.5 * ((hue - 50) / 22) ** 2)                                                      # skin sits around hue 35-65 in CIELAB: its colour is the one that looks wrong first, so it is left nearer the neutral picture
    w = 1 / (1 + (C / 45) ** 2) * (1 - SKIN_HOLD * skin); s = 1 + (scale - 1) * w                                                             # a vibrance-like boost: strong colours (already saturated) and skin get less of it
    wl = np.clip(L2 / 20.0, 0, 1) * (1 - SKIN_HOLD * skin)                                                                                  # no cast pushed into black, or into skin
    out = cv2.cvtColor(np.dstack([L2, a * s + da * wl, b * s + db * wl]), cv2.COLOR_LAB2BGR)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8), dict(curve=[(round(float(i), 1), round(float(j), 1)) for i, j in zip(x, y)], chroma_scale=round(scale, 3), cast=(round(da, 2), round(db, 2)), source_L=[round(v, 1) for v in src])


def load():
    rows = D.load_catalogue(); X, names, F, C, idx = D.load_features(); Y = np.hstack([F, C]); G = np.array([os.path.dirname(rows[i]['final']) for i in idx])
    return rows, X, names, Y, idx, G


def evaluate(a):
    from sklearn.model_selection import GroupKFold
    rows, X, names, Y, idx, G = load(); P = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y[:, 0], G): P[te] = predict(train(X[tr], Y[tr]), X[te])
    cam = X[:, [names.index(f'L_p{p}') for p in D.PCTS]]
    print(f'{len(X)} pairs, {len(set(G))} shoots; mean error (cross-validated by shoot):')
    for j, t in enumerate(TARGETS): print(f'  {t:8s} model {np.abs(P[:, j] - Y[:, j]).mean():5.2f}' + (f'   camera {np.abs(cam[:, j] - Y[:, j]).mean():5.2f}' if j < len(D.PCTS) else ''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('step', choices=['eval']); a = ap.parse_args(); evaluate(a)
