"""An automatic version of the person's own Lightroom look, learned from their edits (scripts/lr_dataset.py): from stats of the camera's picture, predict the lightness percentiles and colour of the final they would have made, then bend our neutral picture to them.
  python scripts/lr_look.py eval     cross-validated by shoot: how far the predicted look is from the finals (lightness units, 0-100)
Version 1 is global: one tone curve on CIELAB L through the predicted percentiles (monotone, slope held between 0.25 and 4), the colour strength and cast changed by as much as the model predicts the final differs from the camera's picture. Lightroom's Highlights/Shadows are local, so a picture with a strong local edit will not be matched; that is the next step."""
import argparse, os, sys, hashlib
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D

TARGETS = D.FINAL_L + D.FINAL_C
SLOPE = (0.25, 4.0)
CHROMA_SCALE = (0.7, 1.6)
CAST_MAX = 6.0


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
    x = np.concatenate([[0.0], src_L, [100.0]]); y = np.concatenate([[0.0], dst_L, [100.0]])
    keep = np.concatenate([[True], np.diff(x) > 0.5]); x, y = x[keep], y[keep]                                                    # percentiles that coincide (a flat picture) say nothing about the slope between them
    for _ in range(3):
        dy = np.clip(np.diff(y), SLOPE[0] * np.diff(x), SLOPE[1] * np.diff(x)); y = np.concatenate([[0.0], np.cumsum(dy)])
        y = y * 100.0 / max(y[-1], 1e-6) if y[-1] > 100 else y
    return x, y


def apply_look(bgr, pred, cam):
    """`bgr` (uint8, our neutral picture) bent to the predicted look `pred` (a row of TARGETS) -> (uint8 picture, what was done). Lightness: the curve through the predicted percentiles. Colour: only the CHANGE the model predicts from the camera's picture `cam` (its features: chroma_mean, a_mean, b_mean), because the final's mean colour is mostly the scene's (a crowd in red is not a red cast): chroma scaled by predicted / camera mean chroma, the cast moved by predicted - camera mean a/b."""
    lab = _lab(bgr); L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    src = [float(np.percentile(L, p)) for p in D.PCTS]; dst = list(pred[:len(D.PCTS)])
    x, y = curve_knots(np.array(src), np.array(dst)); L2 = np.interp(L, x, y).astype(np.float32)
    p = dict(zip(TARGETS, pred)); scale = float(np.clip(p['C_mean'] / max(cam['chroma_mean'], 0.5), *CHROMA_SCALE))
    da, db = (float(np.clip(p['a_mean'] - cam['a_mean'], -CAST_MAX, CAST_MAX)), float(np.clip(p['b_mean'] - cam['b_mean'], -CAST_MAX, CAST_MAX)))
    w = np.clip(L2 / 20.0, 0, 1)                                                                                                   # no cast pushed into black
    out = cv2.cvtColor(np.dstack([L2, a * scale + da * w, b * scale + db * w]), cv2.COLOR_LAB2BGR)
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
