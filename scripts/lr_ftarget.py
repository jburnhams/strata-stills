"""Can the colour of the faces in the person's final be predicted? (the face half of the look model; faces from scripts/lr_faces.py)
  python scripts/lr_ftarget.py     held-out shoots: the error of the largest face's lightness / a / b / hue / chroma in the final, for: the camera preview's face as it is, its typical change, and the model (inputs: everything in lr_look.load plus the preview's face colour, face size and count).
The model predicts the CHANGE from the preview's face (final minus preview), so that a picture whose face is already right is left alone."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D, lr_look as K

FACE_FEATS = ['face_n', 'face_area', 'face_L', 'face_a', 'face_b', 'face_C', 'face_hue', 'face_score']


def face_table(idx):
    """Per row of `idx` (catalogue rows): the face inputs (FACE_FEATS, NaN without a face) and the final's face (L, a, b) of the largest face."""
    faces = json.load(open(os.path.join(D.OUT, 'faces.json'))); F = np.full((len(idx), len(FACE_FEATS)), np.nan, np.float32); T = np.full((len(idx), 3), np.nan, np.float32)
    for r, i in enumerate(idx):
        fl = faces.get(str(int(i)))
        if not fl: continue
        f = fl[0]; L, a, b = f['cam']; F[r] = [len(fl), f['box'][2] * f['box'][3], L, a, b, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360, f['score']]; T[r] = f['fin']
    return F, T


def main():
    import xgboost as xgb
    rows, X, names, Y, idx, G = K.load(); F, T = face_table(idx); has = ~np.isnan(T[:, 0]); test = np.array([K.test_shoot(g) for g in G])
    XF = np.hstack([X, F]); tr, te = has & ~test, has & test; print(f'{has.sum()} pictures with a face of {len(X)}; {tr.sum()} to train on, {te.sum()} held out', flush=True)
    cam = F[:, 2:5]; delta = T - cam; P = np.zeros((len(X), 3), np.float32)
    for j in range(3):
        m = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.06, subsample=0.8, colsample_bytree=0.6, min_child_weight=5, n_jobs=os.cpu_count(), tree_method='hist').fit(XF[tr], delta[tr, j]); P[te, j] = m.predict(XF[te])
    typical = np.median(delta[tr], 0)
    def err(pred):
        d = pred[te] - T[te]; hue = lambda v: np.degrees(np.arctan2(v[:, 2], v[:, 1])); dh = (hue(pred[te]) - hue(T[te]) + 180) % 360 - 180; dc = np.hypot(pred[te][:, 1], pred[te][:, 2]) - np.hypot(T[te][:, 1], T[te][:, 2])
        return np.abs(d[:, 0]).mean(), np.abs(d[:, 1]).mean(), np.abs(d[:, 2]).mean(), np.abs(dh).mean(), np.abs(dc).mean(), np.sqrt((d ** 2).sum(1)).mean()
    print(f'{"largest face of the final":36s}{"L":>7s}{"a":>7s}{"b":>7s}{"hue":>7s}{"chroma":>8s}{"dE76":>7s}')
    for nm, pred in (('preview face as it is', cam), ('+ typical change', cam + typical), ('+ predicted change', cam + P)): print(f'{nm:36s}' + ''.join(f'{v:7.2f}' for v in err(pred)))
    print('the final face: lightness median %.1f (p10 %.1f, p90 %.1f), hue median %.1f (p10 %.1f, p90 %.1f), chroma median %.1f' % (np.median(T[te, 0]), *np.percentile(T[te, 0], [10, 90]), np.median(np.degrees(np.arctan2(T[te, 2], T[te, 1]))), *np.percentile(np.degrees(np.arctan2(T[te, 2], T[te, 1])), [10, 90]), np.median(np.hypot(T[te, 1], T[te, 2]))))


if __name__ == '__main__':
    main()
