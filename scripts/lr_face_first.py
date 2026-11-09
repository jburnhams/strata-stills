"""Face first: after the automatic edit, move the picture so the main face's skin lands near the colour the person gives skin, trading a
little whole-picture accuracy for a face that looks right (people see a wrong face at once).

Faces: candidates/personal/faces2.pkl {catalogue row: [{box (x, y, w, h fractions of the final's frame), lm (68 landmarks, fractions),
pose, score, real (P(real person) from the strata-models face-kind role), kind, by (which look found it)}]}, from the re-detection of
2026-11-09 (plain + padded look, harder looks where Sonnet/CLIP expected more faces). Only real faces (real >= 0.5) are used, so a statue
or a poster never sets the skin colour.

Skin: the face outline from the landmarks (jaw and a strip of forehead) minus eyes, brows and mouth, eroded; its median CIELAB.

Target: XGBoost (out/cache/lr-skin-target.pkl, one model per L/a/b) predicts the final's skin minus the edited picture's skin, from the
picture features (lr_look.load), the edited and the neutral skin colour, the face size and pose; trained on 3,829 training faces with
the network's OUT-OF-FOLD edits (two networks, each trained on half the training shoots), so it learns what the edit actually gets wrong.

Step: the whole picture's a/b moved GLOBAL of the way to the target (kept off the blacks), then the rest of the way LOCAL of it on the
face only, through a mask feathered over a quarter of the face's size. L is left alone.
Held out (861 faces, 512 px): face colour distance 6.76 -> 5.49, whole picture 7.48 -> 7.57; tungsten faces 15.9 -> 11.0, whole 13.7 -> 12.8."""
import os, pickle

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
FACES_FILE = os.path.join(ROOT, 'candidates', 'personal', 'faces2.pkl'); MODEL_FILE = os.path.join(ROOT, 'out', 'cache', 'lr-skin-target.pkl')
GLOBAL, LOCAL = 0.5, 0.6
MIN_AREA, MIN_REAL, MIN_SCORE = 0.005, 0.5, 0.5
_cache = {}


def faces():
    if 'faces' not in _cache: _cache['faces'] = pickle.load(open(FACES_FILE, 'rb')) if os.path.exists(FACES_FILE) else {}
    return _cache['faces']


def main_face(row):
    """The largest real face of catalogue row `row`, or None."""
    fs = [f for f in faces().get(str(int(row)), []) if f['real'] >= MIN_REAL and f['score'] >= MIN_SCORE and f['box'][2] * f['box'][3] >= MIN_AREA]
    return max(fs, key=lambda f: f['box'][2] * f['box'][3]) if fs else None


def skin_mask(shape, lm, erode=0.06):
    H, W = shape[:2]; p = np.asarray(lm, np.float32) * [W, H]; m = np.zeros((H, W), np.uint8)
    brow = p[17:27]; top = brow - [0, 0.12 * (p[8, 1] - brow[:, 1].mean())]
    cv2.fillConvexPoly(m, cv2.convexHull(np.vstack([p[0:17], top]).astype(np.int32)), 1)
    w = max(4, int(np.ptp(p[:, 0]) * 0.05))
    for a, b in ((36, 42), (42, 48), (48, 60), (17, 22), (22, 27)):
        cv2.fillConvexPoly(m, cv2.convexHull(p[a:b].astype(np.int32)), 0); cv2.polylines(m, [p[a:b].astype(np.int32)], True, 0, w)
    k = max(1, int(np.ptp(p[:, 0]) * erode)); return cv2.erode(m, np.ones((k, k), np.uint8)) > 0


def lab(bgr): return cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def skin(L, m):
    v = L[m]; v = v[(v[:, 0] > 15) & (v[:, 0] < 95)] if len(v) else v
    return np.median(v, 0) if len(v) >= 20 else None


def features(x, face, so, sn):
    h = lambda s: np.degrees(np.arctan2(s[2], s[1]))
    return np.concatenate([x, so, sn, [np.hypot(*so[1:]), h(so), np.hypot(*sn[1:]), h(sn), np.log(face['box'][2] * face['box'][3]), face['pose'][0], face['pose'][1]]])


def models():
    if 'models' not in _cache: _cache['models'] = pickle.load(open(MODEL_FILE, 'rb'))
    return _cache['models']


def apply(out_bgr, neu_bgr, x, face, glob=GLOBAL, local=LOCAL):
    """`out_bgr` (the edited picture) with its face moved toward the predicted skin; `face` with box/lm as fractions of this frame. (picture, what was done)."""
    m = skin_mask(out_bgr.shape, face['lm']); Lo = lab(out_bgr); so, sn = skin(Lo, m), skin(lab(neu_bgr), m)
    if so is None or sn is None: return out_bgr, {}
    f = features(np.asarray(x, np.float64), face, so, sn)[None]; target = so + np.array([mm.predict(f)[0] for mm in models()])
    need = target[1:] - so[1:]
    if not np.all(np.isfinite(need)): return out_bgr, {}
    wl = np.clip(Lo[..., 0] / 20, 0, 1)[..., None]; L2 = Lo.copy(); L2[..., 1:] += glob * need * wl
    H, W = Lo.shape[:2]; r = max(3, int(0.25 * max(face['box'][2] * W, face['box'][3] * H)))
    fm = cv2.GaussianBlur(cv2.dilate(m.astype(np.uint8), np.ones((r, r), np.uint8)).astype(np.float32), (0, 0), r / 2)[..., None]; L2[..., 1:] += local * (1 - glob) * need * fm
    out = np.clip(cv2.cvtColor(L2, cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)
    return out, dict(face_target=[round(float(v), 1) for v in target], face_shift_ab=[round(float(v), 1) for v in need])
