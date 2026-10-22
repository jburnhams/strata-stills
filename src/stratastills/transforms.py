"""The transforms offered on a picture: each takes a working image and returns (image, info), where info says what was done and why (so the app can show and explain it).

  original   the picture as it is
  level      rotate so the lines of the scene are square to the frame
               straight-down (nadir) shots: the dominant direction of the long straight edges (roof ridges, kerbs, fences), folded to the nearest multiple of 90 degrees, so up to 45 degrees of turn;
               oblique shots (gimbal pitch above -80): only a small tilt (MAX_OBLIQUE_DEG) of near-vertical and near-horizontal lines is believed, since perspective makes the rest converge.
             The turn is applied together with the zoom that crops away the empty corners, keeping the aspect ratio. Nothing is applied when the evidence is weak (info says `applied: false` and why).

Bump a transform's VERSION when its result changes: it is part of the cache key."""
import math

import cv2, numpy as np

VERSION = dict(original=1, level=1)
LABELS = dict(original='Original', level='Levelled')
ORDER = ('original', 'level')
MAX_OBLIQUE_DEG = 10.0
MIN_APPLY_DEG = 0.15            # a smaller turn is not worth a resample
MIN_CONFIDENCE = 0.35
MIN_LINES = 12
WORK_PX = 1400


def line_angles(gray):
    """(deviation of each long straight segment from the nearest axis in degrees (-45..45), its length) found with Canny + probabilistic Hough on a copy of the image no bigger than WORK_PX."""
    h, w = gray.shape; s = WORK_PX / max(h, w)
    if s < 1: gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0); med = float(np.median(gray)); e = cv2.Canny(gray, max(10, 0.66 * med), max(40, 1.33 * med))
    L = max(gray.shape); segs = cv2.HoughLinesP(e, 1, np.pi / 720, threshold=60, minLineLength=max(30, int(0.06 * L)), maxLineGap=int(0.01 * L))
    if segs is None: return np.zeros(0), np.zeros(0)
    d = np.asarray(segs).reshape(-1, 4).astype(float)                  # (N,1,4) in OpenCV 4, (N,4) in 5
    dx, dy = d[:, 2] - d[:, 0], d[:, 3] - d[:, 1]
    th = np.degrees(np.arctan2(dy, dx)); dev = (th + 45.0) % 90.0 - 45.0
    return dev, np.hypot(dx, dy)


def estimate_level(gray, nadir):
    """{angle, confidence, lines, max_deg, reason}: `angle` is how far (degrees, positive = the scene is turned clockwise on screen) the lines are from square; the picture should be turned the other way by it."""
    max_deg = 45.0 if nadir else MAX_OBLIQUE_DEG
    dev, w = line_angles(gray); r = dict(angle=0.0, confidence=0.0, lines=int(len(dev)), max_deg=max_deg, reason=None)
    if len(dev) < MIN_LINES: r['reason'] = f'too few straight lines ({len(dev)})'; return r
    if nadir:                                                       # axial data with a period of 90 degrees: average in the 4x angle
        a = np.radians(dev * 4.0); c, s = (w * np.cos(a)).sum(), (w * np.sin(a)).sum()
        r['angle'] = float(np.degrees(math.atan2(s, c)) / 4.0); r['confidence'] = float(math.hypot(c, s) / w.sum())
    else:
        k = np.abs(dev) <= max_deg
        if k.sum() < MIN_LINES: r['reason'] = 'too few lines near vertical or horizontal'; return r
        d, ww = dev[k], w[k]; o = np.argsort(d); cw = np.cumsum(ww[o]); med = float(d[o][np.searchsorted(cw, cw[-1] / 2)])
        r['angle'] = med; r['confidence'] = float(ww[np.abs(d - med) <= 1.5].sum() / w.sum())
    return r


def inscribed_scale(w, h, deg):
    """The zoom that makes the rotated picture fill the frame again (same aspect ratio, no empty corners)."""
    t = math.radians(abs(deg)); return math.cos(t) + math.sin(t) * max(w / h, h / w)


def rotate_fill(bgr, deg):
    h, w = bgr.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, inscribed_scale(w, h, deg))
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def level(bgr, ctx):
    r = estimate_level(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bool(ctx.get('nadir')))
    r['applied'] = False
    if r['reason'] is None and r['confidence'] < MIN_CONFIDENCE: r['reason'] = f"the lines do not agree (confidence {r['confidence']:.2f})"
    if r['reason'] is None and abs(r['angle']) < MIN_APPLY_DEG: r['reason'] = 'already level'
    if r['reason'] is not None: return bgr, r
    r['applied'] = True; return rotate_fill(bgr, r['angle']), r


def original(bgr, ctx): return bgr, dict(applied=False)


FUNCS = dict(original=original, level=level)


def apply(name, bgr, ctx):
    if name not in FUNCS: raise KeyError(f'unknown transform {name!r}')
    return FUNCS[name](bgr, ctx or {})
