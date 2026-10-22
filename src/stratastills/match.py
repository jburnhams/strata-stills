"""Matching a picture's levels to those of other pictures of the same scene: a merged panorama (which is exposed by its own content, not as the frames it was made from were) is brought to the output levels of the frames next to it.

How: the panorama and a developed reference frame are matched by SIFT features and a RANSAC homography; around every feature both agree on, the colour of a small patch is taken in each (display-encoded, 0..1). Those pairs (panorama value, reference value) are the data of a monotone curve per colour channel (quantile matching, pinned at black and white, pulled towards the identity when there are few points), which is then applied to the whole panorama. The curve moves levels, and the tint along with them, but it can only be as good as the frames: it never adds detail.
`fit` and `apply` are the maths (tests/unit/test_match.py); `references` picks the frames; pipeline.py runs it as part of the develop stage (the develop setting `match`)."""
import re

import cv2, numpy as np

MIN_POINTS = 40                 # fewer agreeing features than this: no curve is made (the match is not trusted)
ENOUGH = 80                     # references are used in order (the frame the panorama is named after first) until this many points agree: frames are each exposed by their own content, so mixing several pulls the levels between them
FULL_TRUST = 150                # with this many points the fitted curve is used in full; with fewer it is pulled towards the identity in proportion
BINS = 12
WINDOW = (-150.0, 15.0)         # seconds around the panorama's own time (taken from the file name) in which its frames were shot
MAX_REFERENCES = 3
WORK_PX = 3000                  # long side of the copies the features are found on


def stamp(stem):
    """Seconds (a number to subtract) of the DJI time stamp in a file stem (DJI_20250402190211_0046_D), or None."""
    m = re.search(r'_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_', stem)
    if not m: return None
    import datetime as dt
    return dt.datetime(*(int(g) for g in m.groups())).timestamp()


def references(item, items, n=MAX_REFERENCES):
    """The items to match `item` (a panorama) to: single shots of the same source whose time stamp is within WINDOW of its own: the shot it is named after first, then nearest in time. Items without a time stamp in their name give none."""
    t0 = stamp(item['stem'])
    if t0 is None: return []
    out = []
    for it in items:
        if it['id'] == item['id'] or it['kind'] != 'still' or it.get('video'): continue
        t = stamp(it['stem'])
        if t is not None and WINDOW[0] <= t - t0 <= WINDOW[1]: out.append((abs(t - t0), it))
    out.sort(key=lambda x: (x[1]['stem'] != item['stem'].replace('-Pano', ''), x[0]))                                         # the shot it is named after first, then by time
    return [it for _, it in out][:n]


def _gray(rgb): return (np.clip(cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY), 0, 1) * 255).astype(np.uint8)


def _small(rgb, px=WORK_PX):
    k = px / max(rgb.shape[:2]); return cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else rgb


NEAR = 300                      # grid points count only within this many pixels (of the reduced reference) of a feature that agrees
FLAT = 0.045                    # the most colour spread (std of display values) a patch may have to count


def pairs(pano, ref, radius=10, grid=100, flat=None):
    """(n, 2, 3): the mean colour of a patch at points of a grid over the reference, and of the same place in the panorama, [pano, reference] (RGB, 0..1, as the pictures are). The place is found with SIFT features and a RANSAC homography (the panorama's projection is not exactly the frame's, but patches are small); only FLAT patches count (colour spread within the patch below `flat`, default FLAT, in both), since patches over an edge differ by the slightest misalignment, which would pass for a tone difference. Empty (0, 2, 3) when the pictures do not match."""
    flat = FLAT if flat is None else flat; empty = np.zeros((0, 2, 3)); p, r = _small(pano), _small(ref); sift = cv2.SIFT_create(8000); kp1, d1 = sift.detectAndCompute(_gray(r), None); kp2, d2 = sift.detectAndCompute(_gray(p), None)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8: return empty
    good = [m[0] for m in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 8: return empty
    src = np.float32([kp1[m.queryIdx].pt for m in good]); dst = np.float32([kp2[m.trainIdx].pt for m in good]); H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < 15: return empty
    scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2])))); keep = inl.ravel() > 0; near = np.zeros(r.shape[:2], np.uint8)                                  # one homography is only locally exact (a panorama's projection is curved): only grid points near features that agree are used
    for x, y in src[keep]: near[min(int(y), r.shape[0] - 1), min(int(x), r.shape[1] - 1)] = 255
    near = cv2.dilate(near, np.ones((2 * NEAR + 1, 2 * NEAR + 1), np.uint8)) > 0
    if not 0.2 < scale < 5: return empty
    rr = max(2, int(round(radius / scale))); gy, gx = np.mgrid[rr:r.shape[0] - rr:grid * 1j, rr:r.shape[1] - rr:grid * 1j]; pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2); q = cv2.perspectiveTransform(pts, H).reshape(-1, 2); pts = pts.reshape(-1, 2); out = []
    for (x1, y1), (x2, y2) in zip(pts, q):
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if radius <= x2 < p.shape[1] - radius and radius <= y2 < p.shape[0] - radius:
            a = p[y2 - radius:y2 + radius + 1, x2 - radius:x2 + radius + 1].reshape(-1, 3); b = r[y1 - rr:y1 + rr + 1, x1 - rr:x1 + rr + 1].reshape(-1, 3)
            if near[min(y1, near.shape[0] - 1), min(x1, near.shape[1] - 1)] and a.std(0).max() < flat and b.std(0).max() < flat: out.append([a.mean(0), b.mean(0)])
    return np.array(out, float).reshape(-1, 2, 3)


def fit(data, size=256):
    """({curves: (3, size) float32 lookup tables over 0..1 input, points, before, after} or None): the monotone curve of each colour channel from `data` (n, 2, 3) pairs [source, target] by quantile matching. before and after are the rms differences of the pairs without and with the curves."""
    n = len(data)
    if n < MIN_POINTS: return None
    xs = np.linspace(0, 1, size); trust = min(1.0, n / FULL_TRUST); curves = np.empty((3, size), np.float32)
    for c in range(3):
        x, y = data[:, 0, c], data[:, 1, c]; probs = np.linspace(0.02, 0.98, BINS)                                                              # quantile matching: the levels the panorama shows at these places take the levels the reference shows there. (Binned medians of y given x would be flattened by the scatter between the two views, a regression to the mean that is not a tone difference.)
        kx = np.concatenate([[0.0], np.quantile(x, probs), [1.0]]); ky = np.maximum.accumulate(np.clip(np.concatenate([[0.0], np.quantile(y, probs), [1.0]]), 0, 1)); keep = np.concatenate([[True], np.diff(kx) > 1e-4]); g = np.interp(xs, kx[keep], ky[keep])
        f = np.maximum.accumulate(xs + trust * (g - xs)); curves[c] = np.clip(f, 0, 1)
    ap = np.stack([np.interp(data[:, 0, c], xs, curves[c]) for c in range(3)], 1)
    return dict(curves=curves, points=int(n), before=float(np.sqrt(np.mean((data[:, 0] - data[:, 1]) ** 2))), after=float(np.sqrt(np.mean((ap - data[:, 1]) ** 2))))


def apply(rgb, curves):
    """The picture (float32 RGB 0..1) through the per-channel curves from `fit`."""
    xs = np.linspace(0, 1, curves.shape[1]); out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3): out[..., c] = np.interp(rgb[..., c], xs, curves[c])
    return out


def match_levels(pano, refs):
    """(matched picture, record) for the panorama `pano` and a list of developed reference pictures: the record says how many features agreed and how well the curves fit (`applied` False with a `reason` when it did not match enough)."""
    parts = []
    for r in refs:
        parts.append(pairs(pano, r))
        if sum(len(x) for x in parts) >= ENOUGH: break
    data = np.concatenate(parts or [np.zeros((0, 2, 3))]); f = fit(data)
    if f is None: return pano, dict(applied=False, points=int(len(data)), reason=f'too few features agree with the neighbouring pictures ({len(data)} of {MIN_POINTS} needed)')
    return apply(pano, f['curves']), dict(applied=True, points=f['points'], rms_before=round(f['before'], 4), rms_after=round(f['after'], 4))
