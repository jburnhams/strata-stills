"""Fit a device's tone curve to what the camera itself does: for raws that come with their own camera JPEG (Sony .ARW + .JPG ...), pair each picture's developed linear luminance (no exposure change, no tone curve) with the JPEG's luminance, and take the median JPEG value in each of N equal-count slices of the sorted linear values. This is the quantile matching LightCraft's `calibrate` / camera_preview::fit_tone does (https://github.com/storytold/lightcraft, MIT OR Apache-2.0: the method, not its code).
Writes `ProfileToneCurve` (pairs of linear input, display-encoded output, as the DNG tag) into a bundle made by build_profile_bundle.py.
Usage: python scripts/fit_tone_curve.py <folder with raw+JPG pairs> --bundle src/stratastills/sony_ilce7c_profile.json [--n 24] [--ext .ARW]"""
import argparse, glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import develop

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
PROXY = 384                                                                                          # long side of the pair compared: area-averaged, so differences of sharpness, alignment and noise do not count


def pair(raw, jpg):
    """(linear luminance of the developed raw, linear luminance of the camera JPEG) as flat arrays, plus the largest channel of the raw (to leave clipped pixels out)."""
    p = develop.develop_parts(raw, half=True, ev=0.0, recover=0.0)['linear']; j = cv2.imread(jpg, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    k = PROXY / max(p.shape[:2]); size = (int(p.shape[1] * k), int(p.shape[0] * k)); p = cv2.resize(p, size, interpolation=cv2.INTER_AREA); j = cv2.resize(j, size, interpolation=cv2.INTER_AREA)
    return (p @ LUMA).ravel(), (develop.srgb_eotf(j) @ LUMA).ravel(), p.max(axis=2).ravel()


def fit(xs, ys, bins):
    """Knots (x = median linear value of each slice, y = median display-encoded JPEG luminance), forced monotone, with (0, 0) and (1, top) at the ends."""
    o = np.argsort(xs); xs, ys = xs[o], ys[o]; cut = np.linspace(0, len(xs), bins + 1).astype(int)
    kx = np.array([np.median(xs[a:b]) for a, b in zip(cut[:-1], cut[1:])]); ky = np.array([np.median(ys[a:b]) for a, b in zip(cut[:-1], cut[1:])])
    ky = np.maximum.accumulate(ky); keep = np.concatenate([[True], np.diff(kx) > 1e-6]); kx, ky = kx[keep], ky[keep]
    top = min(1.0, ky[-1] + (1.0 - kx[-1]) * (ky[-1] - ky[-2]) / max(kx[-1] - kx[-2], 1e-6))
    return np.concatenate([[0.0], kx, [1.0]]), np.concatenate([[0.0], ky, [max(top, ky[-1])]])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('--bundle', required=True); ap.add_argument('--n', type=int, default=24); ap.add_argument('--ext', default='.ARW'); ap.add_argument('--bins', type=int, default=48); a = ap.parse_args()
    raws = sorted(glob.glob(os.path.join(a.folder, '*' + a.ext))); raws = [raws[i] for i in np.linspace(0, len(raws) - 1, min(a.n, len(raws))).astype(int)]
    X, Y = [], []
    for r in raws:
        jpg = os.path.splitext(r)[0] + '.JPG'
        if not os.path.exists(jpg): continue
        x, y, mx = pair(r, jpg); ok = (mx < 0.97) & (x > 1e-4) & (y > 3e-4) & (y < 0.9); X.append(x[ok]); Y.append(y[ok]); print(os.path.basename(r), ok.mean().round(2), flush=True)
    xs = np.concatenate(X); ys = np.concatenate(Y); ys_enc = np.where(ys <= 0.0031308, 12.92 * ys, 1.055 * ys ** (1 / 2.4) - 0.055)
    kx, ky = fit(xs, ys_enc, a.bins); b = json.load(open(a.bundle)); b['ProfileToneCurve'] = [round(float(v), 5) for p in zip(kx, ky) for v in p]
    json.dump(b, open(a.bundle, 'w'), separators=(',', ':')); print('curve', len(kx), 'knots; encoded output at linear 0.01/0.18/0.5/1:', np.interp([0.01, 0.18, 0.5, 1.0], kx, ky).round(3))
