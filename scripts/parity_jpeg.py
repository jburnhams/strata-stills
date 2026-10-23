"""How close is our development of a raw to the camera's own JPEG of it? For raw+JPG pairs: develop each raw (half size, `exposure` as given) and compare with the JPEG area-averaged to 1/8 (CIELAB): median lightness and chroma of both, the median CIE76 colour difference, and the hue difference of the coloured pixels. Prints one line per picture and the medians over the set.
Usage: python scripts/parity_jpeg.py <folder> [--n 40] [--ev camera] [--ext .ARW]
  --candidates docs/a7c_candidates.json   only those pictures (scripts/pick_candidates.py): each is compared with the numbers saved in the file and any that got worse (the 90th-percentile and the per-region colour differences, or the median, by more than 5 % and 0.3) is flagged: the check that a change helps the worst without hurting the best
  --sheets <dir>                          also write a picture per candidate: ours | the camera's | where they differ (colour difference, white = 12 or more)"""
import argparse, glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import develop, tonematch


def lab(rgb8):
    l = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32); l[..., 0] /= 2.55; l[..., 1:] -= 128; return l


def write_sheet(path, a, j):
    """ours | the camera's | the colour difference (CIE76 per pixel, black = 0, white = 12 or more), at a quarter of the half size."""
    size = (a.shape[1] // 4, a.shape[0] // 4); a = cv2.resize(a, size, interpolation=cv2.INTER_AREA); j = cv2.resize(np.ascontiguousarray(j), size, interpolation=cv2.INTER_AREA)
    d = (np.clip(np.linalg.norm(lab(a) - lab(j), axis=2) / 12, 0, 1) * 255).astype(np.uint8); cv2.imwrite(path, np.hstack([a, j, cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)[..., ::-1]])[..., ::-1])


WORSE = ('dE', 'dE_p90', 'dE_sat', 'dE_dark', 'dE_bright')                                          # the colour differences that must not grow; the others are signed


def regressions(now, was):
    return [f'{k} {was[k]:.2f} -> {now[k]:.2f}' for k in WORSE if k in was and now[k] > was[k] * 1.05 + 0.3]


def local(A, J, cj):
    """Where the picture differs, not just on the whole: the 90th percentile of the colour difference, and its median in the saturated colours (JPEG chroma over 25), the shadows (JPEG L* under 20) and the highlights (over 70); 0 where the picture has none of them."""
    d = np.linalg.norm(A - J, axis=2); med = lambda m: float(np.median(d[m])) if m.sum() > 100 else 0.0
    return dict(dE_p90=float(np.percentile(d, 90)), dE_sat=med(cj > 25), dE_dark=med(J[..., 0] < 20), dE_bright=med(J[..., 0] > 70))


def compare(raw, jpg, ev, sheet=None):
    if ev == 'match': c = tonematch.curve_for(raw); a = develop.develop(raw, half=True, ev='camera', tone=c) if c else None
    else: a = develop.develop(raw, half=True, ev=ev)
    if a is None: return None
    a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8); j = cv2.imread(jpg, cv2.IMREAD_COLOR)[..., ::-1]
    if sheet: write_sheet(sheet, a, j)
    k = 1 / 8; size = (int(j.shape[1] * k), int(j.shape[0] * k)); a = cv2.resize(a, size, interpolation=cv2.INTER_AREA); j = cv2.resize(np.ascontiguousarray(j), size, interpolation=cv2.INTER_AREA)
    A, J = lab(a), lab(j); ca, cj = np.hypot(A[..., 1], A[..., 2]), np.hypot(J[..., 1], J[..., 2]); m = (ca > 8) & (cj > 8)
    dh = np.degrees(np.angle(np.exp(1j * (np.arctan2(A[..., 2], A[..., 1]) - np.arctan2(J[..., 2], J[..., 1]))))) if m.any() else np.zeros(1)
    return dict(dL=float(np.median(A[..., 0]) - np.median(J[..., 0])), Lj=float(np.median(J[..., 0])), chroma_ratio=float(np.median(ca) / max(np.median(cj), 1e-3)), dE=float(np.median(np.linalg.norm(A - J, axis=2))), dhue=float(np.median(dh[m] if m.any() else dh)), dL_pix=float(np.median(A[..., 0] - J[..., 0])), **local(A, J, cj))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('--n', type=int, default=40); ap.add_argument('--ev', default='camera'); ap.add_argument('--ext', default='.ARW'); ap.add_argument('--candidates'); ap.add_argument('--sheets'); a = ap.parse_args()
    ev = a.ev if a.ev in ('auto', 'camera', 'match') else float(a.ev); raws = sorted(glob.glob(os.path.join(a.folder, '*' + a.ext))); raws = [raws[i] for i in np.linspace(0, len(raws) - 1, min(a.n, len(raws))).astype(int)]; rows = []; was = {}
    if a.candidates:
        import json; was = {c['stem']: c['metrics'] for c in json.load(open(a.candidates))}; raws = [r for r in sorted(glob.glob(os.path.join(a.folder, '*' + a.ext))) if os.path.splitext(os.path.basename(r))[0] in was]
    if a.sheets: os.makedirs(a.sheets, exist_ok=True)
    for r in raws:
        jpg = os.path.splitext(r)[0] + '.JPG'
        stem = os.path.splitext(os.path.basename(r))[0]; d = compare(r, jpg, ev, os.path.join(a.sheets, stem + '.jpg') if a.sheets else None) if os.path.exists(jpg) else None
        if d is None: print(os.path.basename(r), 'no match'); continue
        rows.append(d); bad = regressions(d, was[stem]) if stem in was else []; print(os.path.basename(r), ' '.join(f'{k} {v:6.2f}' for k, v in d.items()), ('  WORSE: ' + ', '.join(bad)) if bad else '', flush=True)
    print('MEDIAN', ' '.join(f'{k} {np.median([r[k] for r in rows]):6.2f}' for k in rows[0]), '| worst |dL_pix|', round(max(abs(r['dL_pix']) for r in rows), 2))
