"""How close is our development of a raw to the camera's own JPEG of it? For raw+JPG pairs: develop each raw (half size, `exposure` as given) and compare with the JPEG area-averaged to 1/8 (CIELAB): median lightness and chroma of both, the median CIE76 colour difference, and the hue difference of the coloured pixels. Prints one line per picture and the medians over the set.
Usage: python scripts/parity_jpeg.py <folder> [--n 40] [--ev camera] [--ext .ARW]"""
import argparse, glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import develop


def lab(rgb8):
    l = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32); l[..., 0] /= 2.55; l[..., 1:] -= 128; return l


def compare(raw, jpg, ev):
    a = (np.clip(develop.develop(raw, half=True, ev=ev), 0, 1) * 255 + 0.5).astype(np.uint8); j = cv2.imread(jpg, cv2.IMREAD_COLOR)[..., ::-1]
    k = 1 / 8; size = (int(j.shape[1] * k), int(j.shape[0] * k)); a = cv2.resize(a, size, interpolation=cv2.INTER_AREA); j = cv2.resize(np.ascontiguousarray(j), size, interpolation=cv2.INTER_AREA)
    A, J = lab(a), lab(j); ca, cj = np.hypot(A[..., 1], A[..., 2]), np.hypot(J[..., 1], J[..., 2]); m = (ca > 8) & (cj > 8)
    dh = np.degrees(np.angle(np.exp(1j * (np.arctan2(A[..., 2], A[..., 1]) - np.arctan2(J[..., 2], J[..., 1]))))) if m.any() else np.zeros(1)
    return dict(dL=float(np.median(A[..., 0]) - np.median(J[..., 0])), Lj=float(np.median(J[..., 0])), chroma_ratio=float(np.median(ca) / max(np.median(cj), 1e-3)), dE=float(np.median(np.linalg.norm(A - J, axis=2))), dhue=float(np.median(dh[m] if m.any() else dh)), dL_pix=float(np.median(A[..., 0] - J[..., 0])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('--n', type=int, default=40); ap.add_argument('--ev', default='camera'); ap.add_argument('--ext', default='.ARW'); a = ap.parse_args()
    ev = a.ev if a.ev in ('auto', 'camera') else float(a.ev); raws = sorted(glob.glob(os.path.join(a.folder, '*' + a.ext))); raws = [raws[i] for i in np.linspace(0, len(raws) - 1, min(a.n, len(raws))).astype(int)]; rows = []
    for r in raws:
        jpg = os.path.splitext(r)[0] + '.JPG'
        if os.path.exists(jpg): d = compare(r, jpg, ev); rows.append(d); print(os.path.basename(r), ' '.join(f'{k} {v:6.2f}' for k, v in d.items()), flush=True)
    print('MEDIAN', ' '.join(f'{k} {np.median([r[k] for r in rows]):6.2f}' for k in rows[0]), '| worst |dL_pix|', round(max(abs(r['dL_pix']) for r in rows), 2))
