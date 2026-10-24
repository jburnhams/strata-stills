"""Distortion of a lens from the camera's own JPEG, for a lens no lensfun profile describes: the raw is developed with no lens correction (as the JPEG is, when its distortion correction is off), the optical flow from that picture to the JPEG is measured on the edge maps (so that the different tone does not matter), and a radial displacement  dr = r (k1 r^2 + k2 r^4)  (r from the centre, 1 at the corner) is fitted to the flow by least squares over the pictures. k1 is the poly3 coefficient lensfun uses  (r_source = r_dest (1 + k1 r_dest^2), r relative to the half diagonal... here normalised to the corner: lensfun normalises to the half of the SHORT side, the conversion is printed).
  python scripts/fit_distortion.py candidates/a7c --lens "SAMYANG AF 50mm F1.4 II" [--size 1500]
Checked on a lens with a lensfun profile first (--lens "SAMYANG AF 35mm F2.8"): with `--mode geometry` (lensfun's correction applied) what is left should be near 0 where lensfun is right."""
import argparse, glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from stratastills import develop as D, lensfundb as F


def edges(img):
    g = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32); g = cv2.GaussianBlur(g, (0, 0), 1.5)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)); return np.clip(m / max(np.percentile(m, 99.5), 1e-3), 0, 1)


def one(raw, jpg, size, mode):
    ours = D.develop(raw, half=True, lens=mode, backend='cpu', tone=None); j = cv2.cvtColor(cv2.imread(jpg), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    s = size / max(ours.shape[:2]); ours = cv2.resize(ours, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); j = cv2.resize(j, (ours.shape[1], ours.shape[0]), interpolation=cv2.INTER_AREA)
    a, b = (edges(ours) * 255).astype(np.uint8), (edges(j) * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 5, 31, 5, 7, 1.5, 0)                      # a -> b: where each pixel of ours is in the JPEG
    h, w = a.shape; yy, xx = np.mgrid[:h, :w]; x = (xx - (w - 1) / 2) / (np.hypot(w, h) / 2); y = (yy - (h - 1) / 2) / (np.hypot(w, h) / 2); r = np.hypot(x, y)
    conf = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 6); ok = (conf > np.percentile(conf, 60)) & (r > 0.05) & (r < 0.98)
    dr = (flow[..., 0] * x + flow[..., 1] * y) / np.maximum(r, 1e-6) / (np.hypot(w, h) / 2)          # radial displacement, in units of the half diagonal
    return r[ok], dr[ok]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('--lens', required=True); ap.add_argument('--size', type=int, default=1500); ap.add_argument('--patch', action='append', default=[], help='module.NAME=value for a constant of stratastills.<module>, e.g. lenscorr.GEOMETRY_FIRST="camera" lenscorr.USE_EMBEDDED_GEOMETRY=True'); ap.add_argument('--max', type=int, default=8); ap.add_argument('--mode', default='off', help='the lens setting our picture is made with: off (what the fit measures), or geometry/auto/camera to see what is left over after a source'); a = ap.parse_args()
    import importlib
    for pt in a.patch: k, _, v = pt.partition('='); m, _, n = k.partition('.'); setattr(importlib.import_module('stratastills.' + m), n, eval(v))
    R, DR, per = [], [], []
    for raw in sorted(glob.glob(os.path.join(a.folder, '*.ARW')))[:]:
        sh = F.shot_of(raw)
        if not sh or sh[0].lower() != a.lens.lower() or len(per) >= a.max: continue
        jpg = os.path.splitext(raw)[0] + '.JPG'
        if not os.path.exists(jpg): continue
        r, dr = one(raw, jpg, a.size, a.mode); k = np.polyfit(r ** 3, dr, 1, w=None)[0] if len(r) else float('nan')
        per.append((os.path.basename(raw), sh[1], float(np.median(dr[r > 0.8])) if (r > 0.8).any() else float('nan'), k)); R.append(r[::50]); DR.append(dr[::50]); print('%-14s %5.1f mm  corner shift %+.4f  k %+.4f' % per[-1], flush=True)
    r, dr = np.concatenate(R), np.concatenate(DR); A = np.stack([r ** 3, r ** 5], 1); k1, k2 = np.linalg.lstsq(A, dr, rcond=None)[0]
    print('fit over %d pictures: dr = %+.4f r^3 %+.4f r^5 (r, dr in half diagonals; at the corner %+.4f, i.e. %+.2f %% of the half diagonal)' % (len(per), k1, k2, k1 + k2, 100 * (k1 + k2)))

if __name__ == '__main__': main()
