"""Sweep the chroma limiter's knee and ceiling against the camera JPEGs (see colour_eval.py). Usage: python scripts/chroma_sweep.py <folder> <stem> [...]"""
import os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import develop as D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colour_eval import lab


def main(folder, stems):
    shots = []
    for stem in stems:
        jp = next(os.path.join(folder, n) for n in os.listdir(folder) if n.upper().startswith(stem.upper()) and n.upper().endswith('.JPG'))
        j = cv2.cvtColor(cv2.imread(jp), cv2.COLOR_BGR2RGB).astype(np.float32) / 255; lj = lab(j); cj = np.hypot(lj[..., 1], lj[..., 2])
        hi = cj > np.percentile(cj, 90)                                                                                         # the most coloured tenth of the camera's picture
        out = D.develop(jp[:-4] + '.DNG', half=True, backend='gpu', highlight_contrast=0); out = cv2.resize(out, (j.shape[1], j.shape[0]), interpolation=cv2.INTER_AREA); shots.append((stem, cj, hi, out))
    for knee, ceil in [(None, None), (0.10, 0.30), (0.12, 0.30), (0.15, 0.35), (0.15, 0.30), (0.20, 0.35), (0.20, 0.45)]:
        row = []
        for stem, cj, hi, out in shots:
            o = out if knee is None else D.limit_chroma(out, knee, ceil); c = np.hypot(*np.moveaxis(lab(o)[..., 1:], -1, 0)); row.append(f'{stem[-7:]}: all {np.median(c):5.2f}/{np.median(cj):5.2f}  top10% {np.median(c[hi]):5.2f}/{np.median(cj[hi]):5.2f}')
        print(f'knee {knee} ceil {ceil}  ' + '   '.join(row))


if __name__ == '__main__': main(sys.argv[1], sys.argv[2:])
