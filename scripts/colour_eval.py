"""Compare the colour of our development with the camera's JPEG: whole-picture and orange-area median chroma and lightness (Lab) for each setting of the profile map and the tone-curve mode.
Usage: python scripts/colour_eval.py <folder> <name stem> [...]   (e.g. DJI_20250402190111_0037_D)"""
import itertools, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import develop as D


def lab(rgb): return cv2.cvtColor(np.ascontiguousarray(rgb.astype(np.float32)), cv2.COLOR_RGB2Lab)


def stats(l, ref_hue, mask):
    c = np.hypot(l[..., 1], l[..., 2]); return dict(chroma=float(np.median(c)), c_orange=float(np.median(c[mask])), L_orange=float(np.median(l[..., 0][mask])), a_ratio=float(np.median(l[..., 1][mask]) / max(np.median(l[..., 2][mask]), 1e-3)))


def main(folder, stems):
    for stem in stems:
        jp = next(os.path.join(folder, n) for n in os.listdir(folder) if n.upper().startswith(stem.upper()) and n.upper().endswith('.JPG'))
        j = cv2.cvtColor(cv2.imread(jp), cv2.COLOR_BGR2RGB).astype(np.float32) / 255; lj = lab(j); cj = np.hypot(lj[..., 1], lj[..., 2]); hue = np.degrees(np.arctan2(lj[..., 2], lj[..., 1]))
        mask = (cj > 20) & (hue > 40) & (hue < 85)                                                                             # orange: warm and clearly coloured in the camera's own rendering
        print(f'\n{stem}: orange area {mask.mean() * 100:.1f} % of the picture'); print('  camera JPEG', {k: round(v, 2) for k, v in stats(lj, hue, mask).items()})
        for hs, blend in itertools.product((False, True), (1.0, 0.0)):
            D.HUE_SAT_MAP = hs; D.TONE_BLEND = blend; out = D.develop(os.path.join(folder, os.path.basename(jp)[:-4] + '.DNG'), half=True, backend='gpu', highlight_contrast=0)
            out = cv2.resize(out, (j.shape[1], j.shape[0]), interpolation=cv2.INTER_AREA); print(f'  hue-sat map {hs!s:5} tone {"per-channel" if blend else "RGBTone    "}', {k: round(v, 2) for k, v in stats(lab(out), hue, mask).items()})


if __name__ == '__main__': main(sys.argv[1], sys.argv[2:])
