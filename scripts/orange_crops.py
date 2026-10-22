"""Side-by-side crops of the sunlit area of one shot: camera JPEG | old look (no profile map, no chroma limit) | new look. Usage: python scripts/orange_crops.py <folder> <stem> <out.png>"""
import os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import develop as D


def main(folder, stem, out_path):
    jp = next(os.path.join(folder, n) for n in os.listdir(folder) if n.upper().startswith(stem.upper()) and n.upper().endswith('.JPG')); j = cv2.cvtColor(cv2.imread(jp), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    new = D.develop(jp[:-4] + '.DNG', half=True, backend='gpu'); D.HUE_SAT_MAP = False; D.CHROMA_KNEE = 0; old = D.develop(jp[:-4] + '.DNG', half=True, backend='gpu')
    f = lambda a: cv2.resize(a, (j.shape[1], j.shape[0]), interpolation=cv2.INTER_AREA); old, new = f(old), f(new)
    lj = cv2.cvtColor(j, cv2.COLOR_RGB2Lab); c = np.hypot(lj[..., 1], lj[..., 2]); hue = np.degrees(np.arctan2(lj[..., 2], lj[..., 1])); m = ((c > 20) & (hue > 40) & (hue < 85)).astype(np.float32)
    box = cv2.boxFilter(m, -1, (400, 400)); y, x = np.unravel_index(np.argmax(box), box.shape); y0 = int(np.clip(y - 300, 0, j.shape[0] - 600)); x0 = int(np.clip(x - 200, 0, j.shape[1] - 400))
    tiles = [a[y0:y0 + 600, x0:x0 + 400] for a in (j, old, new)]
    img = np.concatenate(tiles, 1); img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    for i, label in enumerate(('camera JPEG', 'before', 'after')): cv2.putText(img, label, (i * 400 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__': main(*sys.argv[1:4])
