"""How well a generated panorama's levels agree with the generated picture of a frame it was made from: median Lab differences and chroma ratio over flat patches both show (match.pairs).
Usage: python scripts/pano_match_check.py <project dir> <pano stem> <frame stem>"""
import glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import match as M


def latest(project, stem): return max(glob.glob(os.path.join(project, 'generated', f'*~{stem}', '*.jpg')), key=os.path.getmtime)
def rd(p): return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def main(project, pano, frame):
    pr = M.pairs(rd(latest(project, pano)), rd(latest(project, frame))); L = cv2.cvtColor(pr.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2Lab).reshape(-1, 2, 3)
    print(f'{len(pr)} flat patches: dL {np.median(L[:, 0, 0] - L[:, 1, 0]):+.2f}  da {np.median(L[:, 0, 1] - L[:, 1, 1]):+.2f}  db {np.median(L[:, 0, 2] - L[:, 1, 2]):+.2f}  chroma ratio {np.median(np.hypot(L[:, 0, 1], L[:, 0, 2]) / np.maximum(np.hypot(L[:, 1, 1], L[:, 1, 2]), 1)):.2f}')


if __name__ == '__main__': main(*sys.argv[1:4])
