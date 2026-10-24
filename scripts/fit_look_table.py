"""Fit a device's hue/saturation look (develop.hue_look) to what its camera JPEGs do beyond the colour matrices: for every raw+JPG pair, our picture as the pipeline makes it with `exposure: match` against the camera's, both at 1/8 size; the hue the camera turned each colour to and the strength it gave it, in cells of (hue, colour strength), pooled over the whole shoot. Every `holdout`-th picture is left out of the fit and the result is judged on those (and on the rest).
  python scripts/fit_look_table.py build <folder> [<folder> ...] --cache pairs.npz [--ext .DNG --lens auto]        develops every picture (about 5 s each) and keeps the 1/8 size pairs (several folders: the shoot's name is kept in the picture's name; `--lens auto` for cameras whose JPEGs are lens-corrected)
  python scripts/fit_look_table.py fit --cache pairs.npz [--holdout-folder NAME] [--bundle src/stratastills/sony_ilce7c_profile.json]     fits, prints the numbers before and after, and with --bundle stores `LookTable` in the profile (fitted on the pictures other than the held-out ones)"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import parity_jpeg as PJ
from stratastills import develop as D

HUES = 24
N0 = 300                                    # a cell with this many pixels is believed halfway; fewer are pulled towards "no change"


def build(folders, cache, ext='.ARW', lens='off'):
    D.look_table = lambda spec: None                      # the pairs are what the pipeline makes BEFORE the look table (the one being fitted)
    stems, A, J = [], [], []
    for folder in folders:
        for r in sorted(glob.glob(os.path.join(folder, '*' + ext))):
            jpg = os.path.splitext(r)[0] + '.JPG'
            if not os.path.exists(jpg): continue
            p = PJ.proxies(r, jpg, 'match', lens=lens)
            if p is None: continue
            stems.append((os.path.basename(os.path.normpath(folder)) + '/' if len(folders) > 1 else '') + os.path.basename(r)); A.append(p[0]); J.append(p[1]); print(stems[-1], flush=True)
    np.savez_compressed(cache, stems=np.array(stems), **{f'a{i}': a for i, a in enumerate(A)}, **{f'j{i}': j for i, j in enumerate(J)})


def load(cache):
    z = np.load(cache); n = len(z['stems']); return list(z['stems']), [z[f'a{i}'] for i in range(n)], [z[f'j{i}'] for i in range(n)]


def polar(rgb):
    y = rgb @ D._W; d = rgb - y[..., None]; a = d @ D._E1; b = d @ D._E2; return y, np.arctan2(b, a), np.hypot(a, b)


def fit(A, J):
    """The table (HUES, columns, 2) = (rotation in degrees, colour scale) for pictures A (ours) and J (the camera's), lists of uint8 RGB."""
    cols = len(D.LOOK_CHROMA); rot = [[[] for _ in range(cols)] for _ in range(HUES)]; lr = [[[] for _ in range(cols)] for _ in range(HUES)]
    for a, j in zip(A, J):
        ya, ha, ca = polar(a.astype(np.float32) / 255); yj, hj, cj = polar(j.astype(np.float32) / 255); ok = (ca > 0.01) & (cj > 0.005) & (ya > 0.03) & (ya < 0.95)
        hi = (np.round(ha[ok] / (2 * np.pi) * HUES).astype(int)) % HUES; ci = np.abs(ca[ok][:, None] - D.LOOK_CHROMA[None]).argmin(1); dh = np.degrees(np.angle(np.exp(1j * (hj[ok] - ha[ok])))); lg = np.log(cj[ok] / ca[ok])
        for k in np.unique(hi * cols + ci):
            m = (hi * cols + ci) == k; rot[k // cols][k % cols].append(dh[m][::7]); lr[k // cols][k % cols].append(lg[m][::7])
    t = np.zeros((HUES, cols, 2), np.float32); t[..., 1] = 1.0
    for h in range(HUES):
        for c in range(cols):
            if rot[h][c]:
                r = np.concatenate(rot[h][c]); l = np.concatenate(lr[h][c]); w = len(r) * 7 / (len(r) * 7 + N0); t[h, c] = (w * np.median(r), np.exp(w * np.median(l)))
    sm = t.copy(); sm[:, :, 0] = (np.roll(t[:, :, 0], 1, 0) + 2 * t[:, :, 0] + np.roll(t[:, :, 0], -1, 0)) / 4; sm[:, :, 1] = (np.roll(t[:, :, 1], 1, 0) + 2 * t[:, :, 1] + np.roll(t[:, :, 1], -1, 0)) / 4; return sm


def apply(a, table):
    return (np.clip(D.hue_look(a.astype(np.float32) / 255, table), 0, 1) * 255 + 0.5).astype(np.uint8)


def score(A, J, table=None):
    rows = [PJ.metrics(apply(a, table) if table is not None else a, j) for a, j in zip(A, J)]; return rows, {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('build', 'fit')); ap.add_argument('folder', nargs='*'); ap.add_argument('--ext', default='.ARW'); ap.add_argument('--lens', default='off', choices=('off', 'auto')); ap.add_argument('--holdout-folder', help='fit: leave out every picture of the shoot folder whose name contains this (a different shoot is a harder test than every 5th picture)'); ap.add_argument('--cache', required=True); ap.add_argument('--bundle'); ap.add_argument('--holdout', type=int, default=5); a = ap.parse_args()
    if a.cmd == 'build': build(a.folder, a.cache, a.ext, a.lens); sys.exit()
    stems, A, J = load(a.cache)
    if a.holdout_folder: test = [i for i, s in enumerate(stems) if a.holdout_folder in s]; train = [i for i in range(len(stems)) if i not in set(test)]
    else: test = [i for i in range(len(stems)) if i % a.holdout == 0]; train = [i for i in range(len(stems)) if i % a.holdout]
    table = fit([A[i] for i in train], [J[i] for i in train]); keys = ('dE', 'dE_p90', 'dE_sat', 'dE_dark', 'dE_bright', 'chroma_ratio', 'dhue')
    for name, idx in (('held out', test), ('fitted on', train)):
        r0, m0 = score([A[i] for i in idx], [J[i] for i in idx]); r1, m1 = score([A[i] for i in idx], [J[i] for i in idx], table)
        print(f'{name} ({len(idx)} pictures), median before -> after:', ', '.join(f'{k} {m0[k]:.2f} -> {m1[k]:.2f}' for k in keys))
        print('   pictures better/worse (colour difference at the 90th percentile, 5 % + 0.3):', sum(y['dE_p90'] < x['dE_p90'] * 0.95 - 0.3 for x, y in zip(r0, r1)), '/', sum(y['dE_p90'] > x['dE_p90'] * 1.05 + 0.3 for x, y in zip(r0, r1)), '; saturated colours:', sum(y['dE_sat'] < x['dE_sat'] * 0.95 - 0.3 for x, y in zip(r0, r1)), '/', sum(y['dE_sat'] > x['dE_sat'] * 1.05 + 0.3 for x, y in zip(r0, r1)))
    if a.bundle:
        b = json.load(open(a.bundle)); b['LookTable'] = dict(hue=HUES, data=[round(float(v), 3) for v in table.ravel()]); json.dump(b, open(a.bundle, 'w'), separators=(',', ':')); print('stored LookTable in', a.bundle)
