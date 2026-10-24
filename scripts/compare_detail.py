"""Detail and noise of renderings of the same raws, side by side: ours, darktable's, RawTherapee's and the camera's own JPEG, all at one size (the long side area-averaged to --size), on the candidate pictures (docs/*_candidates.json). The colour is judged elsewhere (parity_jpeg.py, parity_external.py); this is what a person sees in a 100 % crop.
  python scripts/compare_detail.py docs/a7c_candidates.json candidates/a7c candidates/renders/ours candidates/renders/darktable candidates/renders/rawtherapee-auto-matched [--size 3000] [--ext .JPG]
Per picture and program:
  sharp   the mean gradient (8-bit levels per pixel) of the strongest 2 % of the luminance edges: how crisp the edges are (noise and halos raise it too)
  halo    how far the luminance overshoots beside those edges (the mean of the largest overshoot over a 5 px neighbourhood, in levels): sharpening artefacts
  noise   the standard deviation (levels) of the luminance with its 1.5 px blur taken away, in the flat areas (where the camera JPEG, blurred, has almost no gradient)
  cnoise  the same for the two colour channels of CIELAB (the blotchy colour noise)
  detail  the same high-pass in the TEXTURED areas (gradient in the middle range) over the noise: how much of the texture is kept against the noise let through (higher is better)
The programs are named after the last folder of their path; the camera JPEG is the reference called 'camera'."""
import argparse, glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import parity_jpeg as PJ


def load(path, size):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: return None
    k = size / max(im.shape[:2]); return cv2.resize(im, (round(im.shape[1] * k), round(im.shape[0] * k)), interpolation=cv2.INTER_AREA)


def grad(y): gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3) / 8; return np.hypot(gx, gy)


def masks(ref_bgr):
    """(flat, textured) masks from the camera JPEG, the same for every program."""
    y = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32); g = cv2.GaussianBlur(grad(y), (0, 0), 2.0); mid = (y > 25) & (y < 225)
    return mid & (g < np.percentile(g[mid], 25)), mid & (g > np.percentile(g[mid], 60)) & (g < np.percentile(g[mid], 95))


def measure(bgr, flat, tex):
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32); g = grad(y); hp = y - cv2.GaussianBlur(y, (0, 0), 1.5); top = g >= np.percentile(g, 98)
    over = cv2.dilate(y, np.ones((5, 5), np.uint8)) - y; under = y - cv2.erode(y, np.ones((5, 5), np.uint8)); lab = PJ.lab(np.ascontiguousarray(bgr[..., ::-1])); ch = np.mean([(lab[..., c] - cv2.GaussianBlur(lab[..., c], (0, 0), 1.5))[flat].std() for c in (1, 2)])
    noise = float(hp[flat].std()); tex_hp = float(hp[tex].std()); return dict(sharp=float(g[top].mean()), halo=float(np.minimum(over, under)[top].mean()), noise=noise, cnoise=float(ch), detail=tex_hp / max(noise, 1e-3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('candidates'); ap.add_argument('cameras'); ap.add_argument('programs', nargs='+'); ap.add_argument('--size', type=int, default=3000); ap.add_argument('--ext', default='.JPG'); a = ap.parse_args()
    names = ['camera'] + [os.path.basename(os.path.normpath(p)) for p in a.programs]; dirs = [a.cameras] + a.programs; rows = {n: {} for n in names}; sel = [c['stem'] for c in json.load(open(a.candidates))]
    for s in sel:
        ims = []
        for n, d in zip(names, dirs):
            f = next(iter(glob.glob(os.path.join(d, s + '.[jJ][pP][gG]'))), None); ims.append(load(f, a.size) if f else None)
        if any(i is None for i in ims) or len({(i.shape[1] > i.shape[0]) for i in ims}) > 1: print(s, 'skipped (missing or turned differently)'); continue
        h = min(i.shape[0] for i in ims); w = min(i.shape[1] for i in ims); ims = [i[:h, :w] for i in ims]; flat, tex = masks(ims[0])
        for n, i in zip(names, ims): rows[n][s] = measure(i, flat, tex)
    keys = ('sharp', 'halo', 'noise', 'cnoise', 'detail'); common = sorted(set.intersection(*[set(r) for r in rows.values()]))
    print('%-28s' % 'median over %d pictures' % len(common), ' '.join('%8s' % k for k in keys))
    for n in names: print('%-28s' % n, ' '.join('%8.2f' % np.median([rows[n][s][k] for s in common]) for k in keys))
    def iso(s):
        import tifffile
        for ext in ('.ARW', '.DNG', '.arw', '.dng'):
            f = os.path.join(a.cameras, s + ext)
            if os.path.exists(f):
                try: return int(tifffile.TiffFile(f).pages[0].tags['ExifTag'].value['ISOSpeedRatings'])
                except Exception: return 0
        return 0
    isos = {s: iso(s) for s in common}
    for lo, hi_ in ((0, 250), (251, 800), (801, 2500), (2501, 6000), (6001, 10 ** 6)):
        grp = [s for s in common if lo <= isos[s] <= hi_]
        if len(grp) >= 3:
            print('\nISO %d-%d (%d pictures):' % (lo, hi_, len(grp)))
            for n in names: print('%-28s' % n, ' '.join('%8.2f' % np.median([rows[n][s][k] for s in grp]) for k in keys))
    json.dump(rows, open(os.path.join(os.path.dirname(a.programs[0]), 'detail_metrics.json'), 'w'), indent=1)
