"""Does a lens correction make the field flatter, and which source does it better? For each lens in a table of pictures (scripts/scan_exif.py), the pictures are developed (linear, half size) with no lens correction, with the camera's own tables (lenscorr.py) and with the lensfun profile (lensfundb.py), and the log2 luminance of each, relative to its centre, is followed over the radius; the median over the pictures is the falloff of the lens as these scenes see it (a scene is not flat, but many scenes together average out: a correct correction makes the median flat). Also prints the corner gain and the distortion at the corner each source predicts for the shots.
  python scripts/compare_lens_sources.py iso.jsonl --root "\\SERVER\Raw\2023" [--per 16] [--lens SUBSTRING]"""
import argparse, collections, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import develop as D, lenscorr as L, lensfundb as F

BINS = np.linspace(0, 1, 9)


def profile(path, variant):
    orig = L.read_tables
    if variant == 'lensfun': L.read_tables = lambda p: None
    try: lin = D.develop_parts(path, half=True, ev=0.0, recover=0.0, lens='off' if variant == 'off' else 'auto')['linear']
    finally: L.read_tables = orig
    y = (lin @ D._W)[::3, ::3]; h, w = y.shape; yy, xx = np.mgrid[0:h, 0:w]; r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2); ly = np.log2(np.maximum(y, 1e-4)); c = np.median(ly[r < 0.25])
    return [np.median(ly[(r >= lo) & (r < hi)]) - c for lo, hi in zip(BINS[:-1], BINS[1:])]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('table'); ap.add_argument('--root', required=True); ap.add_argument('--per', type=int, default=16); ap.add_argument('--lens'); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.table)]; by = collections.defaultdict(list)
    for r in rows: by[r['lens']].append(r)
    rng = random.Random(1)
    for lens, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if a.lens and a.lens.lower() not in lens.lower(): continue
        rng.shuffle(rs); rs = rs[:a.per]; paths = [os.path.join(a.root, r['folder'], r['file']) for r in rs]; prof = {v: [] for v in ('off', 'camera', 'lensfun')}; have_cam = have_lf = 0; corner = {'camera': [], 'lensfun': []}
        lf = F.find(lens)
        for p in paths:
            t = L.read_tables(p); sh = F.shot_of(p)
            prof['off'].append(profile(p, 'off'))
            if t: prof['camera'].append(profile(p, 'camera')); have_cam += 1; corner['camera'].append(float(np.log2(L.curves(t)[4][-1])))
            if lf and sh and sh[1]:
                prof['lensfun'].append(profile(p, 'lensfun')); have_lf += 1; g, _w = F.corrections(lf, 6000, 4000, sh[1], sh[2], sh[3], camera_crop=1.0)
                if g: corner['lensfun'].append(float(np.log2(g(np.array([1.0]))[0])))
        print('\n%s: %d pictures (camera tables in %d, lensfun profile %s in %d)' % (lens, len(paths), have_cam, 'found' if lf else 'none', have_lf))
        for v, rr in prof.items():
            if rr: m = np.median(rr, axis=0); print('  %-8s log2 luminance, centre -> corner: %s   spread over r > 0.3: %.2f stops' % (v, ' '.join('%+.2f' % x for x in m), m[2:].max() - m[2:].min()))
        for v, c in corner.items():
            if c: print('  %-8s predicted shading gain at the corner: median %.2f stops (%.2f .. %.2f)' % (v, np.median(c), min(c), max(c)))
