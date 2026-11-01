"""The person's own edits as training data: raws paired with the finished Lightroom exports of them, with the Lightroom settings each export carries.
  python scripts/lr_dataset.py pairs      [--finals ~/personal/Backup/Photos] [--raws ~/raw]   finals named after their raw (DSC01234*.jpg in <finals>/<year>/...) paired with the raw of that stem and year
  python scripts/lr_dataset.py catalogue                                                         every paired final's Lightroom settings (the crs: XMP at the head of the JPEG; no exiftool: 21k files in 8 minutes over SMB)
  python scripts/lr_dataset.py features                                                          per pair: stats of the camera's preview cropped as the final was, EXIF, and the final's lightness and colour (resumable, batches of 250)
Everything goes to candidates/personal/ (gitignored, on the data share): these are private photos, never committed and never in a public report. Findings: docs/progress.md, "Learning the person's own edits".
Lightroom's crop (crs:CropTop/Left/Bottom/Right, CropAngle) is in the frame of the sensor: rotate by +CropAngle about the crop's centre, cut the box, then turn by the EXIF orientation (99 % of 160 crops matched the export, against 74 % with the other sign)."""
import argparse, collections, json, math, os, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'candidates', 'personal')
RAW_EXTS = ('.arw', '.cr2', '.dng', '.nef')
STEM = re.compile(r'(DSC\d{5}|_MG_\d{4}|IMG_\d{4}|DJI_\d{4})', re.I)
PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)
FINAL_L = [f'L_p{p}' for p in PCTS] + ['L_mean']                       # the final's lightness (CIELAB L, 0-100) at these percentiles, and its mean
FINAL_C = ['a_mean', 'b_mean', 'C_p50', 'C_p90', 'C_mean', 'C_p99']    # its colour: mean a/b (the cast) and chroma percentiles
BATCH = 250


def num(v, d=np.nan):
    try: return float(v)
    except (TypeError, ValueError): return d


def pairs(a):
    raws = collections.defaultdict(list)
    for root, _, fs in os.walk(a.raws):
        for f in fs:
            if os.path.splitext(f)[1].lower() in RAW_EXTS: raws[os.path.splitext(f)[0].upper()].append(os.path.join(root, f))
    out, first = [], {}
    for root, _, fs in os.walk(a.finals):
        year = os.path.relpath(root, a.finals).split(os.sep)[0]
        for f in sorted(fs):
            s, e = os.path.splitext(f); m = STEM.match(s)
            if e.lower() not in ('.jpg', '.jpeg', '.tif', '.tiff') or not m or '-Pano' in s or os.sep + 'small' in root.lower(): continue      # panoramas are stitched from several raws; small/ holds resized copies of the same edit
            c = [p for p in raws.get(m.group(1).upper(), []) if f'{os.sep}{year}{os.sep}' in p]                                            # file numbers repeat over the years: same stem and same year
            if c: first.setdefault(c[0], os.path.join(root, f))
    out = [dict(final=f, raw=r) for r, f in first.items()]
    json.dump(out, open(os.path.join(OUT, 'pairs.json'), 'w')); print(f'{len(out)} raws with a final')


def _xmp(item):
    f = item['final']; row = dict(item)
    try:
        with open(f, 'rb') as fh: b = fh.read(1 << 20)
    except OSError as e: row['error'] = str(e); return row
    i = b.find(b'<x:xmpmeta'); j = b.find(b'</x:xmpmeta>', i)
    if i < 0 or j < 0: row['error'] = 'no xmp'; return row
    x = b[i:j].decode('utf8', 'replace'); row.update(re.findall(r'crs:(\w+)="([^"]*)"', x))
    for el in set(re.findall(r'<crs:(\w+)>', x)): row.setdefault(el, 'present')                 # masks, curves: element form; only their presence is kept
    return row


def catalogue(a):
    items = json.load(open(os.path.join(OUT, 'pairs.json')))
    with ThreadPoolExecutor(16) as ex: rows = list(ex.map(_xmp, items))
    json.dump(rows, open(os.path.join(OUT, 'catalogue.json'), 'w')); print(f'{len(rows)} finals, {sum("error" in r for r in rows)} unreadable')


def load_catalogue():
    """The pairs with process-2012 settings (all but a handful), in a fixed order: the `idx` of the feature batches indexes this list."""
    return [r for r in json.load(open(os.path.join(OUT, 'catalogue.json'))) if 'error' not in r and 'Exposure2012' in r]


def orient(im, o):
    return {6: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), 8: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE), 3: cv2.rotate(im, cv2.ROTATE_180)}.get(int(o or 1), im)


def unorient(im, o):
    return {6: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE), 8: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), 3: cv2.rotate(im, cv2.ROTATE_180)}.get(int(o or 1), im)


def lr_crop(im, x):
    """The part of `im` (in the sensor's frame, not yet oriented) that Lightroom's crop `x` (a catalogue row) kept."""
    if str(x.get('HasCrop')) not in ('1', 'True'): return im
    t, l, b, r, ang = (num(x.get(k), d) for k, d in (('CropTop', 0), ('CropLeft', 0), ('CropBottom', 1), ('CropRight', 1), ('CropAngle', 0)))
    h, w = im.shape[:2]; cx, cy = w * (l + r) / 2, h * (t + b) / 2; dx, dy = w * (r - l), h * (b - t)
    if ang:                                                                                       # the two corners are opposite corners of the turned box: its own width and height along the turned axes
        im = cv2.warpAffine(im, cv2.getRotationMatrix2D((cx, cy), ang, 1.0), (w, h), flags=cv2.INTER_AREA); th = math.radians(ang); c, s = math.cos(th), math.sin(th); dx, dy = abs(dx * c + dy * s), abs(-dx * s + dy * c)
    out = im[max(0, int(round(cy - dy / 2))):int(round(cy + dy / 2)), max(0, int(round(cx - dx / 2))):int(round(cx + dx / 2))]
    return out if out.size else im


def img_feats(im):
    """Stats of a picture (BGR uint8, any size) that tell how it is exposed and what kind of scene it is: lightness percentiles and histogram, clipping, colour cast and strength, local contrast, where the light is."""
    im = cv2.resize(im, (384, max(8, round(384 * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0] * 100 / 255; a = lab[..., 1] - 128; b = lab[..., 2] - 128
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32); mx = im.max(axis=2); c = np.hypot(a, b)
    f = {f'L_p{p}': float(np.percentile(L, p)) for p in PCTS}
    f.update(L_mean=L.mean(), L_std=L.std(), clip_hi=(mx >= 250).mean(), near_hi=(g >= 235).mean(), clip_lo=(g <= 5).mean(), near_lo=(g <= 20).mean(), a_mean=a.mean(), b_mean=b.mean(), chroma_mean=c.mean(), chroma_p90=np.percentile(c, 90))
    f.update({f'Lh{i}': v for i, v in enumerate(np.histogram(L, 16, (0, 100))[0] / L.size)})
    base = cv2.GaussianBlur(L, (0, 0), 12); f['local_contrast'] = np.abs(L - base).mean(); f['base_spread'] = np.percentile(base, 95) - np.percentile(base, 5); f['lap'] = np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean()
    H, W = L.shape; f['top_minus_bottom'] = L[:H // 3].mean() - L[-H // 3:].mean(); f['centre_minus_edge'] = L[H // 4:-H // 4, W // 4:-W // 4].mean() - L.mean()
    hue = np.degrees(np.arctan2(b, a)) % 360; f.update({f'hue{i}': v for i, v in enumerate(np.histogram(hue, 12, (0, 360), weights=c)[0] / max(c.sum(), 1e-6))})
    return {k: float(v) for k, v in f.items()}


def raw_feats(e):
    t = num(e.get('ExposureTime'))
    return dict(log_iso=math.log2(max(num(e.get('ISO'), 100), 1)), log_t=math.log2(t) if t == t and t > 0 else np.nan, fnum=num(e.get('FNumber')), lv=num(e.get('LightValue')), bv=num(e.get('BrightnessValue')),
                flash=float(int(num(e.get('Flash'), 0)) & 1), f35=num(e.get('FocalLengthIn35mmFormat')), ev_comp=num(e.get('ExposureCompensation')), a7c=float(e.get('Model') == 'ILCE-7C'), year=num(str(e.get('DateTimeOriginal', ''))[:4]))


def final_stats(im):
    """The FINAL_L and FINAL_C numbers of a picture (BGR uint8)."""
    im = cv2.resize(im, (384, max(8, round(384 * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0] * 100 / 255; a, b = lab[..., 1] - 128, lab[..., 2] - 128; c = np.hypot(a, b)
    return [float(np.percentile(L, p)) for p in PCTS] + [float(L.mean())], [float(a.mean()), float(b.mean()), float(np.percentile(c, 50)), float(np.percentile(c, 90)), float(c.mean()), float(np.percentile(c, 99))]


EXIF_TAGS = ['-ISO', '-ExposureTime', '-FNumber', '-LightValue', '-BrightnessValue', '-Flash', '-FocalLengthIn35mmFormat', '-ExposureCompensation', '-Model', '-DateTimeOriginal', '-Orientation']


def camera_picture(raw, tmp=None):
    """The camera's own rendering of a raw (BGR uint8, in the sensor's frame): the JPEG beside it when the camera wrote one, else the preview inside the raw."""
    for ext in ('.JPG', '.jpg'):
        j = os.path.splitext(raw)[0] + ext
        if os.path.exists(j): return unorient(cv2.imread(j), (exif_of([raw])[raw].get('Orientation')))       # the camera's JPEG is stored the way up: back to the sensor's frame, as the preview is
    b = subprocess.run(['exiftool', '-b', '-PreviewImage', raw], capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None


def exif_of(raws):
    return {x['SourceFile']: x for x in json.loads(subprocess.run(['exiftool', '-q', '-json', '-n', *EXIF_TAGS, *raws], capture_output=True, text=True).stdout or '[]')}


def features(a):
    rows = load_catalogue(); d = os.path.join(OUT, 'features'); os.makedirs(d, exist_ok=True)
    for k in range(0, len(rows), BATCH):
        dst = os.path.join(d, f'b{k // BATCH:03d}.npz')
        if os.path.exists(dst): continue
        batch = rows[k:k + BATCH]; tmp = tempfile.mkdtemp(dir=d); lst = os.path.join(tmp, 'list.txt'); open(lst, 'w').write('\n'.join(r['raw'] for r in batch))
        subprocess.run(['exiftool', '-q', '-b', '-PreviewImage', '-w', f'{tmp}/%:1d_%f.jpg', '-@', lst], capture_output=True)
        ex = {x['SourceFile']: x for x in json.loads(subprocess.run(['exiftool', '-q', '-json', '-n', *EXIF_TAGS, '-@', lst], capture_output=True, text=True).stdout or '[]')}
        def fin(r):
            im = cv2.imread(r['final'], cv2.IMREAD_REDUCED_COLOR_4); return None if im is None else final_stats(im)
        with ThreadPoolExecutor(8) as pool: fins = list(pool.map(fin, batch))
        X, F, C, keep, names = [], [], [], [], None
        for i, (r, fs) in enumerate(zip(batch, fins)):
            p = f'{tmp}{os.path.dirname(r["raw"])}/_{os.path.splitext(os.path.basename(r["raw"]))[0]}.jpg'; e = ex.get(r['raw']); im = cv2.imread(p) if os.path.exists(p) else None
            if im is None or e is None or fs is None: continue
            ft = img_feats(orient(lr_crop(im, r), e.get('Orientation'))); ft.update(raw_feats(e)); names = names or list(ft)
            X.append([ft[n] for n in names]); F.append(fs[0]); C.append(fs[1]); keep.append(k + i)
        np.savez(dst, X=np.array(X, np.float32), F=np.array(F, np.float32), C=np.array(C, np.float32), idx=np.array(keep), names=np.array(names or []))
        shutil.rmtree(tmp); print(f'batch {k // BATCH}: {len(keep)}/{len(batch)}', flush=True)


def load_features():
    """(X, names, F, C, idx) over every batch: features, the finals' lightness and colour, and the catalogue row of each."""
    import glob
    parts = [np.load(p, allow_pickle=True) for p in sorted(glob.glob(os.path.join(OUT, 'features', 'b*.npz')))]
    return (np.concatenate([p['X'] for p in parts]), [str(n) for n in parts[0]['names']], np.concatenate([p['F'] for p in parts]), np.concatenate([p['C'] for p in parts]), np.concatenate([p['idx'] for p in parts]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('step', choices=['pairs', 'catalogue', 'features'])
    ap.add_argument('--finals', default=os.path.expanduser('~/personal/Backup/Photos')); ap.add_argument('--raws', default=os.path.expanduser('~/raw'))
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); {'pairs': pairs, 'catalogue': catalogue, 'features': features}[a.step](a)
