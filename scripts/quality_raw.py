"""How good is our output, judged against the RAW alone (no camera JPEG, no other program): one measure for each fault found on the visual report, so that a change that brings one back is caught on pictures that have nothing to compare with.
  python scripts/quality_raw.py candidates/a7c docs/a7c_candidates.json [--n 20] [--stems a,b] [--accept] [--json docs/quality_raw.json]
Each raw is made by the app's own generate (best preset, 3000 px, q95), with the picture kept at each stage (`on_stage`), and developed once more with no clean-up at all (`denoise='off'`): the REFERENCE, the raw rendered with the same tone and colour and nothing else. Per picture:
  curve_flat    share of the display-encoded input range the picture covers where the matched tone curve is flat (slope under 0.05): banding (lower is better)
  curve_steep   the largest slope of that curve (1 = a plain sRGB encoding): banding and amplified noise (lower is better)
  curve_top     the curve's output at the raw's clip: what clips must be white (1.0)
  hl_slope      the curve's slope (encoded in and out) over the picture's brightest 1 % (from its 99th to its 99.9th percentile): lamps, lit windows, a bright sky squeezed into a few levels when low (higher is better)
  colour        median colour (CIELAB chroma, blurred so that noise does not count) of the developed picture over the reference's, where the reference has colour: colour cored away with the noise when low (higher is better, 1 = all kept)
  colour_strong the same over the reference's most colourful 0.5 %: small strong colours (lamps, signs) spread away when low
  specks        isolated dark pixels per megapixel (more than 0.03 below their 3x3 median) in the reference's smooth areas of the developed picture (lower is better)
  pinned        share of textured pixels the sharpening has pinned to their 5x5 neighbourhood's brightest or darkest value, over the picture before sharpening: flat cells, stepped thin lines (lower is better)
  noise         flat-area noise of the final picture over the reference's (at the same size): the clean-up (lower is better)
  detail        texture of the final picture over the reference's: what the clean-up and sharpening keep (higher is better)
  sharp, halo, ring   compare_detail's measures on the final picture, with the flat and textured areas taken from the reference
With --accept the numbers are stored (per stem) as the ones to beat; without it each picture is compared with the stored numbers and anything worse by more than the tolerance is flagged (exit status 1 when anything is)."""
import argparse, json, os, sys, tempfile
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
HIGHER = ('curve_top', 'hl_slope', 'colour', 'colour_strong', 'detail', 'sharp')                     # more is better; for the rest less is
TOL = dict(curve_flat=(0.0, 0.01), curve_steep=(0.10, 0.2), curve_top=(0.0, 0.005), hl_slope=(0.10, 0.02), colour=(0.03, 0.01), colour_strong=(0.05, 0.02), specks=(0.20, 20), pinned=(0.15, 0.005), noise=(0.10, 0.02), detail=(0.05, 0.02), sharp=(0.05, 0.3), halo=(0.10, 0.5), ring=(0.10, 0.2))   # (relative, absolute): worse by more than both is flagged


def enc(v): return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.maximum(v, 1e-9) ** (1 / 2.4) - 0.055)
def dec(v): return np.where(v <= 0.04045, v / 12.92, ((np.maximum(v, 0) + 0.055) / 1.055) ** 2.4)
def grad(y): return np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)) / 8
def to8(rgb): return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
def shrink(im, long): k = long / max(im.shape[:2]); return im if k >= 1 else cv2.resize(im, (round(im.shape[1] * k), round(im.shape[0] * k)), interpolation=cv2.INTER_AREA)


def curve_measures(tone, ref):
    """The tone curve's own faults (it is pairs of linear input, display-encoded output), over the encoded input range the picture covers (from the reference: its luminance taken back through the curve)."""
    if not tone or not tone.get('curve'): return {}
    c = np.asarray(tone['curve'], float).reshape(-1, 2); x, y = c[:, 0], c[:, 1]; ex = enc(x)
    lin = np.interp(np.percentile(ref @ LUMA, [1, 99, 99.9]), y, x); lo, hi = enc(lin[0]), enc(lin[1])
    g = np.linspace(0, 1, 1001); gy = np.interp(dec(g), x, y); inside = (g[:-10] > lo) & (g[10:] < hi)                                             # 0.001 encoded steps
    s = (gy[10:] - gy[:-10]) / 0.01; flat = float((s[inside] < 0.05).mean()) if inside.any() else 0.0; steep = float(s[inside].max()) if inside.any() else 0.0   # slopes over 0.01 of encoded input: a jump over a narrow range of input is not averaged away
    e99, e999 = enc(lin[1]), enc(lin[2]); hl = (np.interp(lin[2], x, y) - np.interp(lin[1], x, y)) / (e999 - e99) if e999 - e99 > 0.01 else 1.0          # the curve over the brightest 1 % of the picture: lamps, lit windows, a bright sky squeezed into a few levels when low
    return dict(curve_flat=flat, curve_steep=steep, curve_top=float(np.interp(1.0, x, y)), hl_slope=float(hl))


def develop_measures(dev, ref):
    """The developed picture against the reference (the same geometry: both straight out of develop), at 2000 px so that the blurs mean the same on every camera."""
    dev, ref = shrink(dev, 2000), shrink(ref, 2000); yd, yr = dev @ LUMA * 255, ref @ LUMA * 255
    gr = cv2.GaussianBlur(grad(cv2.GaussianBlur(yr, (0, 0), 2.0)), (0, 0), 1.0)
    lab = lambda im: cv2.cvtColor(np.ascontiguousarray(cv2.GaussianBlur(np.clip(im, 0, 1).astype(np.float32), (0, 0), 3.0)), cv2.COLOR_RGB2LAB)          # float: L 0..100, a and b signed
    ld, lr = lab(dev), lab(ref); cd, cr = np.hypot(ld[..., 1], ld[..., 2]), np.hypot(lr[..., 1], lr[..., 2])
    has = (cr > 6) & (lr[..., 0] > 4); strong = cr >= np.percentile(cr, 99.5)
    colour = float(np.median(cd[has] / cr[has])) if has.sum() > 500 else 1.0; colour_strong = float(np.median(cd[strong] / np.maximum(cr[strong], 1e-3)))
    flat = (gr < 0.5) & (yr > 10) & (yr < 245); yd8 = yd / 255.0; dip = (cv2.medianBlur(yd8.astype(np.float32), 3) - yd8) > 0.03
    specks = float((dip & flat).sum() / max(flat.sum(), 1) * 1e6)
    return dict(colour=colour, colour_strong=colour_strong, specks=specks)


def final_measures(pre, fin, ref):
    """The sharpening (the picture before and after it, the same geometry) and the final picture's noise, detail and edges against the reference brought to the final size."""
    import compare_detail as CD
    ypre, yfin = pre @ LUMA * 255, fin @ LUMA * 255; g = cv2.GaussianBlur(grad(ypre), (0, 0), 1.0); mid = (ypre > 10) & (ypre < 245)
    tex = mid & (g > np.percentile(g[mid], 60)) & (g < np.percentile(g[mid], 95)); k = np.ones((5, 5), np.uint8); hi, lo = cv2.dilate(ypre, k), cv2.erode(ypre, k)
    pin = lambda y: float(((np.abs(y - hi) < 0.5) | (np.abs(y - lo) < 0.5))[tex].mean()); pinned = pin(yfin) - pin(ypre)
    r = cv2.resize(ref, (fin.shape[1], fin.shape[0]), interpolation=cv2.INTER_AREA); bgr = lambda im: np.ascontiguousarray(to8(im)[..., ::-1]); flat, tx = CD.masks(bgr(r))
    m, mr = CD.measure(bgr(fin), flat, tx), CD.measure(bgr(r), flat, tx)
    return dict(pinned=max(pinned, 0.0), noise=m['noise'] / max(mr['noise'], 1e-3), detail=m['detail'] / max(mr['detail'], 1e-3), sharp=m['sharp'], halo=m['halo'], ring=m['ring'])


def measure(pid, item, path):
    from stratastills import pipeline as PL, develop as D
    st = {}
    def on_stage(name, s):
        if name == 'develop': st['dev'] = s.rgb.copy()
        elif name == 'resize': st['pre'] = s.rgb.copy()
        elif name == 'sharpen': st['fin'] = s.rgb.copy()
        st['tone'] = s.rec.get('tone')
    PL.generate(pid, item['id'], preset='best', options=dict(quality=95, size='3000'), progress=lambda m: None, on_stage=on_stage)
    ref = D.develop(path, ev='auto', tone='match', recover='auto', denoise='off'); fin = st.get('fin', st['pre'])
    out = curve_measures(st['tone'], ref); out.update(develop_measures(st['dev'], ref)); out.update(final_measures(st['pre'], fin, ref)); return out


def worse(now, was):
    out = []
    for k, (rel, ab) in TOL.items():
        if k not in now or k not in was: continue
        d = (was[k] - now[k]) if k in HIGHER else (now[k] - was[k])
        if d > ab and d > rel * abs(was[k]): out.append(f'{k} {was[k]:.3f} -> {now[k]:.3f}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('candidates'); ap.add_argument('--n', type=int); ap.add_argument('--stems'); ap.add_argument('--accept', action='store_true'); ap.add_argument('--json', default=os.path.join(os.path.dirname(__file__), '..', 'docs', 'quality_raw.json')); a = ap.parse_args()
    tmp = tempfile.mkdtemp(); os.environ['STRATASTILLS_PROJECTS'] = os.path.join(tmp, 'projects'); os.environ['STRATASTILLS_HOME'] = os.path.join(tmp, 'home')
    from stratastills import projects as P
    stems = a.stems.split(',') if a.stems else [c['stem'] for c in json.load(open(a.candidates))][:a.n]
    pid = P.create('quality', [os.path.abspath(a.folder)])['id']; sid = P.source_id(P.load(pid)['sources'][0]['path']); items = {i['stem']: i for i in P.source_items(pid, sid)}
    store = json.load(open(a.json)) if os.path.exists(a.json) else {}; rows, flagged = {}, 0
    for s in stems:
        it = items.get(s)
        if not it: print(s, 'missing'); continue
        try: m = measure(pid, it, it['path'] if 'path' in it else os.path.join(a.folder, next(f for f in os.listdir(a.folder) if os.path.splitext(f)[0] == s and os.path.splitext(f)[1].lower() in ('.arw', '.dng', '.cr2', '.nef'))))
        except Exception as e: print(s, 'failed:', e); continue
        rows[s] = m; bad = worse(m, store[s]) if s in store else []; flagged += bool(bad)
        print(s, ' '.join(f'{k} {v:.3f}' for k, v in m.items()), ('  WORSE: ' + ', '.join(bad)) if bad else '', flush=True)
    if rows:
        keys = list(dict.fromkeys(k for r in rows.values() for k in r)); print('MEDIAN', ' '.join(f'{k} {np.median([r[k] for r in rows.values() if k in r]):.3f}' for k in keys))     # a picture with no matched tone (no usable preview) has no curve measures
    if a.accept: store.update({s: {k: round(v, 5) for k, v in m.items()} for s, m in rows.items()}); json.dump(store, open(a.json, 'w'), indent=1, sort_keys=True); print('stored', len(rows), 'in', a.json)
    print(f'{flagged} of {len(rows)} worse than the stored numbers' if store else 'no stored numbers yet: run with --accept'); sys.exit(1 if flagged and not a.accept else 0)
