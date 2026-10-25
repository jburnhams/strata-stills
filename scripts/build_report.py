"""A visual report on the candidate photos: each one rendered by our pipeline (with the picture kept at every stage), set beside the camera's own JPEG and the other raw tools' renders (darktable, RawTherapee) where candidates/renders/ has them, as crops from the places where the programs differ most, each shown in context on an overview of the picture. No full-size picture goes into the report, only an overview and the crops, so it stays a few MB.
  python scripts/build_report.py [--out out/report] [--set a7c --set dji] [--pictures 4 | --all | --stems DSC00014,...] [--quality 95] [--crops 3] [--refresh]
Reads candidates/<set>/ (scripts/copy_candidates.py) and docs/<set>_candidates.json; writes <out>/index.html and <out>/assets/*.jpg (gitignored with out/). The pictures of ours are made again on every run (a temporary project: nothing of the app's projects is touched); an optional step of the build (scripts/setup_env.sh --report), skipped when there are no candidate photos.
How the crops are chosen: every program's picture is brought to the size of ours and registered to it; the disagreement at each place is the colour and brightness difference (blurred, so that a one pixel shift does not count) plus the difference in how much fine detail and noise there is, each relative to its median over the picture; the best window of each kind that does not overlap an earlier one is shown (1: the two together, 2: the detail and noise, 3: the colour and tone)."""
import argparse, html, json, os, shutil, sys, tempfile, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import compare_detail as CD

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SETS = dict(a7c=dict(label='Sony A7C', ext='.ARW', camera_ext='.JPG', prefix='', note='Sony A7C raw (ARW)'), dji=dict(label='DJI drone', ext='.DNG', camera_ext='.JPG', prefix='dji-', note='DJI drone raw (DNG)'))
OTHERS = [('darktable', 'darktable'), ('rawtherapee-auto-matched', 'RawTherapee (auto-matched)'), ('rawtherapee', 'RawTherapee (default)')]   # (folder in candidates/renders, name shown)
OTHER_SETTINGS = {
    'Camera': "The camera's own JPEG (the one next to the raw): its in-camera processing, whatever the camera was set to; lens corrections and noise reduction are the camera's.",
    'darktable': 'darktable 5.6.2, darktable-cli with the default configuration (candidates/dtconf): the scene-referred workflow, its default modules and auto-applied presets, no styles; asked for 3000 px (the long side).',
    'RawTherapee (auto-matched)': "RawTherapee 5.13, rawtherapee-cli with its own 'Auto-Matched Curve' profile for the picture's ISO (Low up to ISO 400, Medium up to 1600, else High); full size (resized to ours for the comparison).",
    'RawTherapee (default)': "RawTherapee 5.13, rawtherapee-cli with its default profile; full size (resized to ours for the comparison).",
}
STAGE_NAMES = [('develop', 'Ours: developed'), ('level', 'Ours: levelled'), ('resize', 'Ours: resized'), ('sharpen', 'Ours: sharpened'), ('final', 'Ours: final JPEG')]
WIN = (200, 140)                # the window of a crop, in pixels of the final picture; shown at ZOOM times
ZOOM = 2
OVERVIEW = 1100
JPEG_Q = 80


def read(path, size=None):
    im = cv2.imread(path, cv2.IMREAD_COLOR) if os.path.exists(path) else None
    return im if im is None or size is None else cv2.resize(im, size, interpolation=cv2.INTER_AREA)


def shift_between(a, b, mask_sigma=3):
    """The translation (dx, dy) to apply to `b` to lay it on `a` (both gray float32, the same size): phase correlation of the band-passed pictures."""
    f = lambda x: cv2.GaussianBlur(x, (0, 0), 1.0) - cv2.GaussianBlur(x, (0, 0), 8.0)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(f(b).astype(np.float32), f(a).astype(np.float32), win)
    return dx, dy, resp


def register(a, b):
    """The 2x3 similarity matrix (scale, rotation, shift) that lays gray uint8 `b` on `a` (ORB features and RANSAC: the programs differ in framing by tens of pixels and a little in scale, which a pure shift cannot undo), or None."""
    orb = cv2.ORB_create(5000); ka, da = orb.detectAndCompute(a, None); kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20: return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da)
    if len(m) < 20: return None
    M, inl = cv2.estimateAffinePartial2D(np.float32([kb[x.queryIdx].pt for x in m]), np.float32([ka[x.trainIdx].pt for x in m]), method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return M if M is not None and int(inl.sum()) >= 20 else None


def shifted(im, dx, dy):
    return cv2.warpAffine(im, np.float32([[1, 0, dx], [0, 1, dy]]), (im.shape[1], im.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def disagreement(ref, others):
    """Two maps (colour/tone, detail/noise) of how much the other programs' pictures differ from `ref` (all BGR uint8, one size), each divided by its median."""
    lab = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 1.5)
    hp = lambda im: (lambda y: cv2.GaussianBlur(np.abs(y - cv2.GaussianBlur(y, (0, 0), 1.5)), (0, 0), 6.0))(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32))
    lr, hr = lab(ref), hp(ref); col = np.mean([np.abs(lab(o) - lr).sum(axis=2) for o in others], axis=0); det = np.mean([np.abs(hp(o) - hr) for o in others], axis=0)
    return col / max(float(np.median(col)), 1e-3), det / max(float(np.median(det)), 1e-3)


def pick_windows(col, det, n, win):
    """`n` windows (x0, y0, kind) of size `win` with the strongest disagreement, none overlapping: the best of the two maps together, then of the detail map, then of the colour map."""
    h, w = col.shape; ww, wh = win; pool = lambda m: cv2.blur(m, (ww, wh))
    maps = [('overall', pool(col) + pool(det)), ('detail and noise', pool(det)), ('colour and tone', pool(col))]; taken = []; out = []
    for i in range(n):
        kind, m = maps[i % len(maps)]; m = m.copy(); m[:wh // 2 + 8] = 0; m[-(wh // 2 + 8):] = 0; m[:, :ww // 2 + 8] = 0; m[:, -(ww // 2 + 8):] = 0
        for (cx, cy) in taken: m[max(0, cy - wh):cy + wh, max(0, cx - ww):cx + ww] = 0
        cy, cx = np.unravel_index(int(np.argmax(m)), m.shape); taken.append((cx, cy)); out.append((int(cx - ww // 2), int(cy - wh // 2), kind))
    return out


def label(im, text):
    """Draw the label on a panel (a dark band at the top)."""
    cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1); cv2.putText(im, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA); return im


def window(im, x0, y0, win):
    """The window (x0, y0, w, h) of a picture of any size, given in fractions of the final picture's size, shown at the panel size."""
    h, w = im.shape[:2]; fx, fy, fw, fh = x0, y0, win[0], win[1]
    xa, ya, xb, yb = int(round(fx[0] * w / fx[1])), int(round(fy[0] * h / fy[1])), int(round((fx[0] + fw) * w / fx[1])), int(round((fy[0] + fh) * h / fy[1]))
    crop = im[max(0, ya):max(ya + 1, yb), max(0, xa):max(xa + 1, xb)]
    return cv2.resize(crop, (win[0] * ZOOM, win[1] * ZOOM), interpolation=cv2.INTER_CUBIC if crop.shape[1] < win[0] * ZOOM else cv2.INTER_AREA)


def run_ours(folder, items_wanted, quality, workdir, say):
    """Our pipeline on the wanted pictures of `folder` (stems) in a temporary project: {stem: dict(record, stages={name: npy path of the picture at that stage (uint8 RGB)}, file)}."""
    os.environ['STRATASTILLS_PROJECTS'] = os.path.join(workdir, 'projects'); os.environ['STRATASTILLS_HOME'] = os.path.join(workdir, 'home')
    from stratastills import projects as P, pipeline as PL, export as X
    pid = P.create('report-' + os.path.basename(os.path.normpath(folder)), [os.path.abspath(folder)])['id']; sid = P.source_id(P.load(pid)['sources'][0]['path']); items = {i['stem']: i for i in P.source_items(pid, sid)}; res = {}
    for stem in items_wanted:
        it = items.get(stem)
        if not it: say(f'  {stem}: not found in {folder}'); continue
        dumps = {}; t0 = time.time()
        def on_stage(name, st, stem=stem, dumps=dumps):
            if name == 'encode': return
            f = os.path.join(workdir, f'{stem}-{name}.npy'); np.save(f, X.to_u8(st.rgb)); dumps[name] = f
        r = PL.generate(pid, it['id'], preset='best', options=dict(quality=quality), progress=lambda m: None, on_stage=on_stage)
        res[stem] = dict(rec=r, stages=dumps, file=r['file']); say(f'  {stem}: ours made in {time.time() - t0:.0f} s')
    return res


def settings_html(rec):
    """The stage settings and what each stage did, from the record the pipeline keeps with the picture."""
    rows = []
    for s in rec['stages']:
        name = s['stage']; params = ', '.join(f'{k} = {v}' for k, v in s.items() if k != 'stage'); rows.append(f'<tr><th>{html.escape(name)}</th><td>{html.escape(params)}</td></tr>')
    steps = ''.join(f'<li>{html.escape(str(x))}</li>' for x in rec.get('steps', []))
    extra = []
    if rec.get('tone'): extra.append('tone: ' + html.escape(json.dumps({k: v for k, v in rec['tone'].items() if isinstance(v, (str, int, float))})))
    if rec.get('size'): extra.append('size: ' + html.escape(f"{rec['size']['long_side']} px long side, {rec['size']['why']}"))
    if rec.get('level'): extra.append('level: ' + html.escape(json.dumps(rec['level'])))
    if rec.get('input'): extra.append('input: ' + html.escape(json.dumps({k: v for k, v in rec['input'].items() if isinstance(v, (str, int, float))})))
    return f"<table class=s>{''.join(rows)}</table><ul>{steps}</ul><p class=m>{'<br>'.join(extra)}</p>"


def metric_table(ims, names):
    ref = ims[0]; flat, tex = CD.masks(ref); keys = ('sharp', 'halo', 'ring', 'noise', 'cnoise', 'detail'); rows = [CD.measure(i, flat, tex) for i in ims]
    head = ''.join(f'<th>{k}</th>' for k in keys); body = ''.join(f"<tr><th>{html.escape(n)}</th>{''.join('<td>%.2f</td>' % r[k] for k in keys)}</tr>" for n, r in zip(names, rows))
    return f'<table class=t><tr><th></th>{head}</tr>{body}</table>'


CSS = """body{font:14px/1.45 system-ui,sans-serif;margin:0;background:#16181c;color:#d8dbe0}header{padding:18px 28px;background:#0e0f12}h1{margin:0 0 4px;font-size:22px}h2{font-size:18px;margin:34px 0 4px}h3{font-size:15px;margin:22px 0 6px}main{max-width:1500px;margin:0 auto;padding:0 28px 60px}a{color:#8ab4ff}nav a{margin-right:12px}
.pic{border-top:1px solid #333;margin-top:30px}.why{color:#9aa}.ov{position:relative;display:inline-block;line-height:0}.ov img{max-width:100%;height:auto}.box{position:absolute;border:2px solid #ffd54a;box-shadow:0 0 0 1px #000;color:#ffd54a;font:bold 13px sans-serif;line-height:1}.box b{position:absolute;left:0;top:-16px;background:#000;padding:1px 4px}
.crop{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;margin:10px 0}.sheet{display:grid;gap:4px}.sheet div{background-repeat:no-repeat;position:relative}.sheet span{position:absolute;left:0;top:0;right:0;background:#000b;color:#fff;font:12px sans-serif;padding:2px 6px}
table{border-collapse:collapse}.s th,.s td,.t th,.t td{padding:2px 10px;text-align:left;border-bottom:1px solid #2a2d33}.t td{text-align:right;font-variant-numeric:tabular-nums}.m{color:#9aa;font-size:12px}details{margin:8px 0}summary{cursor:pointer;color:#8ab4ff}ul{margin:6px 0 6px 18px;padding:0}.note{max-width:1000px;color:#aab}"""


def build(a):
    out = os.path.abspath(a.out); assets = os.path.join(out, 'assets'); shutil.rmtree(out, ignore_errors=True); os.makedirs(assets); say = lambda m: print(m, flush=True)
    work = tempfile.mkdtemp(prefix='report-'); sections = []; legend = {}; nav = []; total_pics = 0
    try:
        for setname in a.set:
            cfg = SETS[setname]; folder = os.path.join(ROOT, 'candidates', setname); cj = os.path.join(ROOT, 'docs', f'{setname}_candidates.json')
            if not os.path.isdir(folder) or not os.path.exists(cj): say(f'{setname}: no candidate photos here ({folder}), skipped'); continue
            cands = json.load(open(cj)); whys = {c['stem']: c.get('why', '') for c in cands}; stems = [c['stem'] for c in cands]
            if a.stems: stems = [s for s in a.stems.split(',') if s in whys or os.path.exists(os.path.join(folder, s + cfg['ext']))]
            elif not a.all: stems = stems[:a.pictures]
            say(f'{cfg["label"]}: {len(stems)} pictures'); ours = run_ours(folder, stems, a.quality, work, say)
            for stem in stems:
                if stem not in ours: continue
                o = ours[stem]; fin = read(o['file'])
                if fin is None: continue
                H, W = fin.shape[:2]; size = (W, H)
                cam = read(os.path.join(folder, stem + cfg['camera_ext']), size); methods = [('Camera', cam)]
                for d, nm in OTHERS:
                    im = read(os.path.join(ROOT, 'candidates', 'renders', cfg['prefix'] + d, stem + '.jpg'), size)
                    if im is not None: methods.append((nm, im))
                k = 1500 / max(W, H); small = lambda im: cv2.resize(im, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA); gfin = cv2.cvtColor(small(fin), cv2.COLOR_BGR2GRAY).astype(np.float32); aligned = []
                g8 = lambda im: cv2.cvtColor(small(im), cv2.COLOR_BGR2GRAY)
                for nm, im in methods:                                                       # the other programs' framing differs from ours: one similarity transform for the whole picture, then a translation for each crop (below)
                    M = register(g8(fin), g8(im))
                    if M is not None: M = M.copy(); M[:, 2] /= k; aligned.append((nm, cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)))
                    else: dx, dy, resp = shift_between(gfin, g8(im).astype(np.float32)); aligned.append((nm, shifted(im, dx / k, dy / k)))
                col, det = disagreement(small(fin), [small(im) for _, im in aligned]); sel = pick_windows(col, det, a.crops, (round(WIN[0] * k), round(WIN[1] * k)))
                # ---- panels: the others, then ours (final), then ours at each stage before it
                stage_ims = [(nm, np.load(o['stages'][s])[..., ::-1]) for s, nm in STAGE_NAMES[:-1] if s in o['stages']]
                panels = [(nm, im) for nm, im in aligned] + [(STAGE_NAMES[-1][1], fin)] + stage_ims
                ov = cv2.resize(fin, (OVERVIEW, round(OVERVIEW * H / W)), interpolation=cv2.INTER_AREA); cv2.imwrite(os.path.join(assets, f'{stem}-overview.jpg'), ov, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                blocks = []
                for n, (cx, cy, kind) in enumerate(sel, 1):
                    x0, y0 = int(round(cx / k)), int(round(cy / k)); x0 = min(max(0, x0), W - WIN[0]); y0 = min(max(0, y0), H - WIN[1]); tiles = []
                    pad = 24; ya, yb, xa, xb = max(0, y0 - pad), min(H, y0 + WIN[1] + pad), max(0, x0 - pad), min(W, x0 + WIN[0] + pad); gref = cv2.cvtColor(fin[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    for pi, (nm, im) in enumerate(panels):
                        if pi < len(aligned):                                                                       # the others: the last few pixels of registration, from this very place
                            dx, dy, resp = shift_between(gref, cv2.cvtColor(im[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.float32))
                            if resp > 0.05 and abs(dx) < pad and abs(dy) < pad: im = shifted(im, dx, dy)
                        t = window(im, (x0, W), (y0, H), WIN).copy(); tiles.append(label(t, nm))
                    cols = 3 if len(tiles) in (6, 9) else 4; rows = -(-len(tiles) // cols); pw, ph = WIN[0] * ZOOM, WIN[1] * ZOOM; sheet = np.zeros((rows * ph, cols * pw, 3), np.uint8)
                    for i, t in enumerate(tiles): sheet[(i // cols) * ph:(i // cols + 1) * ph, (i % cols) * pw:(i % cols + 1) * pw] = t
                    name = f'{stem}-crop{n}.jpg'; cv2.imwrite(os.path.join(assets, name), sheet, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                    blocks.append(dict(n=n, kind=kind, x=x0 / W, y=y0 / H, w=WIN[0] / W, h=WIN[1] / H, sheet=name, sheet_size=(cols * pw, rows * ph), names=[nm for nm, _ in panels], cols=cols, pw=pw, ph=ph))
                metrics = metric_table([im for _, im in aligned] + [fin], [nm for nm, _ in aligned] + ['Ours (final)'])
                legend.update({nm: OTHER_SETTINGS.get(nm, '') for nm, _ in aligned}); total_pics += 1
                sections.append(render_picture(cfg, stem, whys.get(stem, ''), W, H, o['rec'], blocks, metrics)); nav.append((stem, f'{cfg["label"]} {stem}'))
    finally: shutil.rmtree(work, ignore_errors=True)
    if not sections: say('nothing to report on (no candidate photos)'); return 1
    leg = ''.join(f'<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>' for k, v in legend.items())
    page = f"""<!doctype html><meta charset=utf-8><title>stratastills: candidate photos compared</title><style>{CSS}</style><header><h1>Candidate photos: our pipeline against the camera and other raw tools</h1>
<div class=note>Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/build_report.py. Each crop is a place where the programs disagree most, marked on the overview of the picture; the panels show the same place in every program and in ours at each stage. Crops are shown at {ZOOM}x of the final picture (the stages before the resize hold more pixels, so they look finer: the field of view is the same). Ours was made with the 'best' preset at JPEG quality {a.quality} (the quality of the comparison runs).</div>
<nav>{''.join(f'<a href="#{s}">{html.escape(t)}</a>' for s, t in nav)}</nav></header><main><h2>The other programs</h2><table class=s>{leg}</table>{''.join(sections)}</main>"""
    open(os.path.join(out, 'index.html'), 'w', encoding='utf-8').write(page)
    size = sum(os.path.getsize(os.path.join(assets, f)) for f in os.listdir(assets)) + os.path.getsize(os.path.join(out, 'index.html'))
    say(f'report: {os.path.join(out, "index.html")} ({total_pics} pictures, {size / 1e6:.1f} MB)'); return 0


def render_picture(cfg, stem, why, W, H, rec, blocks, metrics):
    boxes = ''.join(f'<div class=box style="left:{b["x"] * 100:.2f}%;top:{b["y"] * 100:.2f}%;width:{b["w"] * 100:.2f}%;height:{b["h"] * 100:.2f}%"><b>{b["n"]}</b></div>' for b in blocks); crops = []
    for b in blocks:
        tiles = ''.join(f'<div style="width:{b["pw"]}px;height:{b["ph"]}px;background-image:url(assets/{b["sheet"]});background-position:-{(i % b["cols"]) * b["pw"]}px -{(i // b["cols"]) * b["ph"]}px"></div>' for i in range(len(b['names'])))
        mini = f'<div class=ov style="width:300px"><img src="assets/{stem}-overview.jpg" width=300><div class=box style="left:{b["x"] * 100:.2f}%;top:{b["y"] * 100:.2f}%;width:{b["w"] * 100:.2f}%;height:{b["h"] * 100:.2f}%"></div></div>'
        crops.append(f'<h3>Crop {b["n"]}: where the programs differ most in {html.escape(b["kind"])}</h3><div class=crop>{mini}<div class=sheet style="grid-template-columns:repeat({b["cols"]},{b["pw"]}px)">{tiles}</div></div>')
    return f"""<section class=pic id="{stem}"><h2>{html.escape(cfg['label'])} {html.escape(stem)} <small class=why>{html.escape(why)} · {W}x{H} final</small></h2><div class=ov style="width:{OVERVIEW}px;max-width:100%"><img src="assets/{stem}-overview.jpg">{boxes}</div>
<details open><summary>What ours did, and the settings of each stage</summary>{settings_html(rec)}</details><details><summary>Numbers (scripts/compare_detail.py, on this picture)</summary>{metrics}<p class=m>sharp: edge crispness; ring/halo: sharpening artefacts; noise/cnoise: luminance and colour noise in flat areas; detail: texture kept against the noise.</p></details>{''.join(crops)}</section>"""


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'report')); ap.add_argument('--set', action='append', choices=sorted(SETS)); ap.add_argument('--pictures', type=int, default=4, help='per set (the first of the candidate list)')
    ap.add_argument('--all', action='store_true'); ap.add_argument('--stems'); ap.add_argument('--quality', type=int, default=95); ap.add_argument('--crops', type=int, default=3)
    a = ap.parse_args(); a.set = a.set or list(SETS); sys.exit(build(a))
