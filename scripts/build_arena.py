"""Make the pictures of the rating arena (/arena/ on the server): for pictures from shoots the models were not trained on, every version in scripts/arena_variants.py (the person's Lightroom final, our neutral picture, auto1, auto2 ...), all on the final's framing, at DISPLAY px on the long side, into out/arena/img/<picture>/<version>.jpg, with out/arena/pool.json listing them (the server reads it as it grows, so voting can start before this finishes).
  python scripts/build_arena.py [--n 60] [--seed 7] [--redo auto3] [--pic 2017-08-19_DSC04866] [--retire NAME] [--unretire NAME]
Pictures are taken round-robin over the held-out shoots in a fixed random order, so the pool covers the whole range and a later run continues where this one stopped. Resumable and incremental: a version is made again only when its code (arena_variants.py, lr_look.py ...) changed or with --redo; a new version in VARIANTS is made for every picture already in the pool. Needs the pictures' neutral develops (cached in out/cache/ by build_lr_report.py, made here when missing: 20-40 s each).
PRIVATE: your own pictures, never public."""
import argparse, hashlib, json, os, sys, tempfile, time, shutil, types
import threading
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import arena_sched as AS, arena_variants as AV, build_lr_report as LRR, lr_dataset as D, lr_look as LK, lr_tmodel as TM, lr_ftarget as FT, lr_face_first as FF
from stratastills import settings

DISPLAY = 1400
JPEG_Q = 88
DEPS = {'lightroom': [], 'camera': [], 'neutral': []}                                   # the files a version depends on besides its own function: the learned looks depend on the models' code
LOOK = ['lr_look.py', 'lr_tmodel.py', 'lr_ftarget.py', 'lr_fit.py']


def code_key(name):
    """A version is made again when its own function, the brightness helper or (for the automatic ones) the look code changed, not when another version did."""
    import inspect
    h = hashlib.sha1(name.encode()); h.update(b'framing 5'); h.update(inspect.getsource(AV.VARIANTS[name]).encode())
    h.update(inspect.getsource(LRR.to_final).encode()); h.update(inspect.getsource(LRR.turn_to_ref).encode())                                  # the framing code itself, so a fix to it (the square-crop turn) reaches pictures already in the pool
    if name in ('camera', 'neutral'): h.update(inspect.getsource(AV.match_brightness).encode())
    if name in ('camera', 'neutral') or name.startswith('auto'): h.update(inspect.getsource(AV.auto1).encode())
    for f in DEPS.get(name, LOOK): h.update(open(os.path.join(os.path.dirname(__file__), f), 'rb').read())
    if name == 'auto6':                                                                                              # face first: its code, the skin-target models and the faces it reads
        import lr_face_first as FF
        for f in (os.path.join(os.path.dirname(__file__), 'lr_face_first.py'), FF.MODEL_FILE, FF.FACES_FILE): h.update(open(f, 'rb').read() if os.path.exists(f) else b'missing')
    if name in ('auto4', 'auto5', 'auto6'):                                                                                    # the neutral-trained network: its code and its saved weights
        import lr_neutral_model as NM
        h.update(open(os.path.join(os.path.dirname(__file__), 'lr_neutral_model.py'), 'rb').read()); h.update(open(NM.NET_FILE, 'rb').read() if os.path.exists(NM.NET_FILE) else b'no net')
    return h.hexdigest()[:12]


def smear_rect(img, frac=0.3, floor=0.35, max_frac=0.08, margin=6):
    """(x0, y0, x1, y1) of `img` (BGR) without the bands along its edges that only repeat the last real column or row (what our own develop leaves where its lens correction pulls in pixels from outside the frame): in such a band the picture does not change along the edge-to-centre direction at all (mean absolute difference between neighbouring columns or rows is 0.0, a real picture's is several grey levels). Only a band found is cut, with a margin for the soft ramp where it meets the picture."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); h, w = g.shape; dx, dy = np.abs(np.diff(g, axis=1)).mean(0), np.abs(np.diff(g, axis=0)).mean(1)
    def band(d, n):
        thr = min(frac * float(np.median(d[len(d) // 4: 3 * len(d) // 4])), floor); lim = int(max_frac * n); a = 0
        while a < lim and d[a] < thr: a += 1
        b = 0
        while b < lim and d[len(d) - 1 - b] < thr: b += 1
        return (a + margin if a >= 3 else 0), (b + margin if b >= 3 else 0)
    l, r = band(dx, w); t, b = band(dy, h); return l, t, w - r, h - b


def order(rows, idx, G, test, seed):
    """The held-out pictures, round-robin over shoots in a fixed random order: [(position in idx, picture id)]."""
    import random
    rng = random.Random(seed); by = {}
    for i in np.where(test)[0]: by.setdefault(G[i], []).append(i)
    for v in by.values(): rng.shuffle(v)
    shoots = sorted(by); rng.shuffle(shoots); out = []
    while any(by.values()):
        for s in shoots:
            if by[s]: i = by[s].pop(); row = rows[idx[i]]; out.append((i, f'{os.path.basename(os.path.dirname(row["final"]))[:10]}_{os.path.splitext(os.path.basename(row["raw"]))[0]}'.replace(' ', '')))
    return out


def _gpu_slots():
    try:
        from stratastills import gpu
        return gpu.slots() if gpu.available() else 1
    except Exception: return 1


def save_json(path, obj):
    tmp = path + '.tmp'; json.dump(obj, open(tmp, 'w'), indent=1); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--n', type=int, default=60, help='how many pictures to have in the pool at the end'); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--redo', action='append', default=[]); ap.add_argument('--pic', action='append', default=[], help='make every version of this picture again (repeatable)'); ap.add_argument('--only', action='store_true', help='make nothing but the --pic pictures'); ap.add_argument('--picks', help='a JSON file ({"pics": [...]} or a list) naming exactly the pictures the pool should hold: the others are dropped from the pool (their votes stay), the missing ones are made'); ap.add_argument('--retire', action='append', default=[]); ap.add_argument('--unretire', action='append', default=[]); ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--serial', action='store_true', help='one picture after the other in this thread (the reference the scheduler is compared with)'); ap.add_argument('--gpu-workers', type=int, help='chunks developed at a time (default 2; gpu.slot() keeps the card memory in bounds; STRATASTILLS_ARENA_GPU_WORKERS)'); ap.add_argument('--cpu-workers', type=int, help='workers making versions besides (default: half the cores, at most 6; STRATASTILLS_ARENA_CPU_WORKERS)'); ap.add_argument('--chunk', type=int, default=3, help='pictures per neutral develop job')
    a = ap.parse_args(); root = settings.arena_dir(); os.makedirs(os.path.join(root, 'img'), exist_ok=True); say = lambda m: print(m, flush=True)
    vf = os.path.join(root, 'variants.json'); var = json.load(open(vf)) if os.path.exists(vf) else {'retired': []}
    if a.retire or a.unretire: var['retired'] = sorted((set(var['retired']) | set(a.retire)) - set(a.unretire)); save_json(vf, var); say(f'retired: {var["retired"]}'); return 0
    pf = os.path.join(root, 'pool.json'); pool = json.load(open(pf)) if os.path.exists(pf) else {}
    rows, X, names, Y, idx, G = LK.load(); test = np.array([LK.test_shoot(g) for g in G]); FACES = json.load(open(os.path.join(D.OUT, 'faces.json'))) if os.path.exists(os.path.join(D.OUT, 'faces.json')) else {}
    models, tmodels, fmodels = LK.train_cached(X[~test], Y[~test]), TM.train_cached(), FT.train_cached(); keys = {n: code_key(n) for n in AV.VARIANTS}
    todo = []; valid = {sid for _, sid in order(rows, idx, G, test, a.seed)}
    picks = None
    if a.picks: d = json.load(open(a.picks)); picks = set(d['pics'] if isinstance(d, dict) else d); valid &= picks; say(f'picks: {len(picks)} pictures named, {len(picks - valid)} of them not held-out pictures of the dataset (skipped)')
    gone = [k for k in pool if k not in valid]
    if gone: pool = {k: v for k, v in pool.items() if k in valid}; save_json(pf, pool); say(f'dropped {len(gone)} pictures from the pool: ' + ('not in --picks or ' if picks else '') + f'no longer in the dataset (the nudity filter flags them): {gone}')           # votes on them stay in votes.jsonl but the server no longer offers them
    for i, sid in order(rows, idx, G, test, a.seed):
        have = pool.get(sid, {}); need = [n for n in AV.VARIANTS if n in a.redo or sid in a.pic or n not in have or have[n].get('key') != keys[n]]
        if (sid in picks) if picks is not None else (sid in pool or len(pool) + len([t for t in todo if t[1] not in pool]) < a.n):
            if need and (not a.only or sid in a.pic): todo.append((i, sid, need))
    say(f'{len(pool)} pictures in the pool; {len(todo)} to make or update')
    work = tempfile.mkdtemp(prefix='arena-', dir=os.path.join(root, '..')); lock = threading.Lock(); predict_lock = threading.Lock()                    # the pool dict and its file, and xgboost's predict, one thread at a time

    def predictions(i, fl):
        with predict_lock: return dict(pred=LK.predict(models, X[i:i + 1])[0], colour=TM.predict(tmodels, X[i:i + 1])[0], face=dict(box=fl[0]['box'], target=FT.face_target(fmodels, X[i], fl[0])) if fl else None)

    def produce(chunk):
        """The neutral develop of a chunk of pictures (GPU-heavy): [(item, developed pictures, exif)], one per picture."""
        raws = [rows[idx[i]]['raw'] for i, _, _ in chunk]; dev = LRR.neutral(raws, work, say, a.rebuild); ex = D.exif_of(raws); return [(item, dev, ex) for item in chunk]

    def make(job):
        """The versions of one picture (CPU-heavy: registration, the look, JPEG)."""
        (i, sid, need), dev, ex = job
        row = rows[idx[i]]; raw = row['raw']; fin = cv2.imread(row['final']); t0 = time.time()
        if fin is None or raw not in dev: say(f'  {sid}: skipped (missing picture)'); return
        k = DISPLAY / max(fin.shape[:2]); fin = cv2.resize(fin, (round(fin.shape[1] * k), round(fin.shape[0] * k)), interpolation=cv2.INTER_AREA)
        neu, _, r1 = LRR.to_final(D.unorient(dev.pop(raw)[0], ex.get(raw, {}).get('Orientation', 1)), row, ex.get(raw, {}).get('Orientation', 1), fin, with_rect=True); fl = FACES.get(str(int(idx[i])), [])
        camera = D.camera_picture(raw); cam, _, r2 = LRR.to_final(camera, row, ex.get(raw, {}).get('Orientation', 1), fin, with_rect=True) if camera is not None else (neu, '', r1)
        x0, y0, x1, y1 = max(r1[0], r2[0]), max(r1[1], r2[1]), min(r1[2], r2[2]), min(r1[3], r2[3]); sx0, sy0, sx1, sy1 = smear_rect(neu); x0, y0, x1, y1 = max(x0, sx0), max(y0, sy0), min(x1, sx1), min(y1, sy1); H, W = fin.shape[:2]
        fl = [dict(f, box=[(f['box'][0] * W - x0) / (x1 - x0), (f['box'][1] * H - y0) / (y1 - y0), f['box'][2] * W / (x1 - x0), f['box'][3] * H / (y1 - y0)]) for f in fl]
        fin, neu, cam = fin[y0:y1, x0:x1], neu[y0:y1, x0:x1], cam[y0:y1, x0:x1]                          # every version on the part where all of them have real pixels: the moved pictures smear their last row and column outside it, and our own develop repeats its edge pixels where the lens correction pulls from outside the frame (streaks on the edges, smear_rect); the person's final is cut the same way so that nothing shows the difference
        f2 = FF.main_face(idx[i])
        if f2 is not None:                                                                                         # on the same cut as the versions: box and landmarks into its frame
            sx, sy = W / (x1 - x0), H / (y1 - y0); ox, oy = x0 / W, y0 / H
            f2 = dict(f2, box=[(f2['box'][0] - ox) * sx, (f2['box'][1] - oy) * sy, f2['box'][2] * sx, f2['box'][3] * sy], lm=(np.asarray(f2['lm']) - [ox, oy]) * [sx, sy])
        c = types.SimpleNamespace(fin=fin, neu=neu, cam=cam, x=X[i], names=names, face2=f2, **predictions(i, fl))
        d = os.path.join(root, 'img', sid); os.makedirs(d, exist_ok=True); entry = pool.get(sid, {})
        for n in need:
            img = AV.VARIANTS[n](c); assert img.shape[:2] == fin.shape[:2], f'{sid}/{n}: {img.shape[:2]} is not the final\'s {fin.shape[:2]}: all versions must be on the same crop'; f = os.path.join(d, f'{n}.jpg'); tmp = os.path.join(d, f'{n}.tmp.jpg'); cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q]); os.replace(tmp, f); entry[n] = dict(file=f'img/{sid}/{n}.jpg', hash=hashlib.sha1(open(f, 'rb').read()).hexdigest()[:16], key=keys[n])
        with lock: pool[sid] = entry; save_json(pf, pool); done = len(pool)
        say(f'  {sid}: {", ".join(need)} in {time.time() - t0:.0f} s ({done} pictures)')

    gw, cw = AS.auto_workers(gpu_slots=_gpu_slots()); gw = a.gpu_workers if a.gpu_workers is not None else gw; cw = a.cpu_workers if a.cpu_workers is not None else cw
    cvt = int(os.environ.get('STRATASTILLS_ARENA_CV2_THREADS') or 2)
    if not a.serial and cw + gw > 1 and cvt > 0: cv2.setNumThreads(cvt)                                                                                           # each worker's cv2 calls use two threads, not all the cores
    chunks = [todo[s:s + a.chunk] for s in range(0, len(todo), a.chunk)]; say(f'{len(chunks)} chunks of up to {a.chunk}; ' + ('serial' if a.serial else f'{gw} GPU + {cw} CPU workers'))
    def failed(kind, what, e): say(f'  {kind} FAILED ({type(e).__name__}: {e}): ' + (', '.join(sid for _, sid, _ in what) if kind == 'chunk' else what[0][1]))
    try: st = AS.run(chunks, produce, make, gpu_workers=gw, cpu_workers=cw, serial=a.serial, on_error=failed)
    finally: shutil.rmtree(work, ignore_errors=True)
    say(f'{st["items"]} pictures made in {st["seconds"]:.0f} s' + (f'; FAILED: {st["failed_chunks"]} chunks, {st["failed_items"]} pictures' if st['failed_chunks'] or st['failed_items'] else ''))
    say(f'pool: {len(pool)} pictures'); return 0


if __name__ == '__main__':
    from stratastills import gpu
    with gpu.lease('arena build'): code = main()                    # the whole build holds the GPU: another program on the card made it write grey, truncated pictures
    sys.exit(code)
