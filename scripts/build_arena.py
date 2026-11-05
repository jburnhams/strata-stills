"""Make the pictures of the rating arena (/arena/ on the server): for pictures from shoots the models were not trained on, every version in scripts/arena_variants.py (the person's Lightroom final, our neutral picture, auto1, auto2 ...), all on the final's framing, at DISPLAY px on the long side, into out/arena/img/<picture>/<version>.jpg, with out/arena/pool.json listing them (the server reads it as it grows, so voting can start before this finishes).
  python scripts/build_arena.py [--n 60] [--seed 7] [--redo auto3] [--retire NAME] [--unretire NAME]
Pictures are taken round-robin over the held-out shoots in a fixed random order, so the pool covers the whole range and a later run continues where this one stopped. Resumable and incremental: a version is made again only when its code (arena_variants.py, lr_look.py ...) changed or with --redo; a new version in VARIANTS is made for every picture already in the pool. Needs the pictures' neutral develops (cached in out/cache/ by build_lr_report.py, made here when missing: 20-40 s each).
PRIVATE: your own pictures, never public."""
import argparse, hashlib, json, os, sys, tempfile, time, shutil, types
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import arena_variants as AV, build_lr_report as LRR, lr_dataset as D, lr_look as LK, lr_tmodel as TM, lr_ftarget as FT
from stratastills import settings

DISPLAY = 1400
JPEG_Q = 88
DEPS = {'lightroom': [], 'camera': [], 'neutral': []}                                   # the files a version depends on besides its own function: the learned looks depend on the models' code
LOOK = ['lr_look.py', 'lr_tmodel.py', 'lr_ftarget.py', 'lr_fit.py']


def code_key(name):
    """A version is made again when its own function, the brightness helper or (for the automatic ones) the look code changed, not when another version did."""
    import inspect
    h = hashlib.sha1(name.encode()); h.update(inspect.getsource(AV.VARIANTS[name]).encode())
    if name in ('camera', 'neutral'): h.update(inspect.getsource(AV.match_brightness).encode())
    if name in ('camera', 'neutral') or name.startswith('auto'): h.update(inspect.getsource(AV.auto1).encode())
    for f in DEPS.get(name, LOOK): h.update(open(os.path.join(os.path.dirname(__file__), f), 'rb').read())
    return h.hexdigest()[:12]


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


def save_json(path, obj):
    tmp = path + '.tmp'; json.dump(obj, open(tmp, 'w'), indent=1); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--n', type=int, default=60, help='how many pictures to have in the pool at the end'); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--redo', action='append', default=[]); ap.add_argument('--retire', action='append', default=[]); ap.add_argument('--unretire', action='append', default=[]); ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args(); root = settings.arena_dir(); os.makedirs(os.path.join(root, 'img'), exist_ok=True); say = lambda m: print(m, flush=True)
    vf = os.path.join(root, 'variants.json'); var = json.load(open(vf)) if os.path.exists(vf) else {'retired': []}
    if a.retire or a.unretire: var['retired'] = sorted((set(var['retired']) | set(a.retire)) - set(a.unretire)); save_json(vf, var); say(f'retired: {var["retired"]}'); return 0
    pf = os.path.join(root, 'pool.json'); pool = json.load(open(pf)) if os.path.exists(pf) else {}
    rows, X, names, Y, idx, G = LK.load(); test = np.array([LK.test_shoot(g) for g in G]); FACES = json.load(open(os.path.join(D.OUT, 'faces.json'))) if os.path.exists(os.path.join(D.OUT, 'faces.json')) else {}
    models, tmodels, fmodels = LK.train_cached(X[~test], Y[~test]), TM.train_cached(), FT.train_cached(); keys = {n: code_key(n) for n in AV.VARIANTS}
    todo = []; valid = {sid for _, sid in order(rows, idx, G, test, a.seed)}
    gone = [k for k in pool if k not in valid]
    if gone: pool = {k: v for k, v in pool.items() if k in valid}; save_json(pf, pool); say(f'dropped {len(gone)} pictures from the pool: no longer in the dataset (the nudity filter flags them): {gone}')           # votes on them stay in votes.jsonl but the server no longer offers them
    for i, sid in order(rows, idx, G, test, a.seed):
        have = pool.get(sid, {}); need = [n for n in AV.VARIANTS if n in a.redo or n not in have or have[n].get('key') != keys[n]]
        if sid in pool or len(pool) + len([t for t in todo if t[1] not in pool]) < a.n:
            if need: todo.append((i, sid, need))
    say(f'{len(pool)} pictures in the pool; {len(todo)} to make or update')
    work = tempfile.mkdtemp(prefix='arena-', dir=os.path.join(root, '..'))
    try:
        for s in range(0, len(todo), 6):
            chunk = todo[s:s + 6]; raws = [rows[idx[i]]['raw'] for i, _, _ in chunk]; dev = LRR.neutral(raws, work, say, a.rebuild); ex = D.exif_of(raws)
            for i, sid, need in chunk:
                row = rows[idx[i]]; raw = row['raw']; fin = cv2.imread(row['final']); t0 = time.time()
                if fin is None or raw not in dev: say(f'  {sid}: skipped (missing picture)'); continue
                k = DISPLAY / max(fin.shape[:2]); fin = cv2.resize(fin, (round(fin.shape[1] * k), round(fin.shape[0] * k)), interpolation=cv2.INTER_AREA)
                neu, _ = LRR.to_final(D.unorient(dev[raw][0], ex.get(raw, {}).get('Orientation', 1)), row, ex.get(raw, {}).get('Orientation', 1), fin); fl = FACES.get(str(int(idx[i])), [])
                camera = D.camera_picture(raw); cam, _ = LRR.to_final(camera, row, ex.get(raw, {}).get('Orientation', 1), fin) if camera is not None else (neu, '')
                c = types.SimpleNamespace(fin=fin, neu=neu, cam=cam, pred=LK.predict(models, X[i:i + 1])[0], x=X[i], names=names, colour=TM.predict(tmodels, X[i:i + 1])[0], face=dict(box=fl[0]['box'], target=FT.face_target(fmodels, X[i], fl[0])) if fl else None)
                d = os.path.join(root, 'img', sid); os.makedirs(d, exist_ok=True); entry = pool.get(sid, {})
                for n in need:
                    img = AV.VARIANTS[n](c); f = os.path.join(d, f'{n}.jpg'); cv2.imwrite(f, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q]); entry[n] = dict(file=f'img/{sid}/{n}.jpg', hash=hashlib.sha1(open(f, 'rb').read()).hexdigest()[:16], key=keys[n])
                pool[sid] = entry; save_json(pf, pool); say(f'  {sid}: {", ".join(need)} in {time.time() - t0:.0f} s ({len(pool)} pictures)')
    finally: shutil.rmtree(work, ignore_errors=True)
    say(f'pool: {len(pool)} pictures'); return 0


if __name__ == '__main__':
    sys.exit(main())
