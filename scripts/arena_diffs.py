"""How far apart each pair of versions of an arena picture looks, for the arena's weighted choice (arena.pick_pair `diffs`): writes `<arena dir>/diffs.json`, {pair_key(hash_a, hash_b): [mean CIELAB distance, mean distance on the main real face or null]}. Keyed by the versions' hashes, so a rebuilt version is measured again and an unchanged one never. Pictures at 1/4 size (the distance is a mean over many pixels). The face box is the main real face of candidates/personal/faces2.pkl (fractions of the final's frame; the pool's pictures are cut a little at the edges, so the box is a few pixels off: padded 30 % to cover it). Read-only on the pool and the database; run again after build_arena.py: `.venv/bin/python scripts/arena_diffs.py`."""
import json, os, sys, itertools

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lr_dataset as D, lr_face_first as FF
from stratastills import arena as AR, arena_store, settings


def sid_of(row): return f'{os.path.basename(os.path.dirname(row["final"]))[:10]}_{os.path.splitext(os.path.basename(row["raw"]))[0]}'.replace(' ', '')


def lab(path):
    im = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    return None if im is None else cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


def main():
    root = settings.arena_dir(); pool = arena_store.get().pool(); retired = set(arena_store.get().retired())
    f = os.path.join(root, 'diffs.json'); diffs = json.load(open(f)) if os.path.exists(f) else {}
    rows = D.load_catalogue(); row_of = {sid_of(r): i for i, r in enumerate(rows)}; todo = done = 0
    for n, (pic, vs) in enumerate(pool.items()):
        names = [v for v in vs if v not in retired]
        pairs = [(a, b) for a, b in itertools.combinations(names, 2) if vs[a]['hash'] != vs[b]['hash'] and AR.pair_key(vs[a]['hash'], vs[b]['hash']) not in diffs]
        if not pairs: continue
        need = {v for p in pairs for v in p}; img = {v: lab(os.path.join(root, vs[v]['file'])) for v in need}; face = FF.main_face(row_of[pic]) if pic in row_of else None
        for a, b in pairs:
            todo += 1
            if img[a] is None or img[b] is None or img[a].shape != img[b].shape: continue
            d = np.sqrt(((img[a] - img[b]) ** 2).sum(2)); H, W = d.shape; fd = None
            if face is not None:
                x, y, w, h = face['box']; x0, x1 = int(max(0, (x - .3 * w) * W)), int(min(W, (x + 1.3 * w) * W)); y0, y1 = int(max(0, (y - .3 * h) * H)), int(min(H, (y + 1.3 * h) * H))
                if x1 > x0 and y1 > y0: fd = round(float(d[y0:y1, x0:x1].mean()), 2)
            diffs[AR.pair_key(vs[a]['hash'], vs[b]['hash'])] = [round(float(d.mean()), 2), fd]; done += 1
        if n % 50 == 49: print(f'{n + 1}/{len(pool)} pictures, {done} pairs', flush=True)
    tmp = f + '.tmp'; json.dump(diffs, open(tmp, 'w')); os.replace(tmp, f); print(f'{done} of {todo} new pairs measured; {len(diffs)} in {f}')


if __name__ == '__main__': main()
