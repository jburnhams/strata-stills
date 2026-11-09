"""Our own neutral picture of every catalogue raw at thumbnail size, beside the camera-preview/final pairs of scripts/lr_dataset.py thumbs, so the person's edit can be learned FROM the picture the automatic look is applied to (our neutral), not from the camera's rendering.
  python scripts/lr_neutral_thumbs.py make [--limit N]     -> candidates/personal/nthumbs/t*.npz: `neu` (THUMB x THUMB uint8 BGR, on the final's crop and laid on it like the arena's versions), `fin` (the final, the same size), `idx` (catalogue rows), `how` (how each was aligned). Batches match thumbs/t*.npz; resumable.
  python scripts/lr_neutral_thumbs.py check [--n 40]       how far the fast develop is from the full one the reports and the arena use (the cached `best` neutrals in out/cache/report-auto-neutral), at thumbnail size
The fast develop is develop.develop(half=True) with the 'best' preset's tone (matched to the camera JPEG), lens correction and recover, but no noise clean-up and no sharpening: at 128 px neither shows, and the half-size decode is several times faster.
PRIVATE: the person's own pictures."""
import argparse, glob, json, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import lr_dataset as D, build_lr_report as LRR

OUT = os.path.join(D.OUT, 'nthumbs')


def fast_neutral(raw):
    """Our neutral picture of `raw` from the half-size decode (BGR uint8, the right way up)."""
    from stratastills import develop
    rgb = develop.develop(raw, half=True, ev='auto', tone='match', denoise='off', lens='auto', recover='auto')                      # the 'best' preset's develop (pipeline.run_develop): recover 'auto' with the matched tone, not develop()'s fixed default
    return np.clip(rgb[..., ::-1] * 255 + 0.5, 0, 255).astype(np.uint8)


def to_thumb(neu, row, orientation, fin_small):
    """`neu` (the right way up) cut to the final's crop and laid on `fin_small` (a reduced copy of the final), both squeezed to THUMB x THUMB."""
    al, how = LRR.to_final(D.unorient(neu, orientation), row, orientation, fin_small); sq = lambda x: cv2.resize(x, (D.THUMB, D.THUMB), interpolation=cv2.INTER_AREA)
    return sq(al), sq(fin_small), how


def make(a):
    rows = D.load_catalogue(); os.makedirs(OUT, exist_ok=True); t0 = time.time(); done = 0
    for k in range(0, len(rows), D.BATCH):
        dst = os.path.join(OUT, f't{k // D.BATCH:03d}.npz')
        if os.path.exists(dst): continue
        batch = rows[k:k + D.BATCH]; ex = D.exif_of([r['raw'] for r in batch]); N, F, keep, how = [], [], [], []
        for i, r in enumerate(batch):
            if a.limit and done >= a.limit: break
            try:
                fin = cv2.imread(r['final'], cv2.IMREAD_REDUCED_COLOR_4)
                if fin is None or r.get('error'): continue
                o = ex.get(r['raw'], {}).get('Orientation', 1); n, f, h = to_thumb(fast_neutral(r['raw']), r, o, fin)
                N.append(n); F.append(f); keep.append(k + i); how.append(h); done += 1
            except Exception as e: print(f'  {os.path.basename(r["raw"])}: FAILED {type(e).__name__}: {str(e)[:80]}', flush=True)
        if a.limit and done >= a.limit and len(keep) < len(batch): print(f'stopped at --limit {a.limit} (batch {k // D.BATCH} not saved)'); return
        tmp = dst + '.tmp.npz'; np.savez(tmp, neu=np.array(N, np.uint8), fin=np.array(F, np.uint8), idx=np.array(keep), how=np.array(how)); os.replace(tmp, dst)
        print(f'batch {k // D.BATCH}: {len(keep)}/{len(batch)} in {time.time() - t0:.0f} s', flush=True)


def check(a):
    """The fast develop against the cached full one, at thumbnail size: mean |dL| and colour distance (a, b) in CIELAB."""
    import hashlib, build_report as BR
    rows = D.load_catalogue(); cdir = os.path.join(BR.ROOT, 'out', 'cache', 'report-auto-neutral')
    key = lambda r: hashlib.sha1(json.dumps([BR.CACHE_VERSION, BR.code_hash(), r, os.path.getsize(r), int(os.path.getmtime(r))]).encode()).hexdigest()
    by_raw = {r['raw']: r for r in rows}; got = []; lab = lambda x: cv2.cvtColor(x.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    for raw, r in by_raw.items():
        if len(got) >= a.n: break
        if not os.path.exists(raw): continue
        base = os.path.join(cdir, key(raw))
        if not os.path.exists(base + '.jpg'): continue
        full = cv2.imread(base + '.jpg'); fin = cv2.imread(r['final'], cv2.IMREAD_REDUCED_COLOR_4)
        if full is None or fin is None or full.std() < 2: continue
        o = D.exif_of([raw]).get(raw, {}).get('Orientation', 1); t = time.time(); fast = fast_neutral(raw); dt = time.time() - t
        A, _, _ = to_thumb(full, r, o, fin); B, _, _ = to_thumb(fast, r, o, fin); d = lab(B) - lab(A)
        got.append((float(np.abs(d[..., 0]).mean()), float(np.hypot(d[..., 1], d[..., 2]).mean()), dt)); print(f'{os.path.basename(raw)}: |dL| {got[-1][0]:.2f}, colour {got[-1][1]:.2f}, {dt:.1f} s', flush=True)
    g = np.array(got); print(f'{len(g)} raws: mean |dL| {g[:, 0].mean():.2f} (max {g[:, 0].max():.2f}), colour {g[:, 1].mean():.2f} (max {g[:, 1].max():.2f}), {np.median(g[:, 2]):.1f} s each (median)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('step', choices=['make', 'check']); ap.add_argument('--limit', type=int); ap.add_argument('--n', type=int, default=40)
    a = ap.parse_args(); {'make': make, 'check': check}[a.step](a)
