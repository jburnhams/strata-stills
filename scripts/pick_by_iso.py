"""Pick pictures at a spread of ISOs (and lenses, and shoots) from the table scripts/scan_exif.py wrote, for a candidate set that should cover the camera's noise range, not just its base ISO. Writes entries {stem, why, source, iso, lens, metrics: {}} in the format of docs/*_candidates.json (`source` is the folder the picture comes from: scripts/copy_candidates.py reads it); the numbers are filled in by `parity_jpeg.py --accept`.
  python scripts/pick_by_iso.py iso.jsonl --root "\\SERVER\Raw\2023" --out new.json [--exclude existing.json] [--bands 125-250:3 320-640:3 ...]"""
import argparse, collections, json, os, random

BANDS = '125-250:3 320-640:3 800-1250:4 1600-2500:4 3200-5000:5 6400-8000:5 10000-12800:5'

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('table'); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True); ap.add_argument('--exclude'); ap.add_argument('--bands', nargs='*', default=BANDS.split()); ap.add_argument('--seed', type=int, default=1); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.table)]; have = {c['stem'] for c in json.load(open(a.exclude))} if a.exclude else set(); rng = random.Random(a.seed); out = []; stems = collections.Counter(os.path.splitext(r['file'])[0] for r in rows)
    for b in a.bands:
        rng_s, _, n = b.partition(':'); lo, _, hi = rng_s.partition('-'); lo, hi, n = int(lo), int(hi), int(n)
        pool = [r for r in rows if lo <= r['iso'] <= hi and os.path.splitext(r['file'])[0] not in have and stems[os.path.splitext(r['file'])[0]] == 1]; rng.shuffle(pool); chosen = []; seen_lens = collections.Counter(); seen_folder = collections.Counter()
        while pool and len(chosen) < n:                                             # the picture whose lens and shoot are the least used so far
            pool.sort(key=lambda r: (seen_lens[r['lens']] + seen_folder[r['folder']])); r = pool.pop(0); chosen.append(r); seen_lens[r['lens']] += 1; seen_folder[r['folder']] += 1
        for r in chosen: out.append(dict(stem=os.path.splitext(r['file'])[0], why=f'ISO {lo}-{hi}: {r["iso"]}, {r["lens"]}', source=os.path.join(a.root, r['folder']), iso=r['iso'], lens=r['lens'], metrics={}))
    json.dump(out, open(a.out, 'w'), indent=1)
    for c in out: print(c['stem'], c['why'], os.path.basename(c['source']))
