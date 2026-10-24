"""Copy the pictures of a candidate set (docs/a7c_candidates.json, scripts/pick_candidates.py) out of the shoot folder into a local folder of the repo that git ignores (`candidates/`), so the set can be worked on without the file server and grows as more are added. Originals are only read. Every file that belongs to a stem (raw, JPG, ...) is copied (from the entry's own `source` folder when it has one, else <shoot folder>); what is already there is skipped.
  python scripts/copy_candidates.py docs/a7c_candidates.json <shoot folder> [--to candidates/a7c] [--add DSC00123 DSC00130 ...]
`--add` appends stems to the JSON as well (with why 'added by hand'; their metrics are filled in by the next `parity_jpeg.py --candidates` accepted run)."""
import argparse, glob, json, os, shutil

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('candidates'); ap.add_argument('source'); ap.add_argument('--to', default=os.path.join('candidates', 'a7c')); ap.add_argument('--add', nargs='*', default=[]); a = ap.parse_args()
    sel = json.load(open(a.candidates)); have = {c['stem'] for c in sel}
    for s in a.add:
        if s not in have: sel.append(dict(stem=s, why='added by hand', metrics={}))
    if a.add: json.dump(sel, open(a.candidates, 'w'), indent=1)
    os.makedirs(a.to, exist_ok=True); n = size = 0
    for c in sel:
        for src in glob.glob(os.path.join(c.get('source', a.source), c['stem'] + '.*')):
            dst = os.path.join(a.to, os.path.basename(src))
            if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src): continue
            shutil.copy2(src, dst); n += 1; size += os.path.getsize(dst)
    print(f'copied {n} files ({size / 1e6:.0f} MB) into {a.to}; {len(sel)} candidates')
