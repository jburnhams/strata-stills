"""Pick a varied set of pictures to work on from the output of parity_jpeg.py (run with --n all): the worst few on each dimension (colour difference, lightness, too colourful, too dull, hue each way) and the best ones, as controls that must not get worse. Writes a JSON list ({stem, why, metrics}) and prints it.
Usage: python scripts/pick_candidates.py parity.txt --out candidates.json [--per 3] [--best 3]"""
import argparse, json, re

KEYS = ('dL', 'Lj', 'chroma_ratio', 'dE', 'dhue', 'dL_pix', 'dE_p90', 'dE_sat', 'dE_dark', 'dE_bright')


def parse(path):
    rows = {}
    for line in open(path, encoding='utf-8', errors='ignore'):
        m = re.match(r'(\S+)\.(?:ARW|DNG)\s+(.*)', line)
        if not m or 'no match' in line: continue
        v = m.group(2).split()
        rows[m.group(1)] = {k: float(x) for k, x in zip(v[0::2], v[1::2]) if k in KEYS}
    return rows


def pick(rows, per=3, best=3):
    """[(stem, why)]: each picture once, under the first dimension that chose it."""
    dims = [('colour difference (dE)', lambda r: -r['dE']), ('local colour difference (dE, 90th percentile)', lambda r: -r['dE_p90']), ('saturated colours (dE)', lambda r: -r.get('dE_sat', 0)), ('shadows (dE)', lambda r: -r.get('dE_dark', 0)), ('highlights (dE)', lambda r: -r.get('dE_bright', 0)), ('lightness too dark', lambda r: r['dL_pix']), ('lightness too bright', lambda r: -r['dL_pix']), ('too colourful', lambda r: -r['chroma_ratio']), ('too dull', lambda r: r['chroma_ratio']),
            ('hue shifted one way', lambda r: -r['dhue']), ('hue shifted the other way', lambda r: r['dhue'])]
    out, seen = [], set()
    for why, key in dims:
        n = 0
        for s in sorted(rows, key=lambda s: key(rows[s])):
            if s in seen: continue
            out.append((s, 'worst: ' + why)); seen.add(s); n += 1
            if n == per: break
    n = 0
    for s in sorted(rows, key=lambda s: rows[s]['dE']):
        if s in seen: continue
        out.append((s, 'best (control: must not get worse)')); seen.add(s); n += 1
        if n == best: break
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('parity'); ap.add_argument('--out', required=True); ap.add_argument('--per', type=int, default=3); ap.add_argument('--best', type=int, default=3); a = ap.parse_args()
    rows = parse(a.parity); sel = [dict(stem=s, why=w, metrics=rows[s]) for s, w in pick(rows, a.per, a.best)]
    json.dump(sel, open(a.out, 'w'), indent=1)
    for x in sel: print(x['stem'], x['why'].ljust(42), ' '.join(f'{k} {v:6.2f}' for k, v in x['metrics'].items()))
