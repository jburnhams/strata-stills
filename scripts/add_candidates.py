"""Add raws from anywhere in the archive to a candidate set: copies each raw into candidates/<set>/ as <shoot folder>_<name> (stems repeat from shoot to shoot, so the shoot is part of the new name), with the camera's JPEG next to it (the one beside the raw, else the full-size JPEG inside the raw, written as <name>.JPG), and appends it to docs/<set>_candidates.json (with `source`, the folder it came from, and `original`, its name there). Originals are only read.
  python scripts/add_candidates.py <set> <raw file> [<raw file> ...] [--why 'text'] [--to candidates/<set>]
  python scripts/add_candidates.py <set> --list picks.json        (a JSON list of {file, why?})
New sets are made as needed (docs/<set>_candidates.json; the set name is yours: a7c, nex5n, eos60d ...). scripts/build_report.py shows every set that has one."""
import argparse, glob, json, os, re, shutil, subprocess, sys


def preview_jpeg(path, out):
    """Write the largest JPEG inside the raw to `out`; False if there is none worth having."""
    best = b''
    for tag in ('JpgFromRaw', 'PreviewImage'):
        b = subprocess.run(['exiftool', '-b', '-' + tag, path], capture_output=True).stdout
        if len(b) > len(best): best = b
    if len(best) < 20000: return False
    open(out, 'wb').write(best); return True


def add(set_name, files, why, to=None, root=None):
    root = root or os.path.join(os.path.dirname(__file__), '..'); to = to or os.path.join(root, 'candidates', set_name); js = os.path.join(root, 'docs', f'{set_name}_candidates.json')
    sel = json.load(open(js)) if os.path.exists(js) else []; have = {c['stem'] for c in sel}; os.makedirs(to, exist_ok=True); n = 0
    for item in files:
        f = item['file'] if isinstance(item, dict) else item; w = (item.get('why') if isinstance(item, dict) else None) or why; d, base = os.path.split(os.path.abspath(f)); name, ext = os.path.splitext(base)
        stem = re.sub(r'[^A-Za-z0-9._-]+', '_', os.path.basename(d)) + '_' + name; dst = os.path.join(to, stem + ext)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(f): shutil.copy2(f, dst); n += 1
        jpg = next(iter(glob.glob(os.path.join(d, name + '.[jJ][pP][gG]'))), None); out = os.path.join(to, stem + '.JPG')
        if not os.path.exists(out):
            if jpg: shutil.copy2(jpg, out)
            else: preview_jpeg(f, out)
        if stem not in have: sel.append(dict(stem=stem, why=w, source=d, original=name, metrics={})); have.add(stem)
    json.dump(sel, open(js, 'w'), indent=1); print(f'{set_name}: {n} raws copied into {to}; {len(sel)} candidates')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('set'); ap.add_argument('files', nargs='*'); ap.add_argument('--list'); ap.add_argument('--why', default='added by hand'); ap.add_argument('--to'); a = ap.parse_args()
    files = list(a.files) + (json.load(open(a.list)) if a.list else [])
    if not files: sys.exit('no files given')
    add(a.set, files, a.why, a.to)
