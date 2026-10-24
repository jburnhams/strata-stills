"""Render the candidate raws with OUR pipeline (the same generate the app runs, in a throw-away project) into candidates/renders/<name>/<stem>.jpg, so that scripts/compare_detail.py and scripts/parity_external.py can set them beside darktable's and RawTherapee's.
  python scripts/render_ours.py candidates/a7c docs/a7c_candidates.json [--out candidates/renders/ours] [--size 3000] [--preset best] [--option key=value ...]
`--option` passes pipeline options (e.g. --option lens=off, --option sharpen=false)."""
import argparse, json, os, shutil, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('candidates'); ap.add_argument('--out', default=os.path.join('candidates', 'renders', 'ours')); ap.add_argument('--size', default='3000'); ap.add_argument('--preset', default='best'); ap.add_argument('--option', action='append', default=[]); ap.add_argument('--patch', action='append', default=[], help='for tuning: module.NAME=python-literal, e.g. detail.SHARPEN_MASK=(2,6)'); a = ap.parse_args()
    tmp = tempfile.mkdtemp(); os.environ['STRATASTILLS_PROJECTS'] = os.path.join(tmp, 'projects'); os.environ['STRATASTILLS_HOME'] = os.path.join(tmp, 'home')
    from stratastills import projects as P, pipeline as PL
    opts = dict(size=a.size)
    for o in a.option:
        k, _, v = o.partition('='); opts[k] = {'true': True, 'false': False}.get(v.lower(), v)
    import importlib
    for pt in a.patch: k, _, v = pt.partition('='); m, _, n = k.partition('.'); setattr(importlib.import_module('stratastills.' + m), n, eval(v))
    pid = P.create('render', [os.path.abspath(a.folder)])['id']; sid = P.source_id(P.load(pid)['sources'][0]['path']); items = {i['stem']: i for i in P.source_items(pid, sid)}; os.makedirs(a.out, exist_ok=True)
    for c in json.load(open(a.candidates)):
        it = items.get(c['stem'])
        if not it: print('missing', c['stem']); continue
        r = PL.generate(pid, it['id'], preset=a.preset, options=opts, progress=lambda m: None); shutil.copy(r['file'], os.path.join(a.out, c['stem'] + '.jpg')); print(c['stem'], flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
