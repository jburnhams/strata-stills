"""Tune the sharpening without redoing the development: the pipeline up to the sharpen stage is run once per candidate and kept (16 bit, with the measured MTF), then any number of sharpening variants are applied to the kept pictures and written as JPEGs for scripts/compare_detail.py.
  python scripts/sharpen_sweep.py cache candidates/a7c docs/a7c_candidates.json --cache candidates/sharpen-cache/a7c [--size 3000]
  python scripts/sharpen_sweep.py apply --cache candidates/sharpen-cache/a7c --out candidates/renders/sweep-x [--amount 1.5] [--patch detail.OVERSHOOT=0.03 ...]
`--patch module.NAME=value` sets a constant of stratastills.<module> for this run (the same way as render_ours.py); `apply` runs the CPU sharpening (the reference) unless STRATASTILLS_GPU is left on."""
import argparse, importlib, json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np


def cache(a):
    tmp = tempfile.mkdtemp(); os.environ['STRATASTILLS_PROJECTS'] = os.path.join(tmp, 'projects'); os.environ['STRATASTILLS_HOME'] = os.path.join(tmp, 'home')
    from stratastills import projects as P, pipeline as PL
    pid = P.create('cache', [os.path.abspath(a.folder)])['id']; sid = P.source_id(P.load(pid)['sources'][0]['path']); items = {i['stem']: i for i in P.source_items(pid, sid)}; os.makedirs(a.cache, exist_ok=True)
    for c in json.load(open(a.candidates)):
        out = os.path.join(a.cache, c['stem'])
        if os.path.exists(out + '.json') and not a.force: continue
        it = items.get(c['stem'])
        if not it: print('missing', c['stem']); continue
        v = PL.source_variant(it, None); st = PL.State(pid, it['id'], it, v, lambda m: None); stages = PL.resolve('best', dict(size=a.size), None)
        for s in stages:
            if s['stage'] in ('encode', 'sharpen'): continue
            p = {k: x for k, x in s.items() if k != 'stage'}; dict(develop=PL.run_develop, level=PL.run_level, resize=PL.run_resize)[s['stage']](st, p)
        res = st.res; np.save(out + '.npy', (np.clip(st.rgb, 0, 1) * 65535 + 0.5).astype(np.uint16)); json.dump(dict(ok=bool(res.get('ok')), freq=res.get('freq'), mtf=res.get('mtf'), scale=st.scale), open(out + '.json', 'w')); print(c['stem'], flush=True)


def apply(a):
    from stratastills import detail, export as X, pipeline as PL
    for pt in a.patch: k, _, v = pt.partition('='); m, _, n = k.partition('.'); setattr(importlib.import_module('stratastills.' + m), n, eval(v))
    os.makedirs(a.out, exist_ok=True); amount = a.amount if a.amount is not None else PL.SHARPEN_AMOUNT; a.quality = a.quality or PL.DEFAULT_QUALITY
    for f in sorted(os.listdir(a.cache)):
        if not f.endswith('.json'): continue
        stem = f[:-5]; m = json.load(open(os.path.join(a.cache, f))); rgb = np.load(os.path.join(a.cache, stem + '.npy')).astype(np.float32) / 65535
        if m['ok']: rgb = detail.sharpen_picture(rgb, np.array(m['freq']), np.array(m['mtf']), m['scale'], amount=amount, backend='cpu')
        X.write_picture(os.path.join(a.out, stem + '.jpg'), rgb, 'jpeg', a.quality, None); print(stem, flush=True)           # the encoder and quality of the product: the JPEG quantisation alone takes the flat-area noise measured by compare_detail from 0.76 to 0.42 at q90 (q98 gave the wrong picture of noise)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cache'); c.add_argument('folder'); c.add_argument('candidates'); c.add_argument('--cache', required=True); c.add_argument('--size', default='3000'); c.add_argument('--force', action='store_true')
    s = sub.add_parser('apply'); s.add_argument('--cache', required=True); s.add_argument('--out', required=True); s.add_argument('--amount', type=float); s.add_argument('--quality', type=int, default=None, help='JPEG quality (default: the pipeline\'s DEFAULT_QUALITY)'); s.add_argument('--patch', action='append', default=[])
    a = ap.parse_args(); cache(a) if a.cmd == 'cache' else apply(a)
