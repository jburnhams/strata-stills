"""Add a camera to the device profiles (inputs.PROFILES) with the measurements that make its raws come out like its own JPEGs, in one repeatable run. Aimed at the best result for the device, not at being automatic: a new camera is rare, so each step is printed and can be rerun alone.
  python scripts/add_device.py <folder with raw+JPG pairs> <candidates.json> --make SONY --model NEX-5N --label 'Sony NEX-5N' [--ext .ARW] [--noise noiseprofiles.json] [--dcp 'SONY X.dcp'] [--steps bundle tone look noise]
Steps (the pictures are the raws of the candidate set with the camera's JPEG beside them; every 5th is held out of the look fit and judged on):
  bundle  src/stratastills/<make>_<model>_profile.json: the colour matrices (a .dcp from --dcp, else LibRaw's own for this camera) and darktable's per-ISO noise table (--noise: data/noiseprofiles.json of darktable; this script says when it has no entry), and the PROFILES line in inputs.py
  tone    scripts/fit_tone_curve.py: ProfileToneCurve fitted to the camera's JPEGs
  look    scripts/fit_look_table.py: the hue/saturation LookTable (develops every pair at 1/8 size, about 5 s each; the pairs are kept in <folder>/../look_pairs_<model>.npz)
  noise   scripts/fit_noise_scale.py: NoiseScale per ISO band so the clean-up leaves the flat-area noise the camera's JPEG has
Each later step uses the bundle the earlier ones made, so run them in this order; rerun a step after changing a candidate set."""
import argparse, glob, json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, '..', 'src', 'stratastills')
BANDS = ['0-250', '251-800', '801-2500', '2501-6000', '6001-100000']


def run(*cmd):
    print('$', ' '.join(cmd), flush=True); r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=os.path.join(HERE, '..', 'src')))
    if r.returncode: sys.exit(f'step failed ({r.returncode})')


def bundle_path(a): return os.path.join(SRC, re.sub(r'[^a-z0-9]+', '', a.make.lower()) + '_' + re.sub(r'[^a-z0-9]+', '', a.model.lower()) + '_profile.json')


def step_bundle(a, path):
    sys.path.insert(0, HERE); import build_profile_bundle as B
    if a.dcp: b = B.read_dcp(a.dcp)
    else:
        import rawpy
        raw = sorted(glob.glob(os.path.join(a.folder, '*' + a.ext)))[0]
        with rawpy.imread(raw) as r: m = r.rgb_xyz_matrix[:3].reshape(-1)
        sys.path.insert(0, os.path.join(SRC, '..')); from stratastills import inputs; r9 = list(inputs._rat(m)); b = dict(ColorMatrix1=r9, ColorMatrix2=r9, CalibrationIlluminant1=17, CalibrationIlluminant2=21, ProfileName='Embedded')          # one matrix serves as both illuminants, as for a file with no profile
        print('matrices from LibRaw:', raw)
    if a.noise: b['NoiseTable'] = B.noise_table(a.noise, a.model)
    else: print('no --noise given: the noise profile stays the generic one')
    json.dump(b, open(path, 'w'), separators=(',', ':')); print('wrote', path)
    code = os.path.join(SRC, 'inputs.py'); s = open(code).read(); key = f"('{a.make.lower()}', '{a.model.lower()}')"
    if key not in s:
        line = f"    {key}: dict(label={a.label!r}, bundle={os.path.basename(path)!r}),\n"; s = s.replace('\n}\n_cache', '\n' + line.rstrip('\n') + '\n}\n_cache', 1); open(code, 'w').write(s); print('added to inputs.PROFILES:', key)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('candidates'); ap.add_argument('--make', required=True); ap.add_argument('--model', required=True); ap.add_argument('--label'); ap.add_argument('--ext', default='.ARW')
    ap.add_argument('--noise'); ap.add_argument('--dcp'); ap.add_argument('--steps', nargs='*', default=['bundle', 'tone', 'look', 'noise']); ap.add_argument('--holdout', default='5'); a = ap.parse_args(); a.label = a.label or f'{a.make.title()} {a.model}'
    path = bundle_path(a); py = sys.executable; pairs = os.path.join(os.path.dirname(os.path.abspath(a.folder)), 'look_pairs_' + os.path.basename(path)[:-len('_profile.json')] + '.npz')
    if 'bundle' in a.steps: step_bundle(a, path)
    if 'tone' in a.steps: run(py, os.path.join(HERE, 'fit_tone_curve.py'), a.folder, '--bundle', path, '--ext', a.ext)
    if 'look' in a.steps:
        if not os.path.exists(pairs): run(py, os.path.join(HERE, 'fit_look_table.py'), 'build', a.folder, '--cache', pairs, '--ext', a.ext, '--lens', 'auto')
        run(py, os.path.join(HERE, 'fit_look_table.py'), 'fit', '--cache', pairs, '--bundle', path, '--holdout', a.holdout)
    if 'noise' in a.steps: run(py, os.path.join(HERE, 'fit_noise_scale.py'), a.folder, a.candidates, '--ext', a.ext, '--bundle', path, '--bands', *BANDS, '--per-band', '8', '--scales', '2', '4', '8', '16', '32')
