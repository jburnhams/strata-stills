"""Calibrate a device's `NoiseScale` (inputs.adjust_tags): the factor on its noise profile that makes the pipeline's clean-up leave as much luminance noise in the flat areas as the camera's own JPEG does. The noise profile of a file or of a table (darktable's fits) says how noisy the raw is; how hard to clean the picture is a taste, and the camera's JPEG is the reference a person compares with. For each scale in --scales the pictures of the candidate set are made with the pipeline (the app's own generate, throw-away project), the luminance noise in the flat areas (scripts/compare_detail.py's measure, masks from the camera JPEG) is taken over them, and the scale whose median noise is closest to the camera's (ours / camera = --target, default 1.0) is found by a straight line through the points in log-log, for each band of ISO of the pictures (--bands, the noise a profile has to be scaled by does depend on the ISO: darktable's table, for one, does not scale with it the way this pipeline needs). A single band gives a number, several a table [[ISO, scale]] (the geometric mean ISO of the pictures of each band), which inputs.noise_scale interpolates. Prints the tables; with --bundle stores `NoiseScale` in the profile.
  python scripts/fit_noise_scale.py candidates/a7c docs/a7c_candidates.json --ext .ARW [--scales 1 2 4 8 16 32] [--target 1.0] [--bands 0-250 251-800 801-2500 2501-6000 6001-100000] [--bundle src/stratastills/sony_ilce7c_profile.json]"""
import argparse, collections, glob, json, os, shutil, sys, tempfile
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import compare_detail as CD

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('candidates'); ap.add_argument('--ext', default='.ARW'); ap.add_argument('--scales', type=float, nargs='+', default=[1, 2, 4, 8, 16, 32]); ap.add_argument('--target', type=float, default=1.0); ap.add_argument('--bundle'); ap.add_argument('--size', type=int, default=3000); ap.add_argument('--bands', nargs='*', default=['0-100000']); a = ap.parse_args()
    tmp = tempfile.mkdtemp(); os.environ['STRATASTILLS_PROJECTS'] = os.path.join(tmp, 'projects'); os.environ['STRATASTILLS_HOME'] = os.path.join(tmp, 'home')
    from stratastills import projects as P, pipeline as PL, inputs
    pid = P.create('fit', [os.path.abspath(a.folder)])['id']; sid = P.source_id(P.load(pid)['sources'][0]['path']); items = {i['stem']: i for i in P.source_items(pid, sid)}; sel = [c['stem'] for c in json.load(open(a.candidates)) if c['stem'] in items]
    ref = {s: CD.load(next(iter(glob.glob(os.path.join(a.folder, s + '.[jJ][pP][gG]')))), a.size) for s in sel}; masks = {s: CD.masks(ref[s]) for s in sel}; cam = {s: CD.measure(ref[s], *masks[s])['noise'] for s in sel}
    name = None; res = {}; isos = {s: inputs._iso(os.path.join(a.folder, s + a.ext)) or 0 for s in sel}; bands = [tuple(int(x) for x in b.split('-')) for b in a.bands]; band_of = {s: next((i for i, (lo, hi) in enumerate(bands) if lo <= isos[s] <= hi), None) for s in sel}
    for idx, k in enumerate(a.scales):
        ratios = collections.defaultdict(list)
        for s in sel:
            it = items[s]
            if band_of[s] is None: continue
            if name is None:
                from stratastills import develop
                t = develop.read_dng_tags(it['variants'][0]['path']); name = inputs.profile(t.get('Make'), t.get('Model')).get('bundle')
            if name: inputs._bundle(name); inputs._cache[name] = dict(inputs._cache[name], NoiseScale=k)
            r = PL.generate(pid, it['id'], preset='best', options=dict(size=str(a.size + idx), denoise='auto'), progress=lambda m: None); im = CD.load(r['file'], a.size)
            h = min(im.shape[0], ref[s].shape[0]); w = min(im.shape[1], ref[s].shape[1]); fl = masks[s][0][:h, :w]; tx = masks[s][1][:h, :w]; ratios[band_of[s]].append(CD.measure(im[:h, :w], fl, tx)['noise'] / max(cam[s], 1e-3))
        for bi, v in ratios.items(): res.setdefault(bi, []).append((k, float(np.median(v)), len(v)))
        print('scale %5.1f: ' % k + '  '.join('ISO %d-%d: %.2f (%d)' % (*bands[bi], res[bi][-1][1], res[bi][-1][2]) for bi in sorted(res)) + '   [our flat-area noise / that of the camera JPEG, median (pictures)]', flush=True)
    table = []
    for bi in sorted(res):
        ks = np.array([r[0] for r in res[bi]]); rs = np.array([r[1] for r in res[bi]]); b, a0 = np.polyfit(np.log(ks), np.log(rs), 1)                                   # the ratio falls with the scale as a power law: a line through all the points in log-log (single pictures are noisy, the line is not)
        best = float(np.clip(np.exp((np.log(a.target) - a0) / b), ks.min() / 4, ks.max() * 4)) if b < -0.02 else float(ks[np.argmin(np.abs(rs - a.target))])
        iso = float(np.exp(np.mean(np.log([max(isos[s], 50) for s in sel if band_of[s] == bi])))); table.append([round(iso), round(best, 1)]); print('ISO %d-%d (geometric mean %d, %d pictures): scale %.1f for a ratio of %.2f' % (*bands[bi], iso, len(res[bi]) and res[bi][0][2], best, a.target))
    best = table if len(table) > 1 else table[0][1]
    if a.bundle and name:
        b = json.load(open(a.bundle)); b['NoiseScale'] = best; json.dump(b, open(a.bundle, 'w'), separators=(',', ':')); print('stored NoiseScale in', a.bundle)
    shutil.rmtree(tmp, ignore_errors=True)
