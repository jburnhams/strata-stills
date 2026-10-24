"""Calibrate a device's `NoiseScale` (inputs.adjust_tags): the factor on its noise profile that makes the pipeline's clean-up leave as much luminance noise in the flat areas as the camera's own JPEG does. The noise profile of a file or of a table (darktable's fits) says how noisy the raw is; how hard to clean the picture is a taste, and the camera's JPEG is the reference a person compares with. For each scale in --scales the pictures of the candidate set are made with the pipeline (the app's own generate, throw-away project), the luminance noise in the flat areas (scripts/compare_detail.py's measure, masks from the camera JPEG) is taken over them, and the scale whose median noise is closest to the camera's (ours / camera = --target, default 1.0) is found by interpolating the log of the ratio. Prints the table; with --bundle stores `NoiseScale` in the profile.
  python scripts/fit_noise_scale.py candidates/a7c docs/a7c_candidates.json --ext .ARW [--scales 1 2 4 8 16 32] [--target 1.0] [--bundle src/stratastills/sony_ilce7c_profile.json]"""
import argparse, glob, json, os, shutil, sys, tempfile
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import compare_detail as CD

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('candidates'); ap.add_argument('--ext', default='.ARW'); ap.add_argument('--scales', type=float, nargs='+', default=[1, 2, 4, 8, 16, 32]); ap.add_argument('--target', type=float, default=1.0); ap.add_argument('--bundle'); ap.add_argument('--size', type=int, default=3000); a = ap.parse_args()
    tmp = tempfile.mkdtemp(); os.environ['STRATASTILLS_PROJECTS'] = os.path.join(tmp, 'projects'); os.environ['STRATASTILLS_HOME'] = os.path.join(tmp, 'home')
    from stratastills import projects as P, pipeline as PL, inputs
    pid = P.create('fit', [os.path.abspath(a.folder)])['id']; sid = P.source_id(P.load(pid)['sources'][0]['path']); items = {i['stem']: i for i in P.source_items(pid, sid)}; sel = [c['stem'] for c in json.load(open(a.candidates)) if c['stem'] in items]
    ref = {s: CD.load(next(iter(glob.glob(os.path.join(a.folder, s + '.[jJ][pP][gG]')))), a.size) for s in sel}; masks = {s: CD.masks(ref[s]) for s in sel}; cam = {s: CD.measure(ref[s], *masks[s])['noise'] for s in sel}
    name = None; res = []
    for idx, k in enumerate(a.scales):
        ratios = []
        for s in sel:
            it = items[s]
            if name is None:
                from stratastills import develop
                t = develop.read_dng_tags(it['variants'][0]['path']); name = inputs.profile(t.get('Make'), t.get('Model')).get('bundle')
            if name: inputs._bundle(name); inputs._cache[name] = dict(inputs._cache[name], NoiseScale=k)
            r = PL.generate(pid, it['id'], preset='best', options=dict(size=str(a.size + idx), denoise='auto'), progress=lambda m: None); im = CD.load(r['file'], a.size)
            h = min(im.shape[0], ref[s].shape[0]); w = min(im.shape[1], ref[s].shape[1]); fl = masks[s][0][:h, :w]; tx = masks[s][1][:h, :w]; ratios.append(CD.measure(im[:h, :w], fl, tx)['noise'] / max(cam[s], 1e-3))
        res.append((k, float(np.median(ratios)))); print('scale %5.1f: our flat-area noise / the camera JPEG\'s = %.2f (median over %d pictures)' % (k, res[-1][1], len(sel)), flush=True)
    ks = np.array([r[0] for r in res]); rs = np.array([r[1] for r in res]); o = np.argsort(rs)
    best = float(np.exp(np.interp(np.log(a.target), np.log(rs[o]), np.log(ks[o])))) if rs.min() <= a.target <= rs.max() else float(ks[np.argmin(np.abs(rs - a.target))])
    print('NoiseScale for a ratio of %.2f: %.1f' % (a.target, best))
    if a.bundle and name:
        b = json.load(open(a.bundle)); b['NoiseScale'] = round(best, 1); json.dump(b, open(a.bundle, 'w'), separators=(',', ':')); print('stored NoiseScale in', a.bundle)
    shutil.rmtree(tmp, ignore_errors=True)
