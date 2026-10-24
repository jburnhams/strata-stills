"""Build a device profile bundle (src/stratastills/<name>_profile.json, see inputs.py) from profiles other projects already made, rather than measuring a camera ourselves:
  --dcp     a DNG camera profile (.dcp: colour matrices, hue/saturation maps), e.g. RawTherapee's rtdata/dcpprofiles/SONY ILCE-7C.dcp (its profiles are public domain, made with Adobe's DNG converter)
  --noise   darktable's data/noiseprofiles.json (per-ISO Poisson/Gaussian noise fits contributed by its users, GPL-3.0) with --model naming the entry
A bundle can also hold a `ProfileToneCurve` (scripts/fit_tone_curve.py adds one fitted to the camera's own JPEGs). Usage: python scripts/build_profile_bundle.py --dcp "SONY ILCE-7C.dcp" --noise noiseprofiles.json --model ILCE-7C --out src/stratastills/sony_ilce7c_profile.json"""
import argparse, json
import numpy as np
import tifffile

MATRICES = ('ColorMatrix1', 'ColorMatrix2', 'CalibrationIlluminant1', 'CalibrationIlluminant2')
HUE_SAT = ('ProfileHueSatMapDims', 'ProfileHueSatMapData1', 'ProfileHueSatMapData2', 'ProfileHueSatMapEncoding')


def read_dcp(path, hue_sat=False):
    """The matrices of a DCP and, with `hue_sat`, its hue/saturation maps. Adobe's maps are a look, not a calibration: on the A7C they took median chroma to 1.9x the camera JPEG's (matrices alone: 1.15x), the same over-saturation the DJI notes in docs/progress.md found, so they are left out unless asked for."""
    with tifffile.TiffFile(path, is_dng=False) as t:
        tags = {tag.name: tag.value for tag in t.pages[0].tags.values()}
    out = {}
    for k in MATRICES + (HUE_SAT if hue_sat else ()):
        if k in tags: v = tags[k]; out[k] = np.asarray(v).tolist() if hasattr(v, '__len__') else v
    out['ProfileName'] = 'Embedded'; return out


def noise_table(path, model):
    """[[iso, S, O]]: darktable's a and b (variance = a * v + b of the raw value 0..1, per channel) averaged over the channels, as the DNG NoiseProfile (S, O) the development uses."""
    for maker in json.load(open(path, encoding='utf-8'))['noiseprofiles']:
        for m in maker['models']:
            if m['model'].lower() == model.lower(): return [[p['iso'], float(np.mean(p['a'])), float(np.mean(p['b']))] for p in m['profiles']]
    raise SystemExit(f'no noise profile for {model}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dcp', required=True); ap.add_argument('--noise'); ap.add_argument('--model'); ap.add_argument('--out', required=True); ap.add_argument('--hue-sat', action='store_true'); a = ap.parse_args()
    b = read_dcp(a.dcp, a.hue_sat)
    if a.noise: b['NoiseTable'] = noise_table(a.noise, a.model)
    json.dump(b, open(a.out, 'w'), separators=(',', ':')); print('wrote', a.out, sorted(b))
