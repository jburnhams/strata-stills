"""Lens profiles from the lensfun database (https://lensfun.github.io, data/db/*.xml: CC-BY-SA 3.0, so it is a separate download that is not part of this repository: `scripts/fetch_lensfun.py` puts it in models/lensfun/, or point STRATASTILLS_LENSFUN_DIR at a copy), for raws whose own file carries no correction data (lenscorr.py reads what the camera wrote; this is the fall-back for lenses it does not describe, e.g. the Sony E 50mm F1.8 OSS on an A7C).
The lens is found from the EXIF lens model: every word and number of a lensfun model name must be in it (after 'f/1.8' and 'F1.8' are made alike). The shot's focal length, aperture and focus distance pick the point of the calibration: distortion and colour fringes by the focal length, shading by the focal length, the aperture (in stops) and the distance (in 1/d), linearly between the calibrated points. Conventions follow lensfun's documentation: distortion and fringe radii are 1 at half the shorter side of the calibration frame, shading radii 1 at its corner; models: distortion poly3, poly5 and ptlens, fringes linear, shading pa (the picture is darker by 1 + k1 r^2 + k2 r^4 + k3 r^6, so the gain is its inverse). The correction is handed to lenscorr.py as the same two opcodes (gain map, warp) as the camera's own tables.
`from_exif(path, width, height)` gives the opcode list (bytes) or None."""
import glob, os, re, xml.etree.ElementTree as ET
import numpy as np

ENV = 'STRATASTILLS_LENSFUN_DIR'
_db = {}


def db_dir():
    d = os.environ.get(ENV)
    if d: return d
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'lensfun'); return p if os.path.isdir(p) else None


def tokens(name):
    s = re.sub(r'\bf\s*/\s*', 'f', str(name).lower()); return [t for t in re.split(r'[^a-z0-9.\-]+', s) if t]


def _f(e, k, default=0.0):
    v = e.get(k); return float(v) if v is not None else default


def _lens(e):
    model = [m.text for m in e.findall('model') if m.get('lang') in (None, 'en')]
    if not model: return None
    ar = (e.findtext('aspect-ratio') or '3:2').strip(); a, _, b = ar.partition(':'); aspect = float(a) / float(b) if b else float(a)
    cal = e.find('calibration'); out = dict(model=model[0], maker=(e.findtext('maker') or ''), mounts=[m.text for m in e.findall('mount')], crop=float(e.findtext('cropfactor') or 1.0), aspect=aspect, distortion=[], tca=[], vignetting=[])
    if cal is not None:
        for d in cal.findall('distortion'): out['distortion'].append(dict(model=d.get('model'), focal=_f(d, 'focal'), a=_f(d, 'a'), b=_f(d, 'b'), c=_f(d, 'c'), k1=_f(d, 'k1'), k2=_f(d, 'k2')))
        for t in cal.findall('tca'): out['tca'].append(dict(model=t.get('model'), focal=_f(t, 'focal'), kr=_f(t, 'kr', 1.0), kb=_f(t, 'kb', 1.0)))
        for v in cal.findall('vignetting'): out['vignetting'].append(dict(focal=_f(v, 'focal'), aperture=_f(v, 'aperture'), distance=_f(v, 'distance', 1000.0), k=[_f(v, 'k1'), _f(v, 'k2'), _f(v, 'k3')]))
    return out


def load(directory=None):
    """[lens dict] of every lens in the database files of `directory` (default db_dir()); [] when there is none."""
    directory = directory or db_dir()
    if not directory: return []
    if directory not in _db:
        lenses = []
        for f in sorted(glob.glob(os.path.join(directory, '*.xml'))):
            try: lenses += [x for x in (_lens(e) for e in ET.parse(f).getroot().iter('lens')) if x]
            except ET.ParseError: pass
        _db[directory] = lenses
    return _db[directory]


def find(lens_model, lenses=None):
    """The lens of the database that a camera's lens model string names: every token of the database's name is in the camera's; the one with the most tokens wins (None when none fits)."""
    want = set(tokens(lens_model)); best = None
    for ln in (load() if lenses is None else lenses):
        t = tokens(ln['model'])
        if t and set(t) <= want and (best is None or len(t) > best[0]): best = (len(t), ln)
    return best[1] if best else None


def _bracket(xs, x):
    xs = sorted(set(xs)); x = min(max(x, xs[0]), xs[-1])
    for lo, hi in zip(xs[:-1], xs[1:]):
        if lo <= x <= hi: return lo, hi, (x - lo) / (hi - lo)
    return xs[0], xs[0], 0.0


def _by_focal(points, focal, fn):
    if not points: return None
    lo, hi, t = _bracket([p['focal'] for p in points], focal); a = fn(next(p for p in points if p['focal'] == lo)); b = fn(next(p for p in points if p['focal'] == hi)); return a + (b - a) * t


def distortion_ratio(lens, focal, r):
    """rd / ru at the lensfun radius `r` (an array) for `focal`; 1 where the lens has no distortion calibration."""
    def one(d):
        if d['model'] == 'poly3': return d['k1'] * r * r + 1 - d['k1']
        if d['model'] == 'poly5': return 1 + d['k1'] * r * r + d['k2'] * r ** 4
        if d['model'] == 'ptlens': return d['a'] * r ** 3 + d['b'] * r * r + d['c'] * r + 1 - d['a'] - d['b'] - d['c']
        return np.ones_like(r)
    out = _by_focal(lens['distortion'], focal, one); return np.ones_like(r) if out is None else out


def tca_ratio(lens, focal, channel):
    """The scale of the radius of the red (0) or the blue (2) plane against green; only the 'linear' model is read."""
    pts = [t for t in lens['tca'] if t['model'] == 'linear']; out = _by_focal(pts, focal, lambda t: t['kr'] if channel == 0 else t['kb']); return 1.0 if out is None else out


def vignetting_attenuation(lens, focal, aperture, distance, r):
    """How much darker the picture is at the radius `r` (1 at the corner; an array): 1 + k1 r^2 + k2 r^4 + k3 r^6 at the point of the calibration the shot is at (focal length linear, aperture in stops, distance in 1/d between the calibrated ones; the widest aperture and the farthest distance when they are not known)."""
    pts = lens['vignetting']
    if not pts: return np.ones_like(r)

    def at(f):
        p = [v for v in pts if v['focal'] == f]; inv = lambda d: 1.0 / max(d, 0.01); want = inv(distance) if distance else 0.0
        near = min({v['distance'] for v in p}, key=lambda d: abs(inv(d) - want)); q = sorted([v for v in p if v['distance'] == near], key=lambda v: v['aperture']); stops = [np.log2(v['aperture']) for v in q]
        lo, hi, t = _bracket(stops, np.log2(aperture) if aperture else stops[0]); a = q[stops.index(lo)]['k']; b = q[stops.index(hi)]['k']; k = [x + (y - x) * t for x, y in zip(a, b)]
        return 1 + k[0] * r ** 2 + k[1] * r ** 4 + k[2] * r ** 6
    lo, hi, t = _bracket([v['focal'] for v in pts], focal); a = at(lo); b = at(hi); return a + (b - a) * t


def corrections(lens, width, height, focal, aperture=None, distance=None, camera_crop=None):
    """(gain_fn, warp_coefficients) for a shot: `gain_fn(r)` the shading gain at the radius r (1 = the half diagonal of the picture), and the [3 planes, 6] WarpRectilinear coefficients (lenscorr's form), or None for what the lens has no data for."""
    cam = camera_crop or lens['crop']; to_vig = lens['crop'] / cam; to_dist = to_vig * np.sqrt(1 + lens['aspect'] ** 2)         # picture radius (1 = half diagonal) -> lensfun's
    gain = (lambda r: 1.0 / vignetting_attenuation(lens, focal, aperture, distance, np.minimum(r * to_vig, 1.0))) if lens['vignetting'] else None             # beyond the calibrated corner (a lens for a smaller sensor on a bigger one) the value of the corner is held: no extrapolation
    warp = None
    if lens['distortion'] or lens['tca']:
        rr = np.linspace(0, 1.05, 200); d = distortion_ratio(lens, focal, np.minimum(rr * to_dist, np.sqrt(1 + lens['aspect'] ** 2))); warp = np.zeros((3, 6))
        for p, f in enumerate((d * tca_ratio(lens, focal, 0), d, d * tca_ratio(lens, focal, 2))): warp[p, :4] = np.linalg.lstsq(np.vander(rr * rr, 4, increasing=True), f, rcond=None)[0]
    return gain, warp


def shot_of(path):
    """(lens model, focal length mm, aperture, subject distance m or None) from the EXIF of a raw, or None."""
    import tifffile
    try:
        with tifffile.TiffFile(path) as t: ex = t.pages[0].tags['ExifTag'].value
        r = lambda k: (ex[k][0] / ex[k][1]) if k in ex and ex[k][1] else None; d = r('SubjectDistance'); return str(ex.get('LensModel', '')), r('FocalLength'), r('FNumber'), (d if d and 0 < d < 1e4 else None)
    except Exception: return None


def corrections_from_exif(path, width, height):
    """(gain_fn, warp_coefficients) for a raw from the lensfun profile of its lens: (None, None) when there is no database, no such lens, or no data for the shot."""
    shot = shot_of(path)
    if not shot or not shot[0] or not shot[1]: return None, None
    lens = find(shot[0])
    return (None, None) if lens is None else corrections(lens, width, height, shot[1], shot[2], shot[3])


def from_exif(path, width, height, lens_setting='auto'):
    """The opcode list for a raw from the lensfun profile of its lens alone (see lenscorr.from_file for the choice between sources), or None."""
    from stratastills import lenscorr
    gain, warp = corrections_from_exif(path, width, height); sh, ge = lenscorr.parts(lens_setting)
    return lenscorr.opcodes_from(gain if sh else None, warp if ge else None, width, height)
