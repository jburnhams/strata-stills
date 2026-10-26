"""Input profiles: what a file is and what the device that made it needs, so that pictures from different sources get the right treatment without the pipeline stages knowing about devices.

`describe(path)` says what kind of file it is: 'raw' (a mosaic LibRaw reads: develop.py), 'linear-dng' (demosaiced RGB in a DNG, e.g. a Lightroom panorama merge: lineardng.py reads it into the same form) or 'rendered' (a JPEG, TIFF, PNG: no development), with the camera's make and model.
`PROFILES` holds per-device settings, keyed by (make, model) in lower case; `profile(make, model)` finds them (a make-only entry is the fallback). What a profile can set:
  bundle       the file name (in this package) of the camera's own colour profile (colour matrices, hue/saturation map, tone curve: the tags its raw DNGs carry). Lightroom's panorama merge replaces them with Adobe Standard's (a stronger map, a look table, and the tone curve left to Lightroom's renderer: measured, the map alone takes saturation up to 2x and brightness up to 1.9x), so for files whose ProfileName says Adobe Standard the camera's own are put back: the pictures then look like the camera's other pictures
To support a new device: add an entry here (and, if its files are a new kind, a reader beside lineardng.py) and a test; the pipelines themselves stay the same."""
import json, os
import numpy as np

PROFILES = {
    ('dji', 'fc8482'): dict(label='DJI Mini 4 Pro', bundle='dji_fc8482_profile.json'),                                          # taken from its raw DNGs
    ('sony', 'ilce-7c'): dict(label='Sony A7C', bundle='sony_ilce7c_profile.json'),                                             # RawTherapee's public-domain DCP (Adobe DNG Converter) + darktable's noise fits: scripts/build_profile_bundle.py
    ('sony', 'nex-5n'): dict(label='Sony NEX-5N', bundle='sony_nex5n_profile.json'),
}
_cache = {}


def _text(v):
    if isinstance(v, bytes): v = v.decode('latin-1', 'ignore')
    return str(v or '').strip().strip('\x00').lower()


def profile(make, model):
    """The profile of a device (dict, maybe empty): the (make, model) entry, else a (make, None) one."""
    return PROFILES.get((_text(make), _text(model))) or PROFILES.get((_text(make), None)) or {}


def _bundle(name):
    if name not in _cache: _cache[name] = json.load(open(os.path.join(os.path.dirname(__file__), name)))
    return _cache[name]


def noise_scale(spec, iso):
    """The factor of a profile's `NoiseScale`: a number, or a table [[iso, factor]] interpolated in log-log between its ISOs (the end ones beyond; the plain number when the ISO is not known)."""
    if spec is None: return None
    if not isinstance(spec, (list, tuple)): return float(spec)
    t = sorted(spec); xs = np.log([r[0] for r in t]); ys = np.log([r[1] for r in t])
    return float(np.exp(np.interp(np.log(iso), xs, ys))) if iso else float(np.exp(np.mean(ys)))


def adjust_tags(tags, iso=None):
    """The DNG tags to develop with: a file Lightroom wrote with the Adobe Standard profile gets its device's own colour profile (see `bundle`); and every file whose device profile has a `LookTable` (the hue/saturation look fitted to the camera's JPEGs: scripts/fit_look_table.py) gets it and has its noise profile multiplied by the profile's `NoiseScale` (a factor, or a table over the ISO; how much stronger the clean-up has to be than the profile says to leave what the camera's own JPEG leaves: scripts/fit_noise_scale.py), whatever kind of file it is (the dict is changed in place)."""
    name = profile(tags.get('Make'), tags.get('Model')).get('bundle')
    if not name: return tags
    if _text(tags.get('ProfileName')) == 'adobe standard':
        for k, v in _bundle(name).items(): tags[k] = tuple(v) if isinstance(v, list) else v
        tags['ProfileName'] = 'Embedded'
    if 'LookTable' in _bundle(name): tags.setdefault('LookTable', _bundle(name)['LookTable'])
    k = noise_scale(_bundle(name).get('NoiseScale'), iso)
    if k and tags.get('NoiseProfile') is not None: tags['NoiseProfile'] = (tags['NoiseProfile'][0] * k, tags['NoiseProfile'][1] * k)                      # the device's calibration of its noise profile (scripts/fit_noise_scale.py), whether the profile is the file's own or made from a table
    return tags


def _rat(vals, d=1000000): return tuple(x for v in vals for x in (int(round(float(v) * d)), d))


def _iso(path):
    import tifffile
    try:
        with tifffile.TiffFile(path) as t: return float(t.pages[0].tags['ExifTag'].value['ISOSpeedRatings'])
    except Exception: return None


def noise_profile(table, iso):
    """The DNG NoiseProfile (S, O) for `iso` from a bundle's NoiseTable [[iso, S, O]]: interpolated between the neighbouring ISOs (S and O scale with the gain, so in log-log), the end ones beyond."""
    t = sorted(table); xs = [np.log(r[0]) for r in t]
    return tuple(float(np.exp(np.interp(np.log(iso), xs, [np.log(max(r[k], 1e-12)) for r in t]))) for k in (1, 2))


def synthesize_tags(path, tags, lens='auto'):
    """A raw that is not a DNG (Sony .ARW ...) has no colour tags, so they are made here: the as-shot white balance from LibRaw, and the colour from the device's profile bundle when there is one (matrices and hue/saturation maps of a DNG profile; the noise profile for the shot's ISO), else LibRaw's own XYZ -> camera matrix (one matrix serves as both illuminants, so the weight between them does not matter). Tags the file has are kept."""
    if 'ColorMatrix1' in tags or os.path.splitext(path)[1].lower() == '.dng': return tags
    try:
        import rawpy
        with rawpy.imread(path) as r:
            m = r.rgb_xyz_matrix[:3]; wb = r.camera_whitebalance; z = r.sizes; w, h = (z.crop_width, z.crop_height) if z.crop_width else (z.iwidth, z.iheight); w, h = (h, w) if z.flip in (5, 6) else (w, h)
    except Exception: return tags
    if wb[1] <= 0 or wb[0] <= 0 or wb[2] <= 0: return tags
    if lens != 'off' and 'OpcodeList3' not in tags:                                                                           # the camera's own shading, distortion and colour-fringe tables, as the opcodes a DNG would carry (lenscorr.py)
        from stratastills import lenscorr
        ops = lenscorr.from_file(path, w, h, lens)
        if ops: tags['OpcodeList3'] = ops
    tags['AsShotNeutral'] = _rat([wb[1] / wb[0], 1.0, wb[1] / wb[2]]); name = profile(tags.get('Make'), tags.get('Model')).get('bundle')
    if name:
        b = _bundle(name)
        for k, v in b.items():
            if k not in ('NoiseTable', 'ForwardMatrix1', 'ForwardMatrix2'): tags.setdefault(k, tuple(v) if isinstance(v, list) else v)
        iso = _iso(path)
        if 'NoiseTable' in b and iso: tags['NoiseProfile'] = noise_profile(b['NoiseTable'], iso)
    elif m.any(): tags['ColorMatrix1'] = tags['ColorMatrix2'] = _rat(m.reshape(-1)); tags['CalibrationIlluminant1'] = 17; tags['CalibrationIlluminant2'] = 21; tags['ProfileName'] = 'LibRaw'
    return tags


def tone_curve(tags):
    """The tone curve (pairs of input, output) to develop with: the file's own ProfileToneCurve, else its device profile's, else None (plain sRGB)."""
    own = tags.get('ProfileToneCurve')
    if own is not None: return own
    name = profile(tags.get('Make'), tags.get('Model')).get('bundle'); return _bundle(name).get('ProfileToneCurve') if name else None


def describe(path):
    """{kind, make, model, profile} of a file ('kind' as in the module doc)."""
    ext = os.path.splitext(path)[1].lower(); make = model = None
    if ext in ('.dng', '.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf', '.rw2'):
        from stratastills import develop, lineardng
        try: t = develop.read_dng_tags(path); make, model = _text(t.get('Make')), _text(t.get('Model'))
        except Exception: pass
        kind = 'linear-dng' if ext == '.dng' and lineardng.is_linear(path) and not _has_mosaic(path) else 'raw'
    else: kind = 'rendered'
    return dict(kind=kind, make=make, model=model, profile=profile(make, model).get('label'))


def _has_mosaic(path):
    import tifffile
    try:
        with tifffile.TiffFile(path) as t: return any(p.photometric.name == 'CFA' for p in [t.pages[0]] + list(getattr(t.pages[0], 'pages', []) or []))
    except Exception: return False
