"""Input profiles: what a file is and what the device that made it needs, so that pictures from different sources get the right treatment without the pipeline stages knowing about devices.

`describe(path)` says what kind of file it is: 'raw' (a mosaic LibRaw reads: develop.py), 'linear-dng' (demosaiced RGB in a DNG, e.g. a Lightroom panorama merge: lineardng.py reads it into the same form) or 'rendered' (a JPEG, TIFF, PNG: no development), with the camera's make and model.
`PROFILES` holds per-device settings, keyed by (make, model) in lower case; `profile(make, model)` finds them (a make-only entry is the fallback). What a profile can set:
  bundle       the file name (in this package) of the camera's own colour profile (colour matrices, hue/saturation map, tone curve: the tags its raw DNGs carry). Lightroom's panorama merge replaces them with Adobe Standard's (a stronger map, a look table, and the tone curve left to Lightroom's renderer: measured, the map alone takes saturation up to 2x and brightness up to 1.9x), so for files whose ProfileName says Adobe Standard the camera's own are put back: the pictures then look like the camera's other pictures
To support a new device: add an entry here (and, if its files are a new kind, a reader beside lineardng.py) and a test; the pipelines themselves stay the same."""
import json, os

PROFILES = {
    ('dji', 'fc8482'): dict(label='DJI Mini 4 Pro', bundle='dji_fc8482_profile.json'),                                          # taken from its raw DNGs
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


def adjust_tags(tags):
    """The DNG tags to develop with: a file Lightroom wrote with the Adobe Standard profile gets its device's own colour profile (see `bundle`); anything else is returned as it is (the dict is changed in place)."""
    if _text(tags.get('ProfileName')) != 'adobe standard': return tags
    name = profile(tags.get('Make'), tags.get('Model')).get('bundle')
    if name:
        for k, v in _bundle(name).items(): tags[k] = tuple(v) if isinstance(v, list) else v
        tags['ProfileName'] = 'Embedded'
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
