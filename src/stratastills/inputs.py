"""Input profiles: what a file is and what the device that made it needs, so that pictures from different sources get the right treatment without the pipeline stages knowing about devices.

`describe(path)` says what kind of file it is: 'raw' (a mosaic LibRaw reads: develop.py), 'linear-dng' (demosaiced RGB in a DNG, e.g. a Lightroom panorama merge: lineardng.py reads it into the same form) or 'rendered' (a JPEG, TIFF, PNG: no development), with the camera's make and model.
`PROFILES` holds per-device settings, keyed by (make, model) in lower case; `profile(make, model)` finds them (a make-only entry is the fallback). What a profile can set:
  tone_curve   the file name (in this package) of the tone curve to use when the file has no ProfileToneCurve of its own: a linear DNG written by Lightroom has none, and without one the picture would not look like the camera's other pictures
To support a new device: add an entry here (and, if its files are a new kind, a reader beside lineardng.py) and a test; the pipelines themselves stay the same."""
import json, os

PROFILES = {
    ('dji', 'fc8482'): dict(label='DJI Mini 4 Pro', tone_curve='dji_fc8482_tone_curve.json'),                                  # the curve is the camera's own ProfileToneCurve (from its raw DNGs)
}
_cache = {}


def _text(v):
    if isinstance(v, bytes): v = v.decode('latin-1', 'ignore')
    return str(v or '').strip().strip('\x00').lower()


def profile(make, model):
    """The profile of a device (dict, maybe empty): the (make, model) entry, else a (make, None) one."""
    return PROFILES.get((_text(make), _text(model))) or PROFILES.get((_text(make), None)) or {}


def tone_curve(tags):
    """The tone curve (pairs of input, output) to develop with: the file's own ProfileToneCurve, else its device profile's, else None (plain sRGB)."""
    own = tags.get('ProfileToneCurve')
    if own is not None: return own
    name = profile(tags.get('Make'), tags.get('Model')).get('tone_curve')
    if not name: return None
    if name not in _cache: _cache[name] = json.load(open(os.path.join(os.path.dirname(__file__), name)))
    return _cache[name]


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
