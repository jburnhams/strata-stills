"""Nudity filter: a picture with exposed nudity in it is skipped for EVERY purpose (the project's items, processing, exports, the reports, the blind arena), so none of them ever shows, makes or ships it.

The check runs locally (NudeNet, an ONNX detector of exposed body parts; the weights ship inside its pip package, nothing is uploaded) on a small decode of the picture's camera JPEG, or of the raw when there is no JPEG. The result (the highest score of the 'exposed' classes and which class) is remembered per file, size and time stamp in `<STRATASTILLS_HOME>/nsfw_cache.json`, so a file is looked at once however many reports and projects list it. Originals are never touched.
`ingest.scan` is the one place every consumer lists pictures through, so it applies `keep_shots`; the scripts that list files themselves (build_report.py, lr_dataset.py) call `flagged(path)`.

Settings (environment first, then `~/.stratastills/settings.json`):
  STRATASTILLS_NSFW=0          or `nsfw_filter: false`: switch the filter off (the tests do). On by default: with it on and no detector installed, listing a folder fails with a clear message rather than letting pictures through.
  STRATASTILLS_NSFW_THRESHOLD  or `nsfw_threshold`: 0..1, a picture is flagged when its highest 'exposed' score is at least this (default 0.45; lower flags more, higher less).
Every skip is logged (loguru warning: file, class, score, threshold) and appended to `<STRATASTILLS_HOME>/nsfw_skipped.jsonl`; `python -m stratastills.nsfw <folder|file> ...` prints the verdict per picture."""
import json, os, sys, threading, time

DEFAULT_THRESHOLD = 0.45
EXPOSED = ('FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED', 'FEMALE_BREAST_EXPOSED', 'ANUS_EXPOSED', 'BUTTOCKS_EXPOSED')                    # not MALE_BREAST_EXPOSED (a bare chest) and not the *_COVERED classes
MODEL = 'nudenet-320n'
PX = 1280                                                                                                                                       # long side of the decode the detector gets (it resizes to its own input size)

_detector = None                                    # callable(bgr ndarray) -> [{'class', 'score', ...}]; set_detector() replaces it (tests)
_cache = {}                                         # path|size|mtime -> [score, class]; loaded from disk once
_loaded = False
_logged = set()
_lock = threading.RLock()                         # the server asks for several listings at once; one writer of the cache file at a time


class NsfwUnavailable(RuntimeError): pass


def home():
    from stratastills import settings
    return settings.home()


def enabled():
    from stratastills import settings
    e = os.environ.get('STRATASTILLS_NSFW')
    if e is not None: return e.strip().lower() not in ('0', 'false', 'off', 'no', '')
    return bool(settings.load().get('nsfw_filter', True))


def threshold():
    from stratastills import settings
    v = os.environ.get('STRATASTILLS_NSFW_THRESHOLD') or settings.load().get('nsfw_threshold')
    try: return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError): return DEFAULT_THRESHOLD


def set_detector(fn):
    """Replace the detector (a callable from a BGR uint8 array to NudeNet-style detections); None goes back to NudeNet. Clears nothing on disk: the cache key carries the model name."""
    global _detector
    _detector = fn


def _default_detector():
    try:
        from nudenet import NudeDetector
    except ImportError as e:
        raise NsfwUnavailable('the nudity filter is on but NudeNet is not installed (pip install nudenet onnxruntime; or STRATASTILLS_NSFW=0 to switch the filter off)') from e
    d = NudeDetector()
    return lambda bgr: d.detect(bgr)


def _detect(bgr):
    global _detector
    if _detector is None: _detector = _default_detector()
    return _detector(bgr)


def _store():
    from stratastills import nsfw_store
    return nsfw_store.get()                         # the files (default) or, once settings.database_url() is set, schema `photos` (nsfw_store.py)


def _load():
    global _loaded, _cache
    with _lock:
        if _loaded: return
        _cache = _store().load(); _loaded = True


def reset():
    """Forget the in-memory cache and the logged set (tests; a changed STRATASTILLS_HOME)."""
    global _loaded, _cache
    _loaded = False; _cache = {}; _logged.clear()


def _key(path):
    st = os.stat(path); return f'{MODEL}|{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}'


def score(path):
    """(highest score of the exposed classes, that class or '') of a picture file, from the cache when seen before. A file that cannot be decoded scores (1.0, 'unreadable'): a picture nobody could check is not let through."""
    _load(); k = _key(path)
    if k in _cache: return tuple(_cache[k])
    from stratastills import images
    try: bgr = images.load(path, PX)
    except Exception: res = (1.0, 'unreadable')
    else:
        best = (0.0, '')
        for d in _detect(bgr) or []:
            if d.get('class') in EXPOSED and float(d.get('score', 0)) > best[0]: best = (float(d['score']), d['class'])
        res = best
    with _lock: _cache[k] = list(res); _store().put(k, list(res), _cache)
    return res


def _allowed():
    """The keys of the cache the person has cleared (a set; scripts test `_key(path) in _allowed()`)."""
    _load(); return _store().allowed_among(list(_cache))


def allow(path, yes=True):
    """Review decision: let this picture (as it is now: size and time stamp) through the filter, or put it back (yes=False)."""
    with _lock: _store().set_allowed(_key(path), yes)


def flagged(path, thr=None):
    """None if the picture may be used, else {'path', 'score', 'class', 'threshold'}. Always None with the filter off, and for a picture the person cleared on the review page (`allow`)."""
    if not enabled(): return None
    thr = threshold() if thr is None else thr; s, c = score(path)
    if s >= thr and _store().allowed_among([_key(path)]): return None
    return dict(path=path, score=round(s, 3), cls=c, threshold=thr) if s >= thr else None


def review():
    """Every picture the cache has seen that scores at or above the threshold, for the review page: [{path, score, cls, allowed}] by score, highest first. Files that changed or went are not listed."""
    _load(); thr = threshold(); out = []
    for k, (sc, cls) in list(_cache.items()):
        model, _, rest = k.partition('|')
        if model != MODEL or sc < thr: continue
        path, size, mt = rest.rsplit('|', 2)
        try: cur = _key(path)
        except OSError: continue
        if cur == k: out.append(dict(path=path, score=round(sc, 3), cls=cls, allowed=False))
    al = _store().allowed_among([r['path'] and _key(r['path']) for r in out])                                  # one lookup for all flagged pictures
    for r in out: r['allowed'] = _key(r['path']) in al
    return sorted(out, key=lambda r: -r['score'])


def _record(info):
    """Log a skip once per process and keep it in the skipped list."""
    if info['path'] in _logged: return
    _logged.add(info['path'])
    try:
        from loguru import logger; logger.warning('nudity filter: skipped {path} ({cls} {score} >= {threshold})', **info)
    except ImportError: print('nudity filter: skipped {path} ({cls} {score} >= {threshold})'.format(**info), file=sys.stderr)
    try: _store().record_skip(dict(info, t=int(time.time()), reason='nudity'))
    except OSError: pass


def _probe(shot):
    """The file to look at for a shot: the cheapest decode, the camera JPEG or another still, else the raw."""
    return shot.jpeg or next((p for _t, p in shot.derived if os.path.splitext(p)[1].lower() not in ('.dng', '.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf', '.rw2')), None) or shot.raw


def keep_shots(shots):
    """`shots` without the flagged ones (a video has no still to check and passes). The check is on the shot's probe file: its other renderings are the same picture."""
    if not enabled(): return shots
    out = []
    for sh in shots:
        p = None if sh.kind == 'video' else _probe(sh); hit = flagged(p) if p and os.path.exists(p) else None
        if hit: _record(hit)
        else: out.append(sh)
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    paths = []
    for a in argv:
        if os.path.isdir(a): paths += [os.path.join(a, f) for f in sorted(os.listdir(a)) if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')]
        else: paths.append(a)
    thr = threshold(); n = 0
    for p in paths:
        s, c = score(p); hit = s >= thr; n += hit; print(f'{"FLAGGED" if hit else "ok     "} {s:.3f} {c or "-":28s} {p}')
    print(f'{n} of {len(paths)} flagged at threshold {thr}'); return 0


if __name__ == '__main__': sys.exit(main())
