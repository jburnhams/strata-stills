"""One output size for a folder of pictures: the 'auto' size is not each picture's own measurement (which wanders by a few pixels with the scene) but the folder's, per native size, so the finished pictures all come out the same, with dimensions that are multiples of 16 (or 8).

The folder's figure is the median of the measured detail limit (detail.RECOMMEND) over a sample of its pictures of that native size (at most SAMPLE, evenly spread; each measurement is cached in the project as it always was), rounded up to a multiple of 64 of the long side, never above native. The plan is cached in the project; it changes when the sample, the development or the measurement does."""
import hashlib, json, math, os, threading

import numpy as np

from stratastills import detail, develop, images, projects as P

SAMPLE = 10
MIN_SHARE = 0.5                 # 'auto' never goes below this share of the native width and height, whatever the measurement says
_lock = threading.Lock()


def output_dims(w, h, long_side):
    """(width, height) of the picture resized to about `long_side` on its long side with the aspect ratio kept: both sides multiples of 16 when the aspect ratio allows it (4:3 and 3:2 at a multiple of 64 do), else of 8, else of 2. The native size is returned unchanged."""
    if long_side >= max(w, h): return int(w), int(h)
    lng, sht = max(w, h), min(w, h)
    for m in (16, 8, 4):
        base = round(long_side / m)
        for k in sorted(range(-4, 5), key=abs):                                                                      # the multiple of m nearest the ask whose short side is also a whole multiple of m
            L = (base + k) * m
            if L <= 0 or L > lng: continue
            s = L * sht / lng
            if abs(s - round(s)) < 1e-6 and round(s) % m == 0: return (L, int(round(s))) if w >= h else (int(round(s)), L)
    L = max(2, 2 * round(long_side / 2)); s = max(2, 2 * round(L * sht / lng / 2)); return (L, s) if w >= h else (s, L)


def _sample(ids):
    if len(ids) <= SAMPLE: return list(ids)
    return [ids[round(i * (len(ids) - 1) / (SAMPLE - 1))] for i in range(SAMPLE)]


def folder_size(pid, item, native, measure, pick):
    """The long side for the pictures of `item`'s source folder that have the native size `native` (w, h), or None when the sample has too little texture to measure. `measure(path)` is export.measure_detail bound to the project; `pick(item)` the source variant to make a picture from."""
    sid = item['id'].partition('~')[0]; group = []
    for it in P.source_items(pid, sid):
        if it['kind'] == 'video': continue
        try: v = pick(it)
        except P.ProjectError: continue
        try: d = images.dims(v['path'])
        except Exception: continue
        if tuple(d) == tuple(native): group.append((it['id'], v['path']))
    group.sort(); chosen = _sample([g[0] for g in group]); paths = dict(group)
    if item['id'] not in paths: return None
    key = hashlib.sha1(json.dumps([sid, list(native), chosen, develop.VERSION, detail.TILE, detail.RECOMMEND]).encode()).hexdigest()[:20]; cache = images.cache_path(P.folder(pid), key, '.json')
    with _lock:
        if os.path.exists(cache):
            try: return json.load(open(cache))['long_side']
            except (ValueError, KeyError): pass
        vals = [r[detail.RECOMMEND] for r in (measure(paths[i]) for i in chosen) if r.get('ok')]
        long_native = max(native); size = int(min(long_native, math.ceil(max(float(np.median(vals)), MIN_SHARE) * long_native / 64) * 64)) if vals else None
        with open(cache, 'w') as f: json.dump(dict(long_side=size, sampled=chosen, measured=len(vals)), f)
        return size
