"""Pictures for the app: an item's variant with a transform applied, at a preview size, cached inside the project (`<project>/cache/`).

`render(pid, iid, variant, transform, px)` returns (path of a JPEG, info of the transform). The first request for a file decodes it into a working image (WORK_PX) (kept in the cache); everything after is cheap.
The info (see transforms.py: for `level` the angle, confidence and whether it was applied) is kept next to the picture so the app can say what was done."""
import json, os

import cv2

from stratastills import develop, images, meta, projects as P, transforms

WORK_PX = 4096                  # the working picture: half the native size of a 48 MP drone raw (what the half-size demosaic gives), so the loupe can look at real detail


def _ctx(item):
    """What the transforms need to know about the shot: is it a straight-down one (from the DJI gimbal pitch in the XMP of a JPEG that belongs to it)."""
    for v in item['variants']:
        if v['ext'] in ('.jpg', '.jpeg'):
            d = meta.dji_fields(meta.read_xmp(v['path']))
            if d: return dict(nadir=meta.is_nadir(d), pitch=d.get('GimbalPitchDegree'))
    return dict(nadir=False, pitch=None)


def render(pid, iid, variant=None, transform='original', px=1600):
    if transform not in transforms.FUNCS: raise P.ProjectError(f'unknown transform {transform!r}')
    item = P.get_item(pid, iid); vid = variant or item['primary']
    v = next((x for x in item['variants'] if x['id'] == vid), None)
    if v is None: raise P.ProjectError(f'item {iid!r} has no variant {vid!r}')
    pdir = P.folder(pid); px = max(64, min(int(px), WORK_PX))
    key = images.cache_key(v['path'], 'render', transform, transforms.VERSION[transform], px, develop.VERSION); out = images.cache_path(pdir, key); side = images.cache_path(pdir, key, '.json')
    if os.path.exists(out) and os.path.exists(side):
        try: return out, json.load(open(side))
        except ValueError: pass
    try: work = images.working(pdir, v['path'], WORK_PX)
    except (images.ImageError, OSError) as e: raise P.ProjectError(str(e))
    ctx = _ctx(item); img, info = transforms.apply(transform, work, ctx); info = dict(info, **{k: ctx[k] for k in ('nadir', 'pitch')})
    h, w = img.shape[:2]; s = px / max(h, w)
    if s < 1: img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    images.write_jpeg(out, img)
    with open(side, 'w') as f: json.dump(info, f)
    return out, info


def crop(pid, iid, variant=None, transform='original', cx=0.5, cy=0.5, size=200, zoom=2):
    """A JPEG (bytes) of the `size` px square around the point (cx, cy) (0..1 of the picture's width and height) of a variant with a transform, taken from the working picture and enlarged `zoom` times (cubic) for the loupe. The same normalised point of every variant and transform is the same place in the scene
    (a levelled picture is turned about its centre, so there the point is only the same to within the turn)."""
    path, _ = render(pid, iid, variant, transform, WORK_PX); im = cv2.imread(path)
    if im is None: raise P.ProjectError('cannot read the picture')
    h, w = im.shape[:2]; size = int(max(16, min(size, h, w))); zoom = max(1, min(int(zoom), 6))
    x0 = int(round(min(max(cx * w - size / 2, 0), w - size))); y0 = int(round(min(max(cy * h - size / 2, 0), h - size))); c = im[y0:y0 + size, x0:x0 + size]
    if zoom > 1: c = cv2.resize(c, (size * zoom, size * zoom), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode('.jpg', c, [cv2.IMWRITE_JPEG_QUALITY, 92]); return buf.tobytes()
