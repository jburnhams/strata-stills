"""Pictures for the app: an item's variant with a transform applied, at a preview size, cached inside the project (`<project>/cache/`).

`render(pid, iid, variant, transform, px)` returns (path of a JPEG, info of the transform). The first request for a file decodes it into a working image (WORK_PX) (kept in the cache); everything after is cheap.
The info (see transforms.py: for `level` the angle, confidence and whether it was applied) is kept next to the picture so the app can say what was done."""
import json, os

import cv2, numpy as np

from stratastills import develop, fullres, images, meta, projects as P, transforms

MAX_OUT = 2048                  # the biggest crop the loupe returns, in pixels a side
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


class NotReady(Exception):
    """The full-resolution copy of a variant is still being made (fullres.py): `status` says how far."""
    def __init__(self, status): super().__init__(status.get('text') or status.get('state')); self.status = status


def _variant(pid, iid, variant):
    item = P.get_item(pid, iid); vid = variant or item['primary']; v = next((x for x in item['variants'] if x['id'] == vid), None)
    if v is None: raise P.ProjectError(f'item {iid!r} has no variant {vid!r}')
    return item, v


def _size(v):
    """(width, height) of a variant at its full size, from its header (a generated variant records it)."""
    return tuple(v['output']) if v.get('output') else images.dims(v['path'])


def fullres_status(pid, iid, variant=None):
    """Start making (if needed) and report the full-resolution copy of a variant that the loupe will cut crops from."""
    _, v = _variant(pid, iid, variant); return fullres.prepare(pid, v['path'])


def crop(pid, iid, variant=None, transform='original', cx=0.5, cy=0.5, size=200, zoom=2):
    """A JPEG (bytes) of the loupe: the same SCENE AREA of any variant, at its full resolution.
    `size` is the width of the area in pixels of the biggest variant of the item (so 200 is 200 px of the 48 MP raw); a variant with fewer pixels (a 6016 px export) is cut proportionally smaller and enlarged to match, so the same place in the scene is shown by every variant and what differs is
    the detail, sharpness, grain and artefacts of each. The area is centred on (cx, cy), 0..1 of the picture, and the result is enlarged `zoom` times (cubic; shown 1:1 for the biggest variant at zoom 1).
    `transform` 'level' turns the picture by the angle found for that variant (transforms.py) while cutting, so only the crop is resampled, not the whole picture. Raises NotReady while the variant's full-size copy is being made."""
    if transform not in transforms.FUNCS: raise P.ProjectError(f'unknown transform {transform!r}')
    item, v = _variant(pid, iid, variant); arr = fullres.get(pid, v['path'])
    if arr is None: raise NotReady(fullres.prepare(pid, v['path']))
    h, w = arr.shape[:2]; ref_w = max([_size(x)[0] for x in item['variants']] + [w]); zoom = max(1, min(int(zoom), 8)); size = int(max(16, min(size, 1024)))
    s = int(max(16, min(round(size * w / ref_w), h, w))); out = min(size * zoom, MAX_OUT)
    x0 = int(round(min(max(cx * w - s / 2, 0), w - s))); y0 = int(round(min(max(cy * h - s / 2, 0), h - s)))
    info = render(pid, iid, variant, transform, WORK_PX)[1] if transform != 'original' else {}
    if info.get('applied'):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), info['angle'], transforms.inscribed_scale(w, h, info['angle'])); M[:, 2] -= (x0, y0)
        c = cv2.warpAffine(np.asarray(arr), M, (s, s), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    else: c = np.ascontiguousarray(arr[y0:y0 + s, x0:x0 + s])
    if out != s: c = cv2.resize(c, (out, out), interpolation=cv2.INTER_CUBIC if out > s else cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', c, [cv2.IMWRITE_JPEG_QUALITY, 95]); return buf.tobytes()
