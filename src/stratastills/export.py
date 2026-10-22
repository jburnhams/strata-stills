"""Exporting the best-quality final picture of an item.

  1  develop the primary variant at full size (develop.py: lens corrections, noise-aware denoise, shadow/highlight recovery, exposure, tone) - a raw is developed, any other still is read as it is
  2  level it (transforms.py): the angle is measured on the working image and the turn is applied to the full-size picture, with the zoom that crops the empty corners
  3  choose the size: `auto` is the detail limit measured on the picture itself (detail.py), `native` keeps every pixel, a number is the long side in pixels (never above native: nothing is invented)
  4  resample (Lanczos for a small reduction, area averaging for a big one) and sharpen with the Wiener filter of the measured blur (detail.sharpen_gain), luminance only, no halos
  5  write it: JPEG (quality 95, no chroma subsampling) or 16 bit TIFF, sRGB profile embedded, the camera's EXIF kept (orientation reset), under `<project>/exports/`, with a `.json` beside it that records every decision, so the result can be explained and reproduced.

Nothing is ever written into a source folder."""
import datetime as dt, io, json, os

import cv2, numpy as np

from stratastills import detail, develop, images, meta, preview, projects as P, transforms

VERSION = 1
FORMATS = ('jpeg', 'tiff')


def exports_dir(pid):
    d = os.path.join(P.folder(pid), 'exports'); os.makedirs(d, exist_ok=True); return d


def _is_camera_raw(path): return os.path.splitext(path)[1].lower() in ('.dng', '.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf', '.rw2')


def _read_full(path, **kw):
    """(float32 RGB 0..1 at full size, how it was made). A raw that is not a plain camera DNG (a Lightroom linear DNG) is read at its largest embedded level."""
    if _is_camera_raw(path):
        try: return develop.develop(path, half=False, **kw), 'developed from the raw'
        except Exception as e:
            try: return images.load(path, 100000)[:, :, ::-1].astype(np.float32) / 255.0, f'embedded level of the DNG ({str(e)[:60]})'
            except Exception as e2: raise images.ImageError(str(e2))
    return images.load(path, 100000)[:, :, ::-1].astype(np.float32) / 255.0, 'read as it is (not a raw)'


def measure_detail(pid, path):
    """The detail measurement of a file, from the cache of the project when there (it takes some seconds on a 48 MP raw)."""
    p = images.cache_path(P.folder(pid), images.cache_key(path, 'detail', develop.VERSION, detail.TILE), '.json')
    if os.path.exists(p):
        try: return json.load(open(p))
        except ValueError: pass
    y = develop.analysis_luma(path) if _is_camera_raw(path) else cv2.cvtColor(images.load(path, 100000), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    r = detail.effective_resolution(y); r['native'] = [int(y.shape[1]), int(y.shape[0])]; r['recommended'] = detail.recommended_size(r, max(y.shape))
    with open(p, 'w') as f: json.dump(r, f)
    return r


def choose_size(size, native_long, res):
    """The long side to export at, and why: `size` is 'auto', 'native' or a number of pixels."""
    if size in (None, 'auto'):
        rec = res.get('recommended')
        return (rec, 'the detail limit measured on this picture') if rec else (native_long, 'detail could not be measured (too little texture): native size')
    if size == 'native': return native_long, 'native size, as asked'
    n = int(size)
    if n < 64: raise P.ProjectError('the size is too small')
    return min(n, native_long), ('as asked' if n <= native_long else 'as asked, limited to native: nothing is invented')


def _icc():
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()


def _exif_of(item):
    """The EXIF block of the item's camera JPEG (orientation reset: the pixels are already upright), or None."""
    from PIL import Image
    for v in item['variants']:
        if v['ext'] in ('.jpg', '.jpeg'):
            try:
                with Image.open(v['path']) as im:
                    ex = im.getexif(); ex[0x0112] = 1; return ex.tobytes()
            except Exception: return None
    return None


def write_picture(path, rgb, fmt, quality=95, exif=None):
    from PIL import Image
    if fmt == 'tiff':
        import tifffile
        icc = _icc(); tifffile.imwrite(path, (np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16), photometric='rgb', compression='deflate', extratags=[(34675, 7, len(icc), icc, True)])
    else:
        Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, 'JPEG', quality=quality, subsampling=0, optimize=True, icc_profile=_icc(), **({'exif': exif} if exif else {}))


def export_item(pid, iid, variant=None, size='auto', fmt='jpeg', level=True, sharpen=True, quality=95, progress=None, **develop_kw):
    """Make the final picture; returns the sidecar dict (with `file`). `progress(text)` is called as the steps go. `develop_kw` are the develop() options (ev, denoise, recover)."""
    say = progress or (lambda t: None)
    if fmt not in FORMATS: raise P.ProjectError(f'unknown format {fmt!r}')
    item = P.get_item(pid, iid); vid = variant or item['primary']; v = next((x for x in item['variants'] if x['id'] == vid), None)
    if v is None: raise P.ProjectError(f'item {iid!r} has no variant {vid!r}')
    path = v['path']; rec = dict(version=VERSION, created=dt.datetime.now().astimezone().isoformat(timespec='seconds'), item=iid, stem=item['stem'], variant=vid, source=path, develop_version=develop.VERSION, steps=[])
    say('developing the full-size picture'); rgb, how = _read_full(path, **develop_kw); rec['steps'].append(how); h, w = rgb.shape[:2]; rec['native'] = [w, h]
    if level:
        say('levelling')
        work = images.working(P.folder(pid), path, preview.WORK_PX); ctx = preview._ctx(item); info = transforms.apply('level', work, ctx)[1]; rec['level'] = {k: info[k] for k in ('applied', 'angle', 'confidence', 'reason') if k in info}
        if info.get('applied'): rgb = transforms.rotate_fill(rgb, info['angle']); rec['steps'].append(f"levelled by {info['angle']:+.2f} degrees")
    say('measuring the real detail'); res = measure_detail(pid, path); rec['detail'] = {k: res.get(k) for k in ('ok', 'f50', 'f30', 'f20', 'scale50', 'scale30', 'scale20', 'recommended', 'alpha')}
    long_native = max(w, h); out_long, why = choose_size(size, long_native, res); rec['size'] = dict(long_side=out_long, why=why)
    s = out_long / long_native
    if s < 0.999:
        say('resizing'); rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_LANCZOS4 if s > 0.5 else cv2.INTER_AREA); rec['steps'].append(f'resized to {rgb.shape[1]}x{rgb.shape[0]}')
    if sharpen and res.get('ok'):
        say('sharpening to the measured blur'); g = detail.sharpen_gain(np.array(res['freq']), np.array(res['mtf']), rgb.shape[:2], s); rgb = detail.sharpen(rgb, g); rec['steps'].append('sharpened with the Wiener filter of the measured MTF')
    rec['output'] = [int(rgb.shape[1]), int(rgb.shape[0])]
    name = f"{item['stem']}{'' if vid in ('raw', 'jpeg') else '-' + vid.split('.')[0]}-{rgb.shape[1]}x{rgb.shape[0]}.{'tif' if fmt == 'tiff' else 'jpg'}"
    out = os.path.join(exports_dir(pid), name); say('writing'); write_picture(out, rgb, fmt, quality, _exif_of(item)); rec['file'] = out; rec['bytes'] = os.path.getsize(out); rec['format'] = fmt
    with open(os.path.splitext(out)[0] + '.json', 'w') as f: json.dump(rec, f, indent=1)
    return rec


def list_exports(pid, stem=None):
    """The exports of a project (newest first), from their sidecars: [{file, name, bytes, output, size, created, stem, level, detail, steps}]."""
    d = exports_dir(pid); out = []
    for n in sorted(os.listdir(d)):
        if not n.endswith('.json'): continue
        try: r = json.load(open(os.path.join(d, n)))
        except ValueError: continue
        if os.path.exists(r.get('file', '')) and (stem is None or r.get('stem') == stem): r['name'] = os.path.basename(r['file']); out.append(r)
    return sorted(out, key=lambda r: r['created'], reverse=True)
