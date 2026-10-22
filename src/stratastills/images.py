"""Reading a picture file into a working image, and the cache of working images and previews inside a project.

`load(path, px)` gives a BGR uint8 array, the right way up, at most `px` on its longest side:
  JPEG / PNG / TIFF / WebP   Pillow (EXIF orientation applied)
  camera raw (.dng ...)      LibRaw through rawpy: half-size demosaic with the camera white balance (fast, and plenty for a preview or analysis; a full-size develop is a later stage)
  Lightroom linear / merged-pano DNG (LibRaw refuses these)  the biggest reduced-size level inside the file that is at least `px`, read with tifffile (JPEG XL needs imagecodecs)
`cached(project_dir, path, kind, px, make)` keeps the result of `make()` as a JPEG under `<project>/cache/`, keyed by the file's path, size and modification time, so a changed file is a new picture and nothing stale is served."""
import hashlib, json, os

import cv2, numpy as np

CACHE = 'cache'
JPEG_QUALITY = 90


class ImageError(RuntimeError): pass


def _fit(bgr, px):
    h, w = bgr.shape[:2]; s = px / max(h, w)
    return cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr


def _read_pillow(path, px):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im.draft('RGB', (px, px))                                     # JPEG: decode at a reduced size (much faster for 48 MP files)
        im = ImageOps.exif_transpose(im).convert('RGB')
        return _fit(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), px)


def _read_raw(path, px):
    import rawpy
    with rawpy.imread(path) as r:
        rgb = r.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
    return _fit(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), px)


def _read_linear_dng(path, px):
    """A demosaiced (LinearRaw) DNG: pick the smallest embedded level that has at least `px` on the long side (else the largest), as 16-bit, and show it with the sRGB curve."""
    import tifffile
    with tifffile.TiffFile(path) as f:
        levels = [s for s in f.series if len(s.shape) == 3 and s.shape[2] == 3 and s.dtype == np.uint16]
        if not levels: raise ImageError('no 16-bit RGB level in the DNG')
        big = [s for s in levels if max(s.shape[:2]) >= px]
        s = min(big, key=lambda s: max(s.shape[:2])) if big else max(levels, key=lambda s: max(s.shape[:2]))
        a = s.asarray().astype(np.float32) / 65535.0
    return _fit(cv2.cvtColor((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR), px)


def load(path, px=2048):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.dng', '.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf', '.rw2'):
        try: return _read_raw(path, px)
        except Exception as e:
            try: return _read_linear_dng(path, px)
            except Exception as e2: raise ImageError(f'cannot read {os.path.basename(path)}: {e}; {e2}')
    try: return _read_pillow(path, px)
    except Exception as e: raise ImageError(f'cannot read {os.path.basename(path)}: {e}')


def cache_key(path, *parts):
    st = os.stat(path)
    return hashlib.sha1(json.dumps([os.path.normcase(os.path.abspath(path)), st.st_size, int(st.st_mtime), *parts]).encode()).hexdigest()[:20]


def cache_path(project_dir, key, ext='.jpg'):
    d = os.path.join(project_dir, CACHE); os.makedirs(d, exist_ok=True); return os.path.join(d, key + ext)


def write_jpeg(path, bgr):
    tmp = path + '.tmp.jpg'; cv2.imwrite(tmp, bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]); os.replace(tmp, path)


def working(project_dir, path, px=2048):
    """The working image of a file (BGR, long side <= px), from the cache when there."""
    p = cache_path(project_dir, cache_key(path, 'working', px))
    if os.path.exists(p):
        im = cv2.imread(p)
        if im is not None: return im
    im = load(path, px); write_jpeg(p, im); return im
