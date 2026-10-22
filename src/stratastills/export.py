"""The building blocks of generating the best-quality final picture of an item (the stages themselves, their configs, presets and batch runs are in pipeline.py):
reading and developing the source at full size, measuring its real detail (cached in the project), choosing the export size, and writing the JPEG or 16 bit TIFF with the sRGB profile and the camera's EXIF (orientation reset).
Nothing is ever written into a source folder.
"""
import json, os

import cv2, numpy as np

from stratastills import detail, develop, images, projects as P

FORMATS = ('jpeg', 'tiff')


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
        if v['kind'] != 'generated' and v['ext'] in ('.jpg', '.jpeg'):
            try:
                with Image.open(v['path']) as im:
                    ex = im.getexif(); ex[0x0112] = 1; return ex.tobytes()
            except Exception: return None
    return None


def write_picture(path, rgb, fmt, quality=95, exif=None):
    """Write the float picture: 16 bit TIFF, or JPEG at `quality` (an int; 'auto' is chosen by the caller from the 8-bit picture, see jpeg_quality)."""
    from PIL import Image, ImageFile
    ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, 1 << 27)                                 # with optimize=True Pillow fails ("Suspension not allowed here") when the encoder's buffer is smaller than the file
    if fmt == 'tiff':
        import tifffile
        icc = _icc(); tifffile.imwrite(path, (np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16), photometric='rgb', compression='deflate', extratags=[(34675, 7, len(icc), icc, True)])
    else:
        Image.fromarray(to_u8(rgb)).save(path, 'JPEG', quality=int(quality), subsampling=0, optimize=True, icc_profile=_icc(), **({'exif': exif} if exif else {}))


def to_u8(rgb): return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
