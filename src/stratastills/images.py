"""Reading a picture file into a working image, and the cache of working images and previews inside a project.

`load(path, px)` gives a BGR uint8 array, the right way up, at most `px` on its longest side:
  JPEG / PNG / TIFF / WebP   Pillow (EXIF orientation applied)
  camera raw (.dng ...)      developed from the raw (develop.py), half size: plenty for a preview or for measuring the tilt; the export develops at full size
  Lightroom linear / merged-pano DNG (LibRaw refuses these)  the biggest reduced-size level inside the file that is at least `px`, read with tifffile (JPEG XL needs imagecodecs)
`cached(project_dir, path, kind, px, make)` keeps the result of `make()` as a JPEG under `<project>/cache/`, keyed by the file's path, size and modification time, so a changed file is a new picture and nothing stale is served."""

import hashlib
import json
import os

import cv2
import numpy as np

CACHE = "cache"
JPEG_QUALITY = 90


class ImageError(RuntimeError):
    pass


def _fit(bgr, px):
    h, w = bgr.shape[:2]
    s = px / max(h, w)
    return cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr


def _read_pillow(path, px):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.draft("RGB", (px, px))  # JPEG: decode at a reduced size (much faster for 48 MP files)
        im = ImageOps.exif_transpose(im).convert("RGB")
        return _fit(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), px)


def _read_tiff(path, px):
    """A TIFF Pillow cannot open (16 bit RGB, as the app's own TIFF exports are): read with tifffile, to 8 bit."""
    import tifffile

    a = tifffile.imread(path)
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    a = a[..., :3]
    a = (a >> 8).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8)
    return _fit(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), px)


def _read_raw(path, px):
    """The picture developed from the raw as in develop.py (half-size demosaic, lens corrections, exposure, tone; colour noise only is removed, the luminance denoise is for the export): quick enough for the app, and looks like the export."""
    from stratastills import develop

    rgb = develop.develop(path, half=True, denoise="chroma")
    return _fit(cv2.cvtColor((rgb * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR), px)


def _read_linear_dng(path, px):
    """A demosaiced (LinearRaw) DNG: pick the smallest embedded level that has at least `px` on the long side (else the largest), as 16-bit, and show it with the sRGB curve."""
    import tifffile

    with tifffile.TiffFile(path) as f:
        levels = [s for s in f.series if len(s.shape) == 3 and s.shape[2] == 3 and s.dtype == np.uint16]
        if not levels:
            raise ImageError("no 16-bit RGB level in the DNG")
        big = [s for s in levels if max(s.shape[:2]) >= px]
        s = min(big, key=lambda s: max(s.shape[:2])) if big else max(levels, key=lambda s: max(s.shape[:2]))
        a = s.asarray().astype(np.float32) / 65535.0
    return _fit(cv2.cvtColor((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR), px)


RAW_EXTS = (".dng", ".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2")


def is_camera_raw(path):
    return os.path.splitext(path)[1].lower() in RAW_EXTS


_dims = {}


def dims(path):
    """(width, height) of the biggest picture in a file, read from its header: Pillow for ordinary pictures, the largest IFD or sub-IFD level for a DNG (the main IFD of a camera DNG is only a thumbnail). Remembered per file (and its size and time stamp): the 'auto' size asks for every picture of a folder for each picture it makes (sizeplan.folder_size), which took 3 s per picture in a folder of 100 raws, and more as the folder grows."""
    try:
        st = os.stat(path)
        key = (path, st.st_size, st.st_mtime_ns)
    except OSError:
        return _dims_of(path)
    if key not in _dims:
        _dims[key] = _dims_of(path)
    return _dims[key]


def _dims_of(path):
    if is_camera_raw(path):
        import tifffile

        with tifffile.TiffFile(path) as t:
            best = (0, 0)
            for pg in [t.pages[0]] + list(getattr(t.pages[0], "pages", []) or []):
                h, w = pg.shape[:2]
                if w * h > best[0] * best[1]:
                    best = (w, h)
        return best
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def load(path, px=2048):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".dng", ".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2"):
        try:
            return _read_raw(path, px)
        except Exception as e:
            try:
                return _read_linear_dng(path, px)
            except Exception as e2:
                raise ImageError(f"cannot read {os.path.basename(path)}: {e}; {e2}")
    try:
        return _read_pillow(path, px)
    except Exception as e:
        if ext in (".tif", ".tiff"):
            try:
                return _read_tiff(path, px)
            except Exception:
                pass
        raise ImageError(f"cannot read {os.path.basename(path)}: {e}")


def _one(v):
    """A TIFF tag value that may come as a 1-tuple."""
    return int(v[0] if isinstance(v, (tuple, list)) else v)


def embedded_preview(path):
    """(JPEG bytes, EXIF orientation) of the largest preview a TIFF-based raw (ARW, DNG, NEF, CR2 ...) carries, read by its offset; None when it has none. CR3/RAF are not TIFF: LibRaw's thumbnail then (it reads more of the file)."""
    import tifffile

    best, orient = None, 1
    try:
        with tifffile.TiffFile(path) as t:
            pages = []
            for pg in t.pages:
                pages.append(pg)
                pages += list(getattr(pg, "pages", None) or [])
            o = t.pages[0].tags.get(274) if t.pages else None
            orient = int(o.value) if o is not None else 1
            for pg in pages:
                tags = pg.tags
                cand = []
                if 513 in tags and 514 in tags:  # JPEGInterchangeFormat(Length): Sony's PreviewImage, a DNG's thumbnail
                    cand.append((_one(tags[513].value), _one(tags[514].value)))
                if pg.compression in (6, 7) and len(pg.dataoffsets) == 1 and pg.photometric not in (32803, 34892):
                    cand.append(
                        (int(pg.dataoffsets[0]), int(pg.databytecounts[0]))
                    )  # a JPEG strip that is not the raw data
                for off, n in cand:
                    if n > 0 and (best is None or n > best[1]):
                        best = (off, n)
            if best is None:
                raise ValueError("no preview")
            fh = t.filehandle
            fh.seek(best[0])
            data = fh.read(best[1])
    except Exception:
        data = None
    if data is None or data[:2] != b"\xff\xd8":
        try:
            import rawpy

            with rawpy.imread(path) as r:
                t, flip = r.extract_thumb(), r.sizes.flip
            if t.format != rawpy.ThumbFormat.JPEG:
                return None
            return bytes(t.data), {3: 3, 5: 8, 6: 6}.get(flip, 1)  # LibRaw's flip as an EXIF orientation
        except Exception:
            return None
    return data, orient


_TURN = {3: cv2.ROTATE_180, 6: cv2.ROTATE_90_CLOCKWISE, 8: cv2.ROTATE_90_COUNTERCLOCKWISE}


def preview_bgr(path, px=2048):
    """The embedded preview of a raw as BGR (turned upright, long side <= px), or None: a quick look at a raw (the nudity filter's probe) without developing it."""
    got = embedded_preview(path)
    if got is None:
        return None
    im = cv2.imdecode(np.frombuffer(got[0], np.uint8), cv2.IMREAD_COLOR)
    if im is None or im.shape[0] * im.shape[1] < 19200:  # a 160x120 thumbnail is too coarse to judge
        return None
    if got[1] in _TURN:
        im = cv2.rotate(im, _TURN[got[1]])
    return _fit(im, px)


def cache_key(path, *parts):
    st = os.stat(path)
    return hashlib.sha1(
        json.dumps([os.path.normcase(os.path.abspath(path)), st.st_size, int(st.st_mtime), *parts]).encode()
    ).hexdigest()[:20]


def cache_path(project_dir, key, ext=".jpg"):
    d = os.path.join(project_dir, CACHE)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, key + ext)


def write_jpeg(path, bgr):
    tmp = path + ".tmp.jpg"
    cv2.imwrite(tmp, bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    os.replace(tmp, path)


def working(project_dir, path, px=2048):
    """The working image of a file (BGR, long side <= px), from the cache when there."""
    from stratastills import develop

    p = cache_path(project_dir, cache_key(path, "working", px, develop.VERSION))
    if os.path.exists(p):
        im = cv2.imread(p)
        if im is not None:
            return im
    im = load(path, px)
    write_jpeg(p, im)
    return im


def readable_jpeg(path, min_std=0.5):
    """The picture at `path` as BGR, or None when it is missing, truncated (Pillow refuses to finish decoding it) or blank (one flat value: the grey picture's std under `min_std`; a real night picture still has a few levels). Used before a develop is cached or served, so a bad one (two GPU workers at once wrote blank neutral pictures, 2026-11-09) is made again instead of being kept."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.load()
    except Exception:
        return None
    im = cv2.imread(path)
    return im if im is not None and float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).std()) >= min_std else None


def _thumb(im, n=48):
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (n, n), interpolation=cv2.INTER_AREA).astype(np.float64)


def same_scene(im, ref, min_corr=0.3):
    """True when `im` shows the structure of `ref` (48 px grey thumbnails correlate at least `min_corr`). The catch for a develop that is neither blank nor truncated but garbage (colour stripes, 2023-08-27_DSC04491 and 2024-06-06_DSC06876 on 2026-11-09): `ref` is the camera's own preview of the same frame. Real edits keep well above 0.3 against it; the garbage was -0.01..0.21."""
    a, b = _thumb(im), _thumb(ref)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return False
    return bool(np.corrcoef(a.ravel(), b.ravel())[0, 1] >= min_corr)


def noise_blown(out, base, factor=4.0):
    """True when `out` has far more fine detail than the picture it was made from, relative to its own contrast (Laplacian std over grey std, so a plain exposure lift is not flagged): a transform that amplified noise to static (2015-07-17_DSC04869 auto4)."""

    def rel(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_32F).std()) / max(float(g.std()), 8.0)

    return rel(out) > factor * max(rel(base), 0.02)
