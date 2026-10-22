"""Choosing the JPEG quality automatically: the lowest quality whose compression error cannot be seen, which depends on the picture.

JPEG error is invisible when it is small compared with the noise the picture already has: grain hides it. A clean picture (or a smooth sky) needs a high quality; a grainy one does not. Measured on a 27 MP Mini 4 Pro export (docs/progress.md): the noise in its smooth areas is 1.85 grey levels, and
JPEG at quality 95 adds 1.74 of error (as much as the noise itself, so blocking can show in a sky), 98 adds 0.79, 99 adds 0.47, 100 adds 0.30 for 36 MB against 24 MB at 98. SSIM was no use as a target: it falls at q=95 on any textured picture.

The rule (`choose_quality`): take TILES tiles of the picture that will really be encoded (the most textured ones, where blocking and ringing show on edges, and the smoothest ones, where banding and blocks show), encode each at a trial quality (4:4:4, as the export does) and measure the RMS error of the luminance, in 8-bit grey levels.
The target is `FRACTION` x the picture's own noise sigma (found from its smooth tiles by the Haar-wavelet MAD estimate), but never below FLOOR (a noiseless picture must not ask for quality 100 at 1.5x the size for nothing) or above CEIL. A binary search over MIN_Q..MAX_Q finds the lowest quality that meets it.
Tiles start on multiples of 16, so a tile is encoded exactly as it will be inside the whole picture (JPEG works in 8x8 / 16x16 blocks)."""
import io

import cv2, numpy as np

TILE = 512
TEXTURED, SMOOTH = 16, 8
FRACTION = 0.45                 # the JPEG's error may be at most this fraction of the picture's noise sigma (that is about a fifth of the noise variance added: not noticeable)
FLOOR, CEIL = 0.45, 1.5         # the target in grey levels, whatever the noise
MIN_Q, MAX_Q = 88, 99


def _encode(a, q):
    from PIL import Image, ImageFile
    ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, 1 << 27)                                 # see export.write_picture
    b = io.BytesIO(); Image.fromarray(a).save(b, 'JPEG', quality=int(q), subsampling=0, optimize=True); return b.getvalue()


def _decode(buf):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(buf)).convert('RGB'))


def noise_sigma(rgb):
    """The noise in a tile, in 8-bit grey levels: median absolute value of the Haar diagonal detail of its luminance / 0.6745 (Donoho). Fine texture counts as noise, so it is only taken on smooth tiles."""
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32); h, w = (y.shape[0] // 2) * 2, (y.shape[1] // 2) * 2; y = y[:h, :w]
    d = (y[0::2, 0::2] - y[0::2, 1::2] - y[1::2, 0::2] + y[1::2, 1::2]) / 2
    return float(np.median(np.abs(d)) / 0.6745)


def pick_tiles(rgb, tile=TILE):
    """[(y, x, size)]: the most textured and the smoothest tiles of the picture (origins on multiples of 16), by the mean absolute Laplacian of the green channel. A picture smaller than a tile is one tile."""
    h, w = rgb.shape[:2]; t = tile if min(h, w) >= tile else (min(h, w) // 16) * 16
    if t < 32: return [(0, 0, 0)]
    g = rgb[..., 1].astype(np.float32); c = []
    for y in range(0, h - t + 1, t):
        for x in range(0, w - t + 1, t): c.append((float(np.abs(cv2.Laplacian(g[y:y + t, x:x + t], cv2.CV_32F)).mean()), y, x))
    c.sort(); chosen = {(y, x) for _, y, x in c[-TEXTURED:]} | {(y, x) for _, y, x in c[:SMOOTH]}
    return [(y, x, t) for y, x in sorted(chosen)], [(y, x, t) for _, y, x in c[:SMOOTH]]


def _rms(rgb, tiles, q):
    e = []
    for y, x, t in tiles:
        a = rgb[y:y + t, x:x + t]; b = _decode(_encode(a, q)); la = cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32); lb = cv2.cvtColor(b, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
        e.append(float(np.sqrt(((la - lb) ** 2).mean())))
    return float(np.mean(e))


def choose_quality(rgb):
    """(quality, info) for the 8-bit RGB array `rgb` that is about to be written as a JPEG: info = {sigma, target, rms, tiles} (rms: the luminance error at the quality chosen, grey levels)."""
    if rgb.shape[0] < 32 or rgb.shape[1] < 32: return MAX_Q, dict(sigma=None, target=None, rms=None, tiles=0, note='too small to measure')
    tiles, smooth = pick_tiles(rgb)
    sigma = float(np.median([noise_sigma(rgb[y:y + t, x:x + t]) for y, x, t in smooth])); target = float(np.clip(FRACTION * sigma, FLOOR, CEIL))
    lo, hi = MIN_Q, MAX_Q
    while lo < hi:                                                                                  # the lowest quality with error <= target (error falls as quality rises)
        mid = (lo + hi) // 2
        if _rms(rgb, tiles, mid) <= target: hi = mid
        else: lo = mid + 1
    return lo, dict(sigma=round(sigma, 2), target=round(target, 2), rms=round(_rms(rgb, tiles, lo), 2), tiles=len(tiles))
