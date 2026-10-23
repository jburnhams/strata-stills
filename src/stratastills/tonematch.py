"""Matching a raw's tone to the camera's own JPEG of it, picture by picture (the develop stage's `exposure: match`).
A camera's JPEG is not one fixed curve of the sensor values: Sony's Dynamic Range Optimizer, Canon's Auto Lighting Optimizer and the like lift the shadows of each scene by its own histogram, and the raw carries none of that. Every raw also carries a small camera JPEG (its preview), so the tone can be taken from it: the raw is developed with no tone curve, and the curve that carries the luminance of that picture onto the luminance of the preview is found by matching the two distributions (quantile by quantile). The method is RawTherapee's "auto-matched tone curve" (histogram matching on the embedded thumbnail); this is written from that description, not ported.
`curve_for(path)` gives {'curve': pairs (linear input, display-encoded output) as a DNG ProfileToneCurve, 'chroma': pairs (display luminance, colour scale)}, which develop(tone=) takes in place of the device's tone curve; it is a plain list of numbers (bump develop.VERSION when this module's result changes: the key of a generated picture follows it), so a pipeline can keep it with the picture (the sidecar) and adjust it later. None when the file has no usable preview or the two do not agree on what the picture is."""
import cv2, numpy as np

PROXY = 384                                    # long side of the pictures compared: area-averaged, so sharpness, noise and lens geometry do not count
BINS = 48
CHROMA_BINS = 12
MIN_PIXELS = 2000
MIN_AGREEMENT = 0.9                            # rank correlation of the two lumas below this: not the same picture (or not the same way up): no curve
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
MIN_PREVIEW = 19200                            # pixels; smaller previews (128x96 ones) are too coarse to match


def embedded_jpeg(path):
    """The camera JPEG inside a raw (uint8 RGB), or None: what LibRaw can extract (every raw format it reads has one, bar a few)."""
    import rawpy
    try:
        with rawpy.imread(path) as r: t = r.extract_thumb()
        if t.format == rawpy.ThumbFormat.JPEG: j = cv2.imdecode(np.frombuffer(t.data, np.uint8), cv2.IMREAD_COLOR); return None if j is None else np.ascontiguousarray(j[..., ::-1])
        if t.format == rawpy.ThumbFormat.BITMAP: return np.ascontiguousarray(t.data)
    except Exception: pass
    return None


def _proxy(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _crop_to_aspect(img, aspect):
    h, w = img.shape[:2]
    if abs(w / h - aspect) / aspect < 0.01: return img
    if w / h > aspect: nw = int(round(h * aspect)); x = (w - nw) // 2; return img[:, x:x + nw]
    nh = int(round(w / aspect)); y = (h - nh) // 2; return img[y:y + nh]


def _rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(np.float32), np.argsort(np.argsort(b)).astype(np.float32); return float(np.corrcoef(ra, rb)[0, 1])


def align(linear, jpeg):
    """(the developed picture, the preview's encoded RGB 0..1) at the same small size and the same way up, or None. The preview may be rotated against the developed raw (EXIF orientation) and may differ in the edges it shows, so it is turned the way that agrees best (by the rank correlation of the luminances) and cropped to the aspect of the raw."""
    from stratastills import develop
    h, w = linear.shape[:2]; best = None
    for k in (0, 1, 2, 3):
        j = np.rot90(jpeg, k); jh, jw = j.shape[:2]
        if abs((jw / jh) / (w / h) - 1) > 0.15: continue                                 # the aspect must be about that of the raw, whatever the crop of the preview
        j = _crop_to_aspect(np.ascontiguousarray(j), w / h); s = PROXY / max(w, h); size = (max(int(w * s), 8), max(int(h * s), 8))
        a = _proxy(linear, size); b = _proxy(j.astype(np.float32) / 255.0, size); c = _rank_corr((a @ LUMA).ravel()[::7], (develop.srgb_eotf(b) @ LUMA).ravel()[::7])
        if best is None or c > best[0]: best = (c, a, b)
    return None if best is None or best[0] < MIN_AGREEMENT else best[1:]


def fit(x, y, bins=BINS):
    """Knots of the curve that takes values with the distribution of `x` (linear luminance) to that of `y` (the preview's, linear): the medians of `x` and of the display-encoded `y` in `bins` slices of equal count of the pixels sorted by `x`; forced to rise, with (0, 0) and (1, top) added."""
    o = np.argsort(x); x, y = x[o], y[o]; cut = np.linspace(0, len(x), bins + 1).astype(int)
    kx = np.array([np.median(x[a:b]) for a, b in zip(cut[:-1], cut[1:])]); ky = np.array([np.median(y[a:b]) for a, b in zip(cut[:-1], cut[1:])])
    ky = np.where(ky <= 0.0031308, 12.92 * ky, 1.055 * np.maximum(ky, 1e-6) ** (1 / 2.4) - 0.055); ky = np.maximum.accumulate(ky)
    keep = np.concatenate([[True], np.diff(kx) > 1e-6]); kx, ky = kx[keep], ky[keep]
    top = min(1.0, ky[-1] + (1.0 - kx[-1]) * (ky[-1] - ky[-2]) / max(kx[-1] - kx[-2], 1e-6)) if len(kx) > 1 else 1.0
    return np.concatenate([[0.0], kx, [1.0]]), np.concatenate([[0.0], ky, [max(top, ky[-1])]])


def match_curve(linear, jpeg):
    """What matches the developed `linear` picture (float32 (H, W, 3), no tone curve applied) to the camera `jpeg` (uint8 RGB): {'curve': flat pairs (linear luminance, display-encoded output), 'chroma': flat pairs (display luminance, scale of the colour)}, or None.
    The curve is the luminance match. The colour of a curve applied to each channel is not the camera's (a strong shadow lift makes shadows far more colourful than Sony's DRO does), so the chroma scale is measured too: in slices of display luminance, the colour of the preview over the colour of our picture after the curve."""
    from stratastills import develop
    if jpeg is None or jpeg.shape[0] * jpeg.shape[1] < MIN_PREVIEW: return None
    al = align(linear, jpeg)
    if al is None: return None
    A, J = al; a = A @ LUMA; b = develop.srgb_eotf(J) @ LUMA; ok = (a > 1e-4) & (b > 3e-4) & (b < 0.97) & (A.max(axis=2) < 0.97)
    if ok.sum() < MIN_PIXELS: return None
    kx, ky = fit(a[ok], b[ok]); curve = [round(float(v), 5) for p in zip(kx, ky) for v in p]
    ours = develop.finish(A, develop.tone_lut(curve)); yo = ours @ LUMA; yj = J @ LUMA; co = np.linalg.norm(ours - yo[..., None], axis=2); cj = np.linalg.norm(J - yj[..., None], axis=2)
    xs, ss = [0.0], [1.0]
    for lo in np.linspace(0, 1, CHROMA_BINS + 1)[:-1]:
        m = ok & (yo >= lo) & (yo < lo + 1 / CHROMA_BINS) & (co > 0.01)
        if m.sum() >= 200: xs.append(lo + 0.5 / CHROMA_BINS); ss.append(float(np.clip(np.median(cj[m]) / np.median(co[m]), 0.2, 1.5)))
    xs.append(1.0); ss.append(ss[-1]); ss[0] = ss[1] if len(ss) > 2 else 1.0
    return dict(curve=curve, chroma=[round(float(v), 4) for p in zip(xs, ss) for v in p])


def curve_for(path, recover=0.35, lens='auto'):
    """`match_curve` for a raw file: developed at half size with the colour and local tone of the normal recipe and no tone curve, against the preview in the file. None when there is no usable preview."""
    from stratastills import develop
    j = embedded_jpeg(path)
    if j is None: return None
    return match_curve(develop.develop_parts(path, half=True, ev=0.0, recover=recover, lens=lens)['linear'], j)
