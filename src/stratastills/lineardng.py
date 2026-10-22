"""Linear DNGs: a DNG that holds demosaiced RGB (PhotometricInterpretation LinearRaw) instead of a mosaic, which LibRaw cannot open. Lightroom writes them for merged panoramas and HDRs and for "Enhance" (the DJI stitched panoramas in the Hazelmere folder are Lightroom merges of this kind).
The data is read the way the DNG spec says, up to the point where develop.py takes over from LibRaw's output: black level and white level per channel -> 0..1, the opcodes of OpcodeList2 (Lightroom stores its merges in a non-linear encoding and a MapPolynomial per plane makes them linear again) -> the as-shot white balance (1 / AsShotNeutral, green = 1) -> 16 bit.
The result is what `develop._raw_u16` gives for a mosaic raw: uint16 (H, W, 3), camera RGB, white-balanced, linear. Decoding the JPEG XL tiles needs `imagecodecs`."""
import struct

import numpy as np


def _black(tag, n):
    """The BlackLevel-like tag as n floats: RATIONAL tags hold (numerator, denominator) pairs, the others plain numbers; fewer values than n are repeated."""
    v = np.asarray(tag.value, float).ravel()
    if str(getattr(tag.dtype, 'name', tag.dtype)).upper() == 'RATIONAL' or tag.dtype == 5: v = v[0::2] / np.where(v[1::2] == 0, 1, v[1::2])
    return np.resize(v, n)


def raw_page(t):
    """The biggest LinearRaw page of an open tifffile.TiffFile, or None (not a linear DNG)."""
    best = None
    for pg in [t.pages[0]] + list(getattr(t.pages[0], 'pages', []) or []):
        if pg.photometric.name == 'LINEAR_RAW' and (best is None or pg.shape[0] * pg.shape[1] > best.shape[0] * best.shape[1]): best = pg
    return best


def is_linear(path):
    import tifffile
    try:
        with tifffile.TiffFile(path) as t: return raw_page(t) is not None
    except Exception: return False


def map_polynomials(blob, n_planes=3):
    """[(plane, first row, last row, first col, last col, coefficients)] of the MapPolynomial opcodes (id 8) of an opcode list; other opcodes are not handled here (they raise, so that nothing is silently skipped)."""
    b = bytes(blob); count, = struct.unpack('>I', b[:4]); o = 4; out = []
    for _ in range(count):
        oid, _ver, _flags, nb = struct.unpack('>IIII', b[o:o + 16]); o += 16; p = b[o:o + nb]; o += nb
        if oid != 8: raise ValueError(f'linear DNG opcode {oid} is not supported')
        top, left, bottom, right, plane, planes, rp, cp, degree = struct.unpack('>9I', p[:36]); coef = struct.unpack(f'>{degree + 1}d', p[36:36 + 8 * (degree + 1)])
        if (rp, cp) != (1, 1): raise ValueError('MapPolynomial with a pitch is not supported')
        out.append((plane, planes, top, bottom, left, right, coef))
    return out


def valid_box(t, margin=8):
    """(top, bottom, left, right) in pixels of the biggest axis-aligned rectangle with data in it, from the alpha MASK pages of a merged panorama (the stitch leaves curved empty borders), found on a reduced copy and pulled in by `margin` pixels; None when the file has no mask or the rectangle would be tiny."""
    import cv2
    full = raw_page(t); masks = [p for p in [t.pages[0]] + list(getattr(t.pages[0], 'pages', []) or []) if p.photometric.name == 'MASK']
    if full is None or not masks: return None
    m = min(masks, key=lambda p: abs(p.shape[1] - 1024) if p.shape[1] >= 256 else 1e9).asarray() > 127; H, W = full.shape[:2]
    sm = cv2.resize(m.astype(np.uint8) * 255, (min(m.shape[1], 512), max(1, round(m.shape[0] * min(m.shape[1], 512) / m.shape[1]))), interpolation=cv2.INTER_AREA) > 250
    h, w = sm.shape; heights = np.zeros(w, int); best = (0, 0, 0, 0, 0)
    for r in range(h):
        heights = np.where(sm[r], heights + 1, 0); stack = []
        for c in range(w + 1):
            cur = heights[c] if c < w else 0; start = c
            while stack and stack[-1][1] >= cur:
                s0, hh = stack.pop(); area = hh * (c - s0)
                if area > best[0]: best = (area, r - hh + 1, r + 1, s0, c)
                start = s0
            stack.append((start, cur))
    area, t0, b0, l0, r0 = best
    if area < 0.1 * h * w: return None
    fy, fx = H / h, W / w; return (int(t0 * fy) + margin, int(b0 * fy) - margin, int(l0 * fx) + margin, int(r0 * fx) - margin)


def read(path, half=False, max_pixels=None):
    """uint16 (H, W, 3), camera RGB, linear, white-balanced as shot (see the module doc). `half`: a quarter-size picture (area average); `max_pixels`: the most pixels the caller can develop (the card's memory): a bigger picture is area-averaged down to fit (never by more than it must)."""
    import cv2, tifffile
    from stratastills import develop as D
    with tifffile.TiffFile(path) as t:
        pg = raw_page(t)
        if pg is None: raise ValueError('not a linear DNG')
        main = t.pages[0]; box = valid_box(t); a = pg.asarray(); black = _black(pg.tags['BlackLevel'], 3) if 'BlackLevel' in pg.tags else np.zeros(3); white = np.resize(np.asarray(pg.tags['WhiteLevel'].value, float), 3) if 'WhiteLevel' in pg.tags else np.full(3, 65535.0)
        ops = map_polynomials(pg.tags['OpcodeList2'].value) if 'OpcodeList2' in pg.tags else []; neutral = D._rationals(main.tags['AsShotNeutral'].value) if 'AsShotNeutral' in main.tags else np.ones(3)
    y0 = x0 = 0
    if box: y0, y1, x0, x1 = box; a = a[y0:y1, x0:x1]
    k = 0.5 if half else 1.0
    if max_pixels and a.shape[0] * a.shape[1] * k * k > max_pixels: k = (max_pixels / (a.shape[0] * a.shape[1])) ** 0.5
    if k < 1.0: a = cv2.resize(a, (int(a.shape[1] * k) // 2 * 2, int(a.shape[0] * k) // 2 * 2), interpolation=cv2.INTER_AREA)
    x = a.astype(np.float32); del a
    for c in range(3): x[..., c] = np.clip((x[..., c] - black[c]) / max(white[c] - black[c], 1.0), 0.0, 1.0)
    for plane, planes, top, bottom, left, right, coef in ops:
        h, w = x.shape[:2]
        top, bottom, left, right = (max(0, int((v - o) * k)) for v, o in ((top, y0), (bottom, y0), (left, x0), (right, x0)))
        for c in range(plane, min(3, plane + planes)):
            v = x[top:bottom, left:right, c]; acc = np.full_like(v, coef[-1])
            for cf in coef[-2::-1]: acc = acc * v + cf                                                                          # Horner: coef[0] + coef[1] v + coef[2] v^2 ...
            x[top:bottom, left:right, c] = np.clip(acc, 0.0, 1.0)
    mult = (1.0 / np.maximum(neutral, 1e-3)); mult = mult / mult[1]
    for c in range(3): x[..., c] = np.clip(x[..., c] * mult[c], 0.0, 1.0)
    return (x * 65535.0 + 0.5).astype(np.uint16)
