"""Developing a camera raw (DNG) into a finished-quality picture, following the DNG processing model with what the file itself says:

  1  LibRaw: black level, white balance (as shot), demosaic, highlight blend; output in camera RGB, linear, 16 bit (no colour matrix, no curve, no brightening)
  2  OpcodeList3: the lens shading GainMap (vignetting) and the per-channel WarpRectilinear (distortion and lateral colour fringing), as the maker wrote them
  3  denoise from the file's NoiseProfile (chroma strongly; luminance only when the noise is visible), in linear light
  4  colour: the camera matrices (ColorMatrix1/2 blended by the white balance's colour temperature) to XYZ, Bradford-adapted to D65, to linear sRGB
  5  exposure (BaselineExposure + automatic, so the median lands where the camera's own JPEGs put it), a highlight shoulder, and the profile's tone curve (ProfileToneCurve, which includes the gamma; plain sRGB when the file has none)

`develop(path, half=False, ev=0, denoise='auto')` returns a float32 RGB array in 0..1 (sRGB encoded), the right way up. `half=True` is a quarter-size version (LibRaw's half-size demosaic) for previews that follows the same recipe.
Files that are not DNG-with-opcodes simply skip the steps they lack. Bump VERSION when the look changes: previews cached under the old one are not reused."""
import struct
from concurrent.futures import ThreadPoolExecutor

import cv2, numpy as np

VERSION = 4
D65 = np.array([0.95047, 1.0, 1.08883]); D50 = np.array([0.96422, 1.0, 0.82521])
XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
T_A, T_D65 = 2856.0, 6504.0


class DevelopError(RuntimeError): pass


# ---- what the DNG says ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def read_dng_tags(path):
    """The DNG tags this module uses, from the raw IFD (and the main IFD): {name: value}. Missing ones are left out."""
    import tifffile
    want = ('ColorMatrix1', 'ColorMatrix2', 'CalibrationIlluminant1', 'CalibrationIlluminant2', 'AsShotNeutral', 'BaselineExposure', 'NoiseProfile', 'ProfileToneCurve', 'OpcodeList3', 'BlackLevel', 'WhiteLevel')
    out = {}
    with tifffile.TiffFile(path) as t:
        pages = [t.pages[0]] + list(getattr(t.pages[0], 'pages', []) or [])
        for pg in pages:
            for tag in pg.tags.values():
                if tag.name in want and tag.name not in out: out[tag.name] = tag.value
    return out


def _rationals(v): return np.array([a / b if b else 0.0 for a, b in zip(v[0::2], v[1::2])], float)


def parse_opcodes(b):
    """[(id, params bytes)] of a DNG opcode list (big-endian: count, then id, version, flags, byte count and parameters of each)."""
    b = bytes(b); n, = struct.unpack('>I', b[:4]); o = 4; out = []
    for _ in range(n):
        oid, _ver, _flags, nb = struct.unpack('>IIII', b[o:o + 16]); o += 16; out.append((oid, b[o:o + nb])); o += nb
    return out


def gain_map(p):
    """A GainMap opcode as (map float32 [rows, cols, planes], first plane, n planes, (top, left, bottom, right), spacing (v, h), origin (v, h)); maps are in 0..1 coordinates of the active area."""
    top, left, bottom, right, plane, planes, rp, cp, nv, nh = struct.unpack('>10I', p[:40]); sv, sh, ov, oh = struct.unpack('>4d', p[40:72]); mp, = struct.unpack('>I', p[72:76])
    m = np.frombuffer(p[76:76 + 4 * nv * nh * mp], dtype='>f4').astype(np.float32).reshape(nv, nh, mp)
    return m, plane, planes, (top, left, bottom, right), (sv, sh), (ov, oh)


def warp_params(p):
    """A WarpRectilinear opcode as (coefficients [planes, 6] (kr0..kr3, kt0, kt1), (cx, cy) normalised 0..1 of the crop)."""
    n, = struct.unpack('>I', p[:4]); d = struct.unpack('>' + 'd' * (6 * n + 2), p[4:4 + 8 * (6 * n + 2)]); return np.array(d[:6 * n]).reshape(n, 6), (d[6 * n], d[6 * n + 1])


def apply_gain_map(img, m, plane, planes):
    """Multiply `img` (H, W, 3) by the gain map, bilinearly stretched over the whole picture (the map covers the active area, which is the picture here)."""
    h, w = img.shape[:2]; field = None
    for k in range(planes):
        g = cv2.resize(m[:, :, min(k, m.shape[2] - 1)], (w, h), interpolation=cv2.INTER_LINEAR); img[:, :, plane + k] *= g
        if k == min(1, planes - 1): field = g
    return img, field


def warp_rectilinear(img, coef, centre):
    """Undo the lens: for each output pixel, where in the source picture it came from (DNG spec 'WarpRectilinear'), per colour plane (so lateral colour fringing is corrected too)."""
    h, w = img.shape[:2]; cx, cy = centre[0] * (w - 1), centre[1] * (h - 1); mr = np.hypot(max(cx, w - 1 - cx), max(cy, h - 1 - cy))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); x = (xx - cx) / mr; y = (yy - cy) / mr; r2 = x * x + y * y; out = np.empty_like(img)

    def plane(c):
        kr0, kr1, kr2, kr3, kt0, kt1 = coef[min(c, len(coef) - 1)]
        f = kr0 + r2 * (kr1 + r2 * (kr2 + r2 * kr3))
        xs = x * f + kt0 * 2 * x * y + kt1 * (r2 + 2 * x * x); ys = y * f + kt1 * 2 * x * y + kt0 * (r2 + 2 * y * y)
        out[:, :, c] = cv2.remap(np.ascontiguousarray(img[:, :, c]), (xs * mr + cx).astype(np.float32), (ys * mr + cy).astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    with ThreadPoolExecutor(img.shape[2]) as ex: list(ex.map(plane, range(img.shape[2])))                  # the planes are independent and OpenCV lets go of the GIL
    return out


# ---- colour ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _xy_to_cct(xy):
    x, y = xy; n = (x - 0.3320) / (0.1858 - y); return 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33          # McCamy


def _cct_to_xy(t):
    t = max(1667.0, min(25000.0, t))
    x = (-0.2661239e9 / t ** 3 - 0.2343589e6 / t ** 2 + 0.8776956e3 / t + 0.179910) if t <= 4000 else (-3.0258469e9 / t ** 3 + 2.1070379e6 / t ** 2 + 0.2226347e3 / t + 0.240390)
    y = (-1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683) if t <= 2222 else ((-0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867) if t <= 4000 else (3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483))
    return x, y


def color_matrix(cm1, cm2, neutral):
    """The XYZ -> camera matrix for this white balance: ColorMatrix1 (illuminant A) and ColorMatrix2 (D65) blended by inverse colour temperature, found by iterating (the temperature depends on the matrix)."""
    t = T_D65
    for _ in range(20):
        w = float(np.clip((1 / t - 1 / T_D65) / (1 / T_A - 1 / T_D65), 0, 1)); cm = w * cm1 + (1 - w) * cm2
        xyz = np.linalg.solve(cm, neutral); xy = xyz[:2] / xyz.sum(); t2 = float(np.clip(_xy_to_cct(xy), T_A - 800, 10000))
        if abs(t2 - t) < 5: break
        t = t2
    return cm


def camera_to_srgb(cm, neutral):
    """3x3 matrix taking white-balanced linear camera RGB (neutral = 1,1,1) to linear sRGB: camera -> XYZ of the scene's own white, Bradford-adapted to D65, then to sRGB."""
    white = np.linalg.solve(cm, neutral); white = white / white[1]
    to_xyz = np.linalg.inv(cm) @ np.diag(neutral)
    src, dst = BRADFORD @ white, BRADFORD @ D65; adapt = np.linalg.inv(BRADFORD) @ np.diag(dst / src) @ BRADFORD
    m = XYZ_TO_SRGB @ adapt @ to_xyz
    return m / m.sum(axis=1, keepdims=True)                                                                  # each row sums to 1: neutral stays neutral exactly


def srgb_oetf(x):
    x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def tone_lut(curve, n=4096):
    """The profile tone curve (pairs of input, output in 0..1) as a lookup table over linear input; None when the file has none."""
    if curve is None: return None
    c = np.asarray(curve, float).reshape(-1, 2); return np.interp(np.linspace(0, 1, n), c[:, 0], c[:, 1]).astype(np.float32)


# ---- denoise ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def noise_sigma(noise_profile, level):
    """The standard deviation of the noise (0..1 linear) of a pixel of linear value `level`, from the DNG NoiseProfile (scale S, offset O: variance = S * level + O)."""
    s, o = noise_profile[0], noise_profile[1]; return float(np.sqrt(max(s * level + o, 1e-12)))


def visible_noise(img, noise_profile):
    """How strong the noise is in the finished picture, in 8-bit grey levels, for `img` (linear, already exposed) whose noise follows `noise_profile` (scale S, offset O of the values as they are now): the noise at the median brightness times the slope of the sRGB curve there."""
    m = max(float(np.median(img[::8, ::8] @ np.array([0.2126, 0.7152, 0.0722], np.float32))), 1e-4)
    return 255.0 * noise_sigma(noise_profile, m) * (1.055 / 2.4) * m ** (1 / 2.4 - 1)


def _nlm(y, h):
    enc = (srgb_oetf(np.clip(y, 0, 1)) * 65535).astype(np.uint16); den = cv2.fastNlMeansDenoising(enc, h=[h * 257], templateWindowSize=5, searchWindowSize=15, normType=cv2.NORM_L1)
    return srgb_eotf(den.astype(np.float32) / 65535)


CORNER_GAIN = 2.0               # lens-shading gain from which the automatic luminance denoise starts to be blended in
CORNER_H_MAX = 3.0


def denoise_linear(img, noise_profile, strength='auto', gain=None):
    """Remove noise in linear light without taking texture with it.
    Colour noise is blurred in proportion to the noise (it carries no detail; capped, so edges keep their colour). Luminance carries the texture (roof tiles, brick, foliage), and a non-local-means pass takes that with the noise: measured on the Hazelmere roofs it halved the fine detail, so
    'auto' NEVER smooths luminance in the middle of the picture. The one place it is needed is the corners, which the lens-shading gain (`gain`, the field the picture was multiplied by) lifted several stops, noise with it: from CORNER_GAIN upwards a light pass (h <= CORNER_H_MAX grey levels)
    is blended in, in proportion to the gain. `strength` a number applies a luminance pass of that h (8-bit grey levels) everywhere; 'chroma' does the colour part only; 0 does nothing to luminance."""
    if noise_profile is None and strength == 'auto': return img
    sig8 = visible_noise(img, noise_profile) if noise_profile is not None else 0.0
    y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32); c = img - y[..., None]
    sc = float(np.clip(sig8 * 0.6, 0.0, 1.5))
    if sc > 0.3: c = cv2.GaussianBlur(c, (0, 0), sc)
    if strength not in ('auto', 'chroma'):
        if float(strength) > 0: y = _nlm(y, float(strength))
    elif strength == 'auto' and gain is not None and float(gain.max()) > CORNER_GAIN:
        gmax = float(np.percentile(gain[::8, ::8], 99.5)); h = min(CORNER_H_MAX, 0.5 * sig8 * (gmax / max(float(np.median(gain[::8, ::8])), 1.0)) ** 0.5)
        if h > 0.5:
            w = np.clip((gain - CORNER_GAIN) / max(gmax - CORNER_GAIN, 1e-3), 0, 1); y = y * (1 - w) + _nlm(y, h) * w
    return np.clip(y[..., None] + c, 0, None)


LUMA_MIN = 2.0


def srgb_eotf(v):
    v = np.clip(v, 0, 1); return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


# ---- exposure ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

TARGET_MEDIAN = 0.42            # the picture's median luminance in the finished (sRGB-encoded) picture, where the camera's own JPEGs sit
MAX_EV = 3.5
KNEE = 0.72                     # linear value above which highlights are rolled off towards white
WHITE = 4.0                     # the linear value that the shoulder maps to white


def shoulder(img, knee=KNEE, white=WHITE):
    """Roll the highlights off smoothly (extended Reinhard above `knee`, driven by the brightest channel and applied to all three so hues hold) instead of clipping them; `white` is the input that maps to 1."""
    m = img.max(axis=2, keepdims=True); span = 1.0 - knee; x = np.maximum(m - knee, 0.0) / span; w = (white - knee) / span
    new = np.where(m > knee, knee + span * x * (1 + x / (w * w)) / (1 + x), m); return img * (new / np.maximum(m, 1e-6))


def local_tone(img, strength=0.35, px=1200):
    """Recover shadows and highlights: split the log luminance into a smooth base (an edge-preserving blur on a reduced copy, so edges stay sharp and there are no halos) and the detail on top of it; squeeze the base towards its median by `strength` and put the detail back untouched.
    The camera's JPEG cannot do this (it has already clipped and crushed); from the raw the shadows under a roof and the sunlit wall beside them can both be shown. 0 = off. Colours scale with the luminance, so hues hold."""
    if strength <= 0: return img
    h, w = img.shape[:2]; y = np.maximum(img @ np.array([0.2126, 0.7152, 0.0722], np.float32), 1e-5); ly = np.log2(y)
    k = min(1.0, px / max(h, w)); small = cv2.resize(ly, (max(8, round(w * k)), max(8, round(h * k))), interpolation=cv2.INTER_AREA)
    base_s = cv2.bilateralFilter(small, 9, 0.5, max(4.0, 0.012 * max(small.shape))); base = cv2.resize(base_s, (w, h), interpolation=cv2.INTER_CUBIC)
    mid = float(np.median(base[::8, ::8])); new = mid + (base - mid) * (1 - strength) + (ly - base)
    return img * (2.0 ** (new - ly))[..., None]


def finish(img, lut):
    """Shoulder, profile tone curve and sRGB curve of linear sRGB-primaries data (0..inf): the last steps of the development."""
    img = np.clip(shoulder(np.clip(img, 0, None)), 0, 1)
    if lut is None: return srgb_oetf(img).astype(np.float32)
    return np.interp(img, np.linspace(0, 1, len(lut)), lut).astype(np.float32)           # the profile curve maps linear to display-encoded values (it includes the gamma): no sRGB curve after it


def auto_ev(img, lut, target=TARGET_MEDIAN):
    """The exposure change (stops, -1..MAX_EV) that puts the median luminance of the finished picture at `target`: found by bisection on a sample of the picture, through the whole finish."""
    s = img[::8, ::8].reshape(-1, 3); y = s @ np.array([0.2126, 0.7152, 0.0722], np.float32); lo, hi = -1.0, MAX_EV
    for _ in range(12):
        mid = (lo + hi) / 2; med = float(np.median(finish((s * 2.0 ** mid)[None], lut) @ np.array([0.2126, 0.7152, 0.0722], np.float32)))
        lo, hi = (mid, hi) if med < target else (lo, mid)
    return (lo + hi) / 2 if y.size else 0.0


# ---- the whole development ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _raw_linear(path, half):
    import rawpy
    with rawpy.imread(path) as r:
        rgb = r.postprocess(half_size=half, use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1), output_color=rawpy.ColorSpace.raw, demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT, highlight_mode=rawpy.HighlightMode.Blend, user_flip=0)
    return rgb.astype(np.float32) / 65535.0


def analysis_luma(path):
    """The luminance of the raw at full size, gamma-encoded 0..1, with nothing done to it that changes its sharpness (no lens warp, no denoise, no local tone): what detail.effective_resolution measures."""
    img = _raw_linear(path, False); y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32); med = max(float(np.median(y[::8, ::8])), 1e-5)
    return srgb_oetf(np.clip(y * (0.18 / med), 0, 1)).astype(np.float32)


def develop(path, half=False, ev='auto', denoise='auto', recover=0.35):
    tags = read_dng_tags(path); img = _raw_linear(path, half); field = None
    if 'OpcodeList3' in tags:
        for oid, p in parse_opcodes(tags['OpcodeList3']):
            if oid == 9:
                m, plane, planes, _box, _sp, _org = gain_map(p); img, field = apply_gain_map(img, m, plane, planes)
            elif oid == 1:
                coef, centre = warp_params(p); img = warp_rectilinear(img, coef, centre)
    if all(k in tags for k in ('ColorMatrix1', 'ColorMatrix2', 'AsShotNeutral')):
        n = _rationals(tags['AsShotNeutral']); cm1 = _rationals(tags['ColorMatrix1']).reshape(3, 3); cm2 = _rationals(tags['ColorMatrix2']).reshape(3, 3)
        img = img @ camera_to_srgb(color_matrix(cm1, cm2, n), n).T.astype(np.float32)
    be = float(_rationals(tags['BaselineExposure'])[0]) if 'BaselineExposure' in tags else 0.0
    img = local_tone(np.clip(img, 0, None) * 2.0 ** be, recover); lut = tone_lut(tags.get('ProfileToneCurve'))
    g = 2.0 ** (auto_ev(img, lut) if ev == 'auto' else float(ev)); img = img * g
    if denoise != 'off':
        npf = tags.get('NoiseProfile'); npf = (npf[0] * g * 2.0 ** be, npf[1] * (g * 2.0 ** be) ** 2) if npf is not None else None          # the profile is for the raw values: after the gain, variance = S*g*v + O*g^2
        img = denoise_linear(img, npf, denoise, field)
    return finish(img, lut)
