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

from stratastills import denoise_ai

VERSION = 6
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

TARGET_MEDIAN = 0.40            # the picture's median luminance in the finished (sRGB-encoded) picture, where the camera's own JPEGs sit
MAX_EV = 3.5
KNEE = 0.55                     # linear value above which highlights are rolled off towards white
WHITE = 16.0                     # the linear value that the shoulder maps to white


def shoulder(img, knee=None, white=None):
    """Roll the highlights off smoothly (extended Reinhard above `knee`, driven by the brightest channel and applied to all three so hues hold) instead of clipping them; `white` is the input that maps to 1."""
    knee = KNEE if knee is None else knee; white = WHITE if white is None else white; m = img.max(axis=2, keepdims=True); span = 1.0 - knee; x = np.maximum(m - knee, 0.0) / span; w = (white - knee) / span
    new = np.where(m > knee, knee + span * x * (1 + x / (w * w)) / (1 + x), m); return img * (new / np.maximum(m, 1e-6))


# ---- colour noise and fringes ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _guided(p, guide, r, eps):
    """He et al.'s guided filter (box-filter form, one channel): smooths `p` but keeps the edges of `guide`. r: radius in pixels, eps: how much contrast (in guide units squared) still counts as an edge."""
    k = (2 * r + 1, 2 * r + 1); mean = lambda a: cv2.boxFilter(a, -1, k, borderType=cv2.BORDER_REFLECT)
    mg, mp = mean(guide), mean(p); cov = mean(guide * p) - mg * mp; var = mean(guide * guide) - mg * mg; a = cov / (var + eps); b = mp - a * mg
    return mean(a) * guide + mean(b)


def clean_colour(rgb, sig8, desat_from=0.86):
    """What the camera's own JPEG does and a raw development must do by hand: keep colour noise, colour fringes and clipped-highlight colour out of the picture.
    The picture is split into luminance Y and two colour differences (B-Y, R-Y). The colour differences are smoothed by a guided filter that follows the edges of Y (so real colour edges stay where the luminance edge is and nothing bleeds), at a radius that grows with the noise there
    (`sig8`, per pixel, 8-bit grey levels: 2 px when it is low, up to 14 px where it is high: the dark, shading-lifted corners), then cored (the threshold is a third of the raw noise: the filter has already averaged most of it away): colour weaker than that goes to grey, so dark roof tiles are grey as they are, not speckled in every colour.
    Near white the colour is rolled off (from luminance `desat_from`), because a clipped channel there gives purple and blue fringes. Luminance is not touched here."""
    y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32); cb, cr = rgb[..., 2] - y, rgb[..., 0] - y; sg = sig8 / 255.0
    w = [np.clip((sg * 255 - lo) / (hi - lo), 0, 1) for lo, hi in ((2.0, 6.0), (5.0, 11.0))]                  # how much of the middle and the widest filter to use
    out = []
    for c in (cb, cr):
        c1 = _guided(c, y, 2, 4e-4); c2 = _guided(c, y, 6, 1.5e-3); c3 = _guided(c, y, 14, 4e-3)
        f = c1 * (1 - w[0]) + (c2 * (1 - w[1]) + c3 * w[1]) * w[0]
        tau = 0.3 * sg; out.append(np.sign(f) * np.maximum(np.abs(f) - tau, 0))
    roll = 1.0 - np.clip((y - desat_from) / (1.0 - desat_from), 0, 1) ** 2 * 0.9
    cb, cr = out[0] * roll, out[1] * roll; r = y + cr; b = y + cb; g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.clip(np.dstack([r, g, b]), 0, 1).astype(np.float32)


def highlight_detail(rgb, amount=1.0, lo=0.62, hi=0.92):
    """More contrast in the bright areas, without changing how bright they are: the luminance is split into a smooth edge-preserving base (guided filter) and the detail on top of it, at two scales (about 2 and 8 px); where the base is bright (from `lo`, fully by `hi`) the detail is multiplied by up to 1 + `amount`, and the base is put back unchanged.
    The white level of a wall stays where it is; the window frames, mouldings, brick lines and render texture inside it stand out more. A pixel that would pass white is held at white."""
    y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32); b2 = _guided(y, y, 2, 2e-4); b8 = _guided(y, y, 8, 1e-3)
    w = np.clip((b8 - lo) / (hi - lo), 0, 1); w = w * w * (3 - 2 * w); f = 1 + amount * w
    new = b8 + (b2 - b8) * f + (y - b2) * f
    return np.clip(rgb + (new - y)[..., None], 0, 1).astype(np.float32)


def core_luma(rgb, sig8, k=0.6):
    """Take the grain out of the finest detail without taking the texture: luminance is split into a smooth part and a fine band (Y minus its 1 px blur); fine detail weaker than `k` x the noise is shrunk to nothing (soft threshold) and stronger detail (the edge of a tile, a brick line) is kept as it is.
    Unlike a non-local-means pass, which averages texture away, strong structure passes through unchanged."""
    y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32); low = cv2.GaussianBlur(y, (0, 0), 1.0); d = y - low; tau = k * sig8 / 255.0
    dn = np.sign(d) * np.maximum(np.abs(d) - tau, 0); return np.clip(rgb + (low + dn - y)[..., None], 0, 1).astype(np.float32)


def noise_map8(img, noise_profile, blur=8.0):
    """The noise expected at each pixel of the finished picture, in 8-bit grey levels, from the file's NoiseProfile: sigma of the linear value (variance = S v + O, with the profile already scaled for the exposure gain) times the slope of the sRGB curve there, on a smoothed luminance (the noise level follows the brightness, not the texture)."""
    h, w = img.shape[:2]; k = 1.0 / 8; y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    v = cv2.GaussianBlur(cv2.resize(y, (max(8, round(w * k)), max(8, round(h * k))), interpolation=cv2.INTER_AREA), (0, 0), blur / 8 * 2); v = np.maximum(v, 1e-5)
    slope = np.where(v < 0.0031308, 12.92, (1.055 / 2.4) * v ** (1 / 2.4 - 1)); s8 = 255.0 * np.sqrt(noise_profile[0] * v + noise_profile[1]) * slope
    return cv2.resize(s8.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def local_tone(img, strength=0.35, px=1200, highlights=None):
    """Recover shadows and highlights: split the log luminance into a smooth base (an edge-preserving blur on a reduced copy, so edges stay sharp and there are no halos) and the detail on top of it; squeeze the base towards its median by `strength` and put the detail back untouched.
    The camera's JPEG cannot do this (it has already clipped and crushed); from the raw the shadows under a roof and the sunlit wall beside them can both be shown. 0 = off. Colours scale with the luminance, so hues hold."""
    if strength <= 0: return img
    h, w = img.shape[:2]; y = np.maximum(img @ np.array([0.2126, 0.7152, 0.0722], np.float32), 1e-5); ly = np.log2(y)
    k = min(1.0, px / max(h, w)); small = cv2.resize(ly, (max(8, round(w * k)), max(8, round(h * k))), interpolation=cv2.INTER_AREA)
    base_s = cv2.bilateralFilter(small, 9, 0.5, max(4.0, 0.012 * max(small.shape))); base = cv2.resize(base_s, (w, h), interpolation=cv2.INTER_CUBIC)
    mid = float(np.median(base[::8, ::8])); hl = HIGHLIGHTS if highlights is None else highlights; k = np.where(base > mid, 1 - hl * min(1.0, strength / 0.35), 1 - strength); new = mid + (base - mid) * k + (ly - base)
    return img * (2.0 ** (new - ly))[..., None]


def finish(img, lut):
    """Shoulder, profile tone curve and sRGB curve of linear sRGB-primaries data (0..inf): the last steps of the development."""
    img = np.clip(shoulder(np.clip(img, 0, None)), 0, 1)
    if lut is None: return srgb_oetf(img).astype(np.float32)
    return np.interp(img, np.linspace(0, 1, len(lut)), lut).astype(np.float32)           # the profile curve maps linear to display-encoded values (it includes the gamma): no sRGB curve after it


HIGHLIGHTS = 0.55               # how hard local_tone squeezes bright regions towards the median (0..1)
HIGHLIGHT_PCT = 99.5
HIGHLIGHT_TARGET = 1.0          # the linear value the 99.5th percentile may reach after exposure (the shoulder and the tone curve take it the rest of the way to white)
HIGHLIGHT_CONTRAST = 0.8        # highlight_detail: how much more contrast the bright areas get (0 = none)
MAX_DARKER = 1.0                # protecting highlights may take the exposure at most this many stops below the median target (the shadows are lifted by the local tone recovery instead)


def auto_ev(img, lut, target=TARGET_MEDIAN):
    """The exposure change (stops, -1..MAX_EV): the one that puts the median luminance of the finished picture at `target` (found by bisection on a sample, through the whole finish), but no brighter than lets the brightest 0.5 % of the picture stay below white: white walls and windows keep their tone
    (a median-only rule pushed them into a flat plateau, while the camera's JPEG kept them). Never more than MAX_DARKER stops below the median rule."""
    s = img[::8, ::8].reshape(-1, 3); w = np.array([0.2126, 0.7152, 0.0722], np.float32); y = s @ w; lo, hi = -1.0, MAX_EV
    for _ in range(12):
        mid = (lo + hi) / 2; med = float(np.median(finish((s * 2.0 ** mid)[None], lut) @ w))
        lo, hi = (mid, hi) if med < target else (lo, mid)
    ev_med = (lo + hi) / 2 if y.size else 0.0
    if not y.size: return ev_med
    ev_hl = float(np.log2(HIGHLIGHT_TARGET / max(float(np.percentile(y, HIGHLIGHT_PCT)), 1e-4)))
    return float(np.clip(min(ev_med, max(ev_hl, ev_med - MAX_DARKER)), -1.0, MAX_EV))


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


def develop_parts(path, half=False, ev='auto', recover=0.35):
    """The development up to, but not including, the final tone curve: {linear: exposed linear sRGB-primaries picture (float32), lut, noise: the NoiseProfile scaled for the exposure gain (or None), gain: the lens-shading field (or None), gain_ev}. `develop` finishes it; experiments and the export can look inside."""
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
    npf = tags.get('NoiseProfile'); npf = (npf[0] * g * 2.0 ** be, npf[1] * (g * 2.0 ** be) ** 2) if npf is not None else None          # the profile is for the raw values: after the gain, variance = S*g*v + O*g^2
    return dict(linear=img, lut=lut, noise=npf, gain=field, gain_ev=float(np.log2(g)))


def develop(path, half=False, ev='auto', denoise='auto', recover=0.35, model=None, strength=None, highlight_contrast=HIGHLIGHT_CONTRAST):
    """The finished picture (float32 RGB, 0..1, sRGB encoded). `denoise`:
      'auto'    the classical clean-up (default): colour noise and fringes removed edge-aware and cored to grey, colour rolled off near white, the finest luminance grain cored (clean_colour, core_luma), the light corner pass on luminance where the lens-shading gain lifted the noise
      'ai'      'auto' plus a learned model (`model`, `strength`: denoise_ai) where the noise is visible
      'chroma'  colour only; 'off' nothing; a number: non-local-means h (8-bit grey levels) everywhere."""
    P = develop_parts(path, half, ev, recover); img, lut, npf, field = P['linear'], P['lut'], P['noise'], P['gain']
    classical = denoise in ('auto', 'ai', 'chroma') and npf is not None
    if denoise in ('auto', 'ai'): img = denoise_linear(img, npf, 'auto', field)                                                           # the corner luminance pass (only where the shading gain is high)
    elif denoise not in ('off', 'chroma'): img = denoise_linear(img, npf, denoise, field)
    sig8 = noise_map8(img, npf) if npf is not None else None
    out = finish(img, lut)
    if highlight_contrast > 0: out = highlight_detail(out, highlight_contrast)
    if classical:
        out = clean_colour(out, sig8)
        if denoise in ('auto', 'ai'): out = core_luma(out, sig8)
    if denoise == 'ai' and sig8 is not None:
        model = model or denoise_ai.DEFAULT; strength = denoise_ai.DEFAULT_STRENGTH if strength is None else strength
        if denoise_ai.available(model): out = denoise_ai.adaptive(out, sig8, model, strength)
    return out
