"""Developing a camera raw (DNG) into a finished-quality picture, following the DNG processing model with what the file itself says:

  1  LibRaw: black level, white balance (as shot), demosaic, highlight blend; output in camera RGB, linear, 16 bit (no colour matrix, no curve, no brightening)
  2  OpcodeList3: the lens shading GainMap (vignetting) and the per-channel WarpRectilinear (distortion and lateral colour fringing), as the maker wrote them
  3  denoise from the file's NoiseProfile (chroma strongly; luminance only when the noise is visible), in linear light
  4  colour: the camera matrices (ColorMatrix1/2 blended by the white balance's colour temperature) to XYZ, Bradford-adapted to ProPhoto (D50), the profile's own hue/saturation map (ProfileHueSatMap) there, then to linear sRGB
  5  exposure (BaselineExposure + automatic, so the median lands where the camera's own JPEGs put it), a highlight shoulder, and the profile's tone curve (ProfileToneCurve, which includes the gamma; plain sRGB when the file has none), a roll-off of the very strongest colours (limit_chroma)

`develop(path, half=False, ev=0, denoise='auto')` returns a float32 RGB array in 0..1 (sRGB encoded), the right way up. `half=True` is a quarter-size version (LibRaw's half-size demosaic) for previews that follows the same recipe.
Files that are not DNG-with-opcodes simply skip the steps they lack. Bump VERSION when the look changes: previews cached under the old one are not reused."""
import struct
from concurrent.futures import ThreadPoolExecutor

import cv2, numpy as np

from stratastills import cpu_tiled, denoise_ai, gpu, gpu_tiled, inputs, lineardng, resources

VERSION = 9
D65 = np.array([0.95047, 1.0, 1.08883]); D50 = np.array([0.96422, 1.0, 0.82521])
XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
T_A, T_D65 = 2856.0, 6504.0


class DevelopError(RuntimeError): pass


# ---- what the DNG says ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def read_dng_tags(path):
    """The DNG tags this module uses, from the raw IFD (and the main IFD): {name: value}. Missing ones are left out."""
    import tifffile
    want = ('ColorMatrix1', 'ColorMatrix2', 'CalibrationIlluminant1', 'CalibrationIlluminant2', 'AsShotNeutral', 'BaselineExposure', 'NoiseProfile', 'ProfileToneCurve', 'OpcodeList3', 'BlackLevel', 'WhiteLevel', 'ProfileHueSatMapDims', 'ProfileHueSatMapData1', 'ProfileHueSatMapData2', 'ProfileHueSatMapEncoding', 'Make', 'Model', 'ProfileName')
    out = {}
    with tifffile.TiffFile(path) as t:
        pages = [t.pages[0]] + list(getattr(t.pages[0], 'pages', []) or [])
        for pg in pages:
            for tag in pg.tags.values():
                if tag.name in want and tag.name not in out: out[tag.name] = tag.value
    return inputs.adjust_tags(out)


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


def color_matrix_weight(cm1, cm2, neutral):
    """(the XYZ -> camera matrix for this white balance, the weight of illuminant 1): ColorMatrix1 (illuminant A) and ColorMatrix2 (D65) blended by inverse colour temperature, found by iterating (the temperature depends on the matrix). The same weight blends the two profile tables."""
    t = T_D65
    for _ in range(20):
        w = float(np.clip((1 / t - 1 / T_D65) / (1 / T_A - 1 / T_D65), 0, 1)); cm = w * cm1 + (1 - w) * cm2
        xyz = np.linalg.solve(cm, neutral); xy = xyz[:2] / xyz.sum(); t2 = float(np.clip(_xy_to_cct(xy), T_A - 800, 10000))
        if abs(t2 - t) < 5: break
        t = t2
    return cm, w


def color_matrix(cm1, cm2, neutral): return color_matrix_weight(cm1, cm2, neutral)[0]


# ---- the camera's own colour profile: the hue / saturation map ---------------------------------------------------------------------------------------------------------------------------------------------------------

D50_WHITE = np.array([0.96422, 1.0, 0.82521])
XYZ_TO_PROPHOTO = np.array([[1.3459433, -0.2556075, -0.0511118], [-0.5445989, 1.5081673, 0.0205351], [0.0, 0.0, 1.2118128]])        # D50 XYZ to linear ProPhoto (ROMM) RGB: the space the DNG profile maps are defined in
HUE_SAT_MAP = True              # apply the profile's ProfileHueSatMap (the camera maker's colour rendering) when the file has one


def _bradford(src_white, dst_white):
    src, dst = BRADFORD @ src_white, BRADFORD @ dst_white; return np.linalg.inv(BRADFORD) @ np.diag(dst / src) @ BRADFORD


def camera_to_prophoto(cm, neutral):
    """3x3 matrix taking white-balanced linear camera RGB to linear ProPhoto RGB (D50): camera -> XYZ of the scene's own white -> Bradford to D50 -> ProPhoto. The scene's white lands on (1, 1, 1)."""
    white = np.linalg.solve(cm, neutral); white = white / white[1]; to_xyz = np.linalg.inv(cm) @ np.diag(neutral)
    m = XYZ_TO_PROPHOTO @ _bradford(white, D50_WHITE) @ to_xyz; return m / m.sum(axis=1, keepdims=True)


def prophoto_to_srgb():
    """3x3 matrix from linear ProPhoto (D50) to linear sRGB (D65). With camera_to_prophoto it composes to camera_to_srgb (Bradford adaptations compose exactly)."""
    m = XYZ_TO_SRGB @ _bradford(D50_WHITE, D65) @ np.linalg.inv(XYZ_TO_PROPHOTO); return m / m.sum(axis=1, keepdims=True)


def hue_sat_table(tags, weight):
    """The profile's hue/saturation map as float32 (hue divisions, saturation divisions, 3: hue shift in degrees, saturation scale, value scale), the two illuminants' tables blended by `weight` (that of illuminant 1); None when the file has no map or one with value divisions (not used by these cameras)."""
    dims = tags.get('ProfileHueSatMapDims'); d1 = tags.get('ProfileHueSatMapData1')
    if dims is None or d1 is None or int(dims[2]) != 1: return None
    h, s = int(dims[0]), int(dims[1]); a = np.asarray(d1, np.float32).reshape(h, s, 3); d2 = tags.get('ProfileHueSatMapData2')
    if d2 is None: return a
    return (weight * a + (1 - weight) * np.asarray(d2, np.float32).reshape(h, s, 3)).astype(np.float32)


def rgb_to_hsv(rgb):
    """HSV of linear RGB (..., 3) as the DNG SDK does it: v = max, s = (max - min) / max, hue 0..6 (sextants). Negative channels are clipped to 0."""
    rgb = np.maximum(rgb, 0); r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]; mx = rgb.max(-1); mn = rgb.min(-1); d = mx - mn; ds = np.where(d > 0, d, 1.0)
    h = np.where(mx == r, ((g - b) / ds) % 6.0, np.where(mx == g, (b - r) / ds + 2.0, (r - g) / ds + 4.0)); h = np.where(d > 0, h, 0.0); s = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0), 0.0)
    return np.stack([h, s, mx], -1)


def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0] % 6.0, hsv[..., 1], hsv[..., 2]; c = v * s; x = c * (1 - np.abs(h % 2.0 - 1)); m = v - c; k = np.floor(h).astype(int) % 6
    r = np.choose(k, [c, x, 0 * c, 0 * c, x, c]); g = np.choose(k, [x, c, c, x, 0 * c, 0 * c]); b = np.choose(k, [0 * c, 0 * c, x, c, c, x]); return np.stack([r + m, g + m, b + m], -1)


def apply_hue_sat_map(rgb, table, rows=256):
    """The DNG ProfileHueSatMap applied to linear ProPhoto `rgb` (H, W, 3): each pixel's hue and saturation index the table (hue wraps, saturation 0 and 1 are the first and last columns), the four neighbours are blended bilinearly into a (hue shift, saturation scale, value scale), and the pixel is shifted, scaled and converted back.
    Done in bands of `rows` rows so the index arrays stay small."""
    hd, sd = table.shape[:2]; out = np.empty_like(rgb, dtype=np.float32)
    for r0 in range(0, rgb.shape[0], rows):
        hsv = rgb_to_hsv(rgb[r0:r0 + rows]); hs = hsv[..., 0] * (hd / 6.0); i0 = np.floor(hs).astype(int) % hd; fh = hs - np.floor(hs); i1 = (i0 + 1) % hd
        ss = np.clip(hsv[..., 1], 0, 1) * (sd - 1); j0 = np.minimum(np.floor(ss).astype(int), sd - 2); fs = ss - j0; j1 = j0 + 1
        d = (table[i0, j0] * ((1 - fh) * (1 - fs))[..., None] + table[i1, j0] * (fh * (1 - fs))[..., None] + table[i0, j1] * ((1 - fh) * fs)[..., None] + table[i1, j1] * (fh * fs)[..., None])
        hsv[..., 0] = (hsv[..., 0] + d[..., 0] * (6.0 / 360.0)) % 6.0; hsv[..., 1] = np.minimum(hsv[..., 1] * d[..., 1], 1.0); hsv[..., 2] = hsv[..., 2] * d[..., 2]; out[r0:r0 + rows] = hsv_to_rgb(hsv)
    return out


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


def visible_noise(img, noise_profile, stride=8):
    """How strong the noise is in the finished picture, in 8-bit grey levels, for `img` (linear, already exposed) whose noise follows `noise_profile` (scale S, offset O of the values as they are now): the noise at the median brightness times the slope of the sRGB curve there."""
    m = max(float(np.median(img[::stride, ::stride] @ np.array([0.2126, 0.7152, 0.0722], np.float32))), 1e-4)
    return 255.0 * noise_sigma(noise_profile, m) * (1.055 / 2.4) * m ** (1 / 2.4 - 1)


def _nlm(y, h):
    enc = (srgb_oetf(np.clip(y, 0, 1)) * 65535).astype(np.uint16); den = cv2.fastNlMeansDenoising(enc, h=[h * 257], templateWindowSize=5, searchWindowSize=15, normType=cv2.NORM_L1)
    return srgb_eotf(den.astype(np.float32) / 65535)


def denoise_linear(img, noise_profile, strength='auto', gain=None):
    """Remove noise in linear light without taking texture with it.
    Colour noise is blurred in proportion to the noise (it carries no detail; capped, so edges keep their colour). LUMINANCE carries the texture (roof tiles, brick, foliage), and a non-local-means pass takes that with the noise (measured on the Hazelmere roofs it halved the fine detail), so 'auto' and 'chroma' leave it to the band coring (`core_luma`),
    which is driven by the noise map. The corners, which the lens-shading gain lifted several stops and the noise with it, used to get a light non-local-means pass here; the noise map now includes that gain (`noise_map8(gain=)`), so the coring treats the corners harder by itself, which measured cleaner (bottom corner grain 2.96 against 3.99 grey levels, roof structure kept) and 2.6 times faster.
    `strength` a number applies a non-local-means luminance pass of that h (8-bit grey levels) everywhere; `gain` is accepted for the callers that have it and is not used here."""
    if noise_profile is None and strength == 'auto': return img
    sig8 = visible_noise(img, noise_profile) if noise_profile is not None else 0.0
    y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32); c = img - y[..., None]
    sc = float(np.clip(sig8 * 0.6, 0.0, 1.5))
    if sc > 0.3: c = cv2.GaussianBlur(c, (0, 0), sc)
    if strength not in ('auto', 'chroma') and float(strength) > 0: y = _nlm(y, float(strength))
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


BAND_CORE = (0.6, 0.18, 0.27)   # soft-threshold of the luminance bands 0.5-1, 1-2 and 2-4 px as a fraction of the pixel noise: found empirically (the noise within a band is a fraction of the noise of the pixel, falling with the band's scale); see docs/progress.md


def core_luma(rgb, sig8, k=None):
    """Take the grain out of the luminance without taking the texture: it is split into bands by scale (finer than 1 px, 1 to 2 px, 2 to 4 px and the smooth rest) and the detail in each band weaker than a threshold (BAND_CORE times the noise expected at that pixel, `sig8`) is soft-thresholded to nothing,
    while stronger structure (the edge of a tile, a brick line, a window frame) passes unchanged. Unlike a non-local-means pass, which averages texture away, strong structure is untouched; and unlike coring the finest band only, the mottling at 1 to 4 px that demosaicing and noise reduction leave (most visible in light walls) is dealt with too.
    The thresholds also follow the local brightness: x0.7 in dark and mid tones, rising to x1.1 in bright walls (smoothstep over a smoothed luminance of 0.55 to 0.85): measured against the camera's JPEG in flat patches, that puts the grain of bright walls at the JPEG's level (it was 25 % above) without making the dark and mid areas smoother than the JPEG's.
    `k` scales all the thresholds (default 1)."""
    k = 1.0 if k is None else k; y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32); n = sig8 / 255.0
    b = cv2.GaussianBlur(y, (0, 0), 8.0); w = np.clip((b - 0.55) / 0.30, 0, 1); w = w * w * (3 - 2 * w); n = n * (0.7 + 0.4 * w)
    g = [y] + [cv2.GaussianBlur(y, (0, 0), s) for s in (1.0, 2.0, 4.0)]; out = g[3].copy()
    for band, a in zip((g[0] - g[1], g[1] - g[2], g[2] - g[3]), BAND_CORE): out += np.sign(band) * np.maximum(np.abs(band) - a * k * n, 0)
    return np.clip(rgb + (out - y)[..., None], 0, 1).astype(np.float32)


def noise_map8(img, noise_profile, blur=8.0, gain=None):
    """The noise expected at each pixel of the finished picture, in 8-bit grey levels, from the file's NoiseProfile: sigma of the linear value (variance = S v + O, with the profile already scaled for the exposure gain) times the slope of the sRGB curve there, on a smoothed luminance (the noise level follows the brightness, not the texture)."""
    h, w = img.shape[:2]; k = 1.0 / 8; y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    v = cv2.GaussianBlur(cv2.resize(y, (max(8, round(w * k)), max(8, round(h * k))), interpolation=cv2.INTER_AREA), (0, 0), blur / 8 * 2); v = np.maximum(v, 1e-5)
    slope = np.where(v < 0.0031308, 12.92, (1.055 / 2.4) * v ** (1 / 2.4 - 1))
    if gain is None: s8 = 255.0 * np.sqrt(noise_profile[0] * v + noise_profile[1]) * slope
    else: gs = cv2.GaussianBlur(cv2.resize(gain, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_AREA), (0, 0), blur / 8 * 2); s8 = 255.0 * np.sqrt(noise_profile[0] * gs * v + noise_profile[1] * gs * gs) * slope       # the lens-shading gain lifts the noise with the signal, beyond the global exposure gain the profile is scaled for
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


TONE_BLEND = 1.0                # how much of the profile tone curve is applied to each colour channel separately (1) rather than to the brightest channel with the colour ratios kept (0): see `finish`


def finish(img, lut, blend=None):
    """Shoulder, profile tone curve and sRGB curve of linear sRGB-primaries data (0..inf): the last steps of the development.
    The profile curve (it maps linear to display-encoded values, gamma included, so no sRGB curve follows it) can be applied two ways. PER CHANNEL, it darkens the weaker channels of a colour more than the strongest (the curve is convex at the bottom), which raises saturation: measured against the camera's JPEG
    on a sunlit orange roof it made the colour 25 to 33 % more saturated and a dozen lightness units darker, a heavy orange. On the BRIGHTEST CHANNEL with the colour ratios kept (the way Adobe's own profile curves are applied) the colour ratios stay as they were: median chroma matched the JPEG's. `blend` (default TONE_BLEND) is the share of the first way."""
    blend = TONE_BLEND if blend is None else blend; img = np.clip(shoulder(np.clip(img, 0, None)), 0, 1)
    if lut is None: return srgb_oetf(img).astype(np.float32)
    xs = np.linspace(0, 1, len(lut)); per = np.interp(img, xs, lut).astype(np.float32) if blend > 0 else None
    if blend >= 1: return per
    mx = img.max(axis=2, keepdims=True); mn = img.min(axis=2, keepdims=True); fx, fn = np.interp(mx, xs, lut), np.interp(mn, xs, lut); d = mx - mn
    ratio = (fn + (fx - fn) * np.where(d > 1e-6, (img - mn) / np.maximum(d, 1e-6), 0.0)).astype(np.float32)                    # Adobe's RGBTone: the curve on the largest and smallest channel, the middle one in proportion between them, so the hue holds
    return ratio if per is None else (blend * per + (1 - blend) * ratio).astype(np.float32)


CHROMA_KNEE = 0.20              # limit_chroma: colour (distance from the grey of the same luma, display-encoded 0..1) up to here is untouched
CHROMA_CEIL = 0.45              # ... and rolls off smoothly towards this at most


def limit_chroma(rgb, knee=None, ceil=None):
    """Roll off the strongest colours of a finished (display-encoded) picture: up to `knee` the colour is as it was, beyond it the excess is compressed towards `ceil`. Sunlit orange and similar are the ones the tone curve over-saturates (it darkens the weaker channels most); the camera's own JPEG is gentler there. Luma is kept."""
    knee = CHROMA_KNEE if knee is None else knee; ceil = CHROMA_CEIL if ceil is None else ceil
    if knee <= 0 or ceil <= knee: return rgb
    w = np.array([0.2126, 0.7152, 0.0722], np.float32); y = (rgb @ w)[..., None]; d = rgb - y; c = np.sqrt((d * d).sum(-1, keepdims=True)) + 1e-9; room = ceil - knee; ex = np.maximum(c - knee, 0)
    new = np.where(c > knee, knee + room * ex / (ex + room), c); return np.clip(y + d * (new / c), 0, 1).astype(np.float32)


HIGHLIGHTS = 0.55               # how hard local_tone squeezes bright regions towards the median (0..1)
HIGHLIGHT_PCT = 99.5
HIGHLIGHT_TARGET = 1.0          # the linear value the 99.5th percentile may reach after exposure (the shoulder and the tone curve take it the rest of the way to white)
HIGHLIGHT_CONTRAST = 0.8        # highlight_detail: how much more contrast the bright areas get (0 = none)
MAX_DARKER = 1.0                # protecting highlights may take the exposure at most this many stops below the median target (the shadows are lifted by the local tone recovery instead)


def auto_ev(img, lut, target=TARGET_MEDIAN, stride=8):
    """The exposure change (stops, -1..MAX_EV): the one that puts the median luminance of the finished picture at `target` (found by bisection on a sample, through the whole finish), but no brighter than lets the brightest 0.5 % of the picture stay below white: white walls and windows keep their tone
    (a median-only rule pushed them into a flat plateau, while the camera's JPEG kept them). Never more than MAX_DARKER stops below the median rule."""
    s = img[::stride, ::stride].reshape(-1, 3); w = np.array([0.2126, 0.7152, 0.0722], np.float32); y = s @ w; lo, hi = -1.0, MAX_EV
    for _ in range(12):
        mid = (lo + hi) / 2; med = float(np.median(finish((s * 2.0 ** mid)[None], lut) @ w))
        lo, hi = (mid, hi) if med < target else (lo, mid)
    ev_med = (lo + hi) / 2 if y.size else 0.0
    if not y.size: return ev_med
    ev_hl = float(np.log2(HIGHLIGHT_TARGET / max(float(np.percentile(y, HIGHLIGHT_PCT)), 1e-4)))
    return float(np.clip(min(ev_med, max(ev_hl, ev_med - MAX_DARKER)), -1.0, MAX_EV))


# ---- the whole development ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _raw_u16(path, half):
    """The demosaiced raw as LibRaw gives it: uint16 (H, W, 3), camera RGB, white-balanced, linear. A linear DNG (demosaiced RGB, which LibRaw refuses: Lightroom's merged panoramas) is read by lineardng.py into the same form."""
    import rawpy
    try:
        with rawpy.imread(path) as r:
            return r.postprocess(half_size=half, use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1), output_color=rawpy.ColorSpace.raw, demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT, highlight_mode=rawpy.HighlightMode.Blend, user_flip=0)
    except Exception as e:
        if not lineardng.is_linear(path): raise
        try: return lineardng.read(path, half)
        except Exception as e2: raise ValueError(f'neither LibRaw ({e}) nor the linear DNG reader ({e2}) could read {path}')


def _raw_linear(path, half): return _raw_u16(path, half).astype(np.float32) / 65535.0


def analysis_luma(path, u16=None):
    """The luminance of the raw at full size, gamma-encoded 0..1, with nothing done to it that changes its sharpness (no lens warp, no denoise, no local tone): what detail.effective_resolution measures."""
    img = _raw_linear(path, False) if u16 is None else u16.astype(np.float32) / 65535.0; y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32); med = max(float(np.median(y[::8, ::8])), 1e-5)
    return srgb_oetf(np.clip(y * (0.18 / med), 0, 1)).astype(np.float32)


def cpu_colour(img, tags):
    """The camera matrices (and the profile's hue/saturation map) on an (H, W, 3) picture: camera RGB to linear sRGB. Pointwise, so it can be run on a tile."""
    if all(k in tags for k in ('ColorMatrix1', 'ColorMatrix2', 'AsShotNeutral')):
        n = _rationals(tags['AsShotNeutral']); cm1 = _rationals(tags['ColorMatrix1']).reshape(3, 3); cm2 = _rationals(tags['ColorMatrix2']).reshape(3, 3)
        cm, w = color_matrix_weight(cm1, cm2, n); table = hue_sat_table(tags, w) if HUE_SAT_MAP else None
        if table is None: img = img @ camera_to_srgb(cm, n).T.astype(np.float32)
        else: img = apply_hue_sat_map(img @ camera_to_prophoto(cm, n).T.astype(np.float32), table) @ prophoto_to_srgb().T.astype(np.float32)
    return img


def develop_parts(path, half=False, ev='auto', recover=0.35, u16=None):
    """The development up to, but not including, the final tone curve: {linear: exposed linear sRGB-primaries picture (float32), lut, noise: the NoiseProfile scaled for the exposure gain (or None), gain: the lens-shading field (or None), gain_ev}. `develop` finishes it; experiments and the export can look inside."""
    tags = read_dng_tags(path); img = _raw_linear(path, half) if u16 is None else u16.astype(np.float32) / 65535.0; field = None
    if 'OpcodeList3' in tags:
        for oid, p in parse_opcodes(tags['OpcodeList3']):
            if oid == 9:
                m, plane, planes, _box, _sp, _org = gain_map(p); img, field = apply_gain_map(img, m, plane, planes)
            elif oid == 1:
                coef, centre = warp_params(p); img = warp_rectilinear(img, coef, centre)
    img = cpu_colour(img, tags)
    be =float(_rationals(tags['BaselineExposure'])[0]) if 'BaselineExposure' in tags else 0.0
    img = local_tone(np.clip(img, 0, None) * 2.0 ** be, recover); lut = tone_lut(inputs.tone_curve(tags))
    g = 2.0 ** (auto_ev(img, lut) if ev == 'auto' else float(ev)); img = img * g
    npf = tags.get('NoiseProfile'); npf = (npf[0] * g * 2.0 ** be, npf[1] * (g * 2.0 ** be) ** 2) if npf is not None else None          # the profile is for the raw values: after the gain, variance = S*g*v + O*g^2
    return dict(linear=img, lut=lut, noise=npf, gain=field, gain_ev=float(np.log2(g)))


def develop(path, half=False, ev='auto', denoise='auto', recover=0.35, model=None, strength=None, highlight_contrast=HIGHLIGHT_CONTRAST, backend='auto', analysis=None):
    """The finished picture (float32 RGB, 0..1, sRGB encoded). `denoise`:
      'auto'    the classical clean-up (default): colour noise and fringes removed edge-aware and cored to grey, colour rolled off near white, the luminance grain cored band by band (clean_colour, core_luma), with the lens-shading gain counted in the noise map so the lifted corners are cored harder
      'ai'      'auto' plus a learned model (`model`, `strength`: denoise_ai) where the noise is visible
      'chroma'  colour only; 'off' nothing; a number: non-local-means h (8-bit grey levels) everywhere.
    `analysis`: pass a dict and it gets 'luma', the analysis luminance of this very raw (what analysis_luma gives, from the same decode, so the caller need not decode the raw a second time). `backend`: 'auto' runs on the GPU (gpu.py) when there is one and the options are ones it does ('auto', 'chroma', 'off'), else on the CPU; 'cpu' and 'gpu' force it (the CPU code is the reference the GPU code is tested against)."""
    if backend != 'cpu' and denoise in ('auto', 'chroma', 'off') and (backend == 'gpu' or gpu.available()): return _develop_gpu(path, half, ev, denoise, recover, highlight_contrast, analysis)
    u16 = _raw_u16(path, half)
    if denoise in ('auto', 'chroma', 'off'):                                                                                  # a picture too big for the memory it may use is developed in tiles (cpu_tiled.py)
        tile = resources.cpu_tile(u16.shape[0] * u16.shape[1])
        if tile:
            try: return cpu_tiled.develop_tiled(path, u16, ev, denoise, recover, highlight_contrast, analysis, tile)
            except cpu_tiled.Unsupported: pass
    if analysis is not None: analysis['luma'] = analysis_luma(path, None if half else u16)
    P = develop_parts(path, half, ev, recover, u16=u16); img, lut, npf, field = P['linear'], P['lut'], P['noise'], P['gain']
    classical = denoise in ('auto', 'ai', 'chroma') and npf is not None
    if denoise in ('auto', 'ai'): img = denoise_linear(img, npf, 'auto', field)                                                           # colour noise (the luminance grain is cored below, driven by the noise map)
    elif denoise not in ('off', 'chroma'): img = denoise_linear(img, npf, denoise, field)
    sig8 = noise_map8(img, npf, gain=field) if npf is not None else None
    out = limit_chroma(finish(img, lut))
    if classical:
        out = clean_colour(out, sig8)
        if denoise in ('auto', 'ai'): out = core_luma(out, sig8)                                                        # grain first, so the contrast boost below acts on cleaned detail
    if highlight_contrast > 0: out = highlight_detail(out, highlight_contrast)
    if denoise == 'ai' and sig8 is not None:
        model = model or denoise_ai.DEFAULT; strength = denoise_ai.DEFAULT_STRENGTH if strength is None else strength
        if denoise_ai.available(model): out = denoise_ai.adaptive(out, sig8, model, strength)
    return out


def _develop_gpu(path, half, ev, denoise, recover, highlight_contrast, analysis=None):
    """The raw is decoded first (LibRaw, on the CPU, holding no device slot, so another picture can be on the card meanwhile); then the picture takes a device slot for the rest. A picture too big for the card (gpu.max_pixels) is developed in tiles (gpu_tiled.py); if that cannot be done for its opcodes, a reduced copy that fits is developed instead."""
    u16 = _raw_u16(path, half); limit = gpu.max_pixels()
    with gpu.slot():
        if limit and u16.shape[0] * u16.shape[1] > limit:
            try: return gpu_tiled.develop_tiled(path, u16, ev, denoise, recover, highlight_contrast, analysis)
            except gpu_tiled.Unsupported:
                k = (limit / (u16.shape[0] * u16.shape[1])) ** 0.5; u16 = cv2.resize(u16, (int(u16.shape[1] * k) // 2 * 2, int(u16.shape[0] * k) // 2 * 2), interpolation=cv2.INTER_AREA)
        return _develop_gpu_on_device(path, u16, half, ev, denoise, recover, highlight_contrast, analysis)


def gpu_colour(img, tags):
    """The camera matrices (and the profile's hue/saturation map) on a (3, H, W) picture on the device: camera RGB to linear sRGB. Pointwise, so it can be run on a tile."""
    t = gpu.torch_()
    if all(k in tags for k in ('ColorMatrix1', 'ColorMatrix2', 'AsShotNeutral')):
        n = _rationals(tags['AsShotNeutral']); cm1 = _rationals(tags['ColorMatrix1']).reshape(3, 3); cm2 = _rationals(tags['ColorMatrix2']).reshape(3, 3); cm, wt = color_matrix_weight(cm1, cm2, n); table = hue_sat_table(tags, wt) if HUE_SAT_MAP else None
        mat = lambda m: t.from_numpy(np.asarray(m, np.float32)).to(img.device); c, h, w = img.shape
        if table is None: img = (mat(camera_to_srgb(cm, n)) @ img.reshape(3, -1)).reshape(3, h, w)
        else: img = (mat(prophoto_to_srgb()) @ gpu.apply_hue_sat_map((mat(camera_to_prophoto(cm, n)) @ img.reshape(3, -1)).reshape(3, h, w), mat(table)).reshape(3, -1)).reshape(3, h, w)
    return img


def gpu_tail(img, lut_t, npf, field, sc, denoise, highlight_contrast):
    """The last part of the GPU development, from the exposed linear picture `img` (3, H, W) to the finished one: colour noise blurred by `sc` (the visible-noise estimate of the whole picture), the noise map, the tone curve and colour limit, the colour and luminance clean-up, the highlight contrast. Everything in it is local (the widest reach is about 90 pixels), so it can be run on a tile with a margin (gpu_tiled.py) and give what the whole picture would."""
    classical = denoise in ('auto', 'chroma') and npf is not None
    if classical and sc > 0.3:                                                                                         # denoise_linear: colour noise blurred in proportion to the noise
        y = gpu.luma(img); cc = gpu.gaussian(img - y[None], sc); img = (y[None] + cc).clamp_(min=0)
    sig8 = gpu.noise_map8(img, npf, gain=field) if npf is not None else None
    out = gpu.limit_chroma(gpu.finish(img, lut_t, KNEE, WHITE, TONE_BLEND), CHROMA_KNEE, CHROMA_CEIL)
    if classical:
        out = gpu.clean_colour(out, sig8)
        if denoise == 'auto': out = gpu.core_luma(out, sig8, BAND_CORE)
    if highlight_contrast > 0: out = gpu.highlight_detail(out, highlight_contrast)
    return out


def _develop_gpu_on_device(path, u16, half, ev, denoise, recover, highlight_contrast, analysis=None):
    """`develop` on the GPU: the same recipe step for step (see gpu.py for the functions and tests/unit/test_gpu.py for the proof they agree with the CPU ones); the picture stays on the card until the finished float32 array comes back."""
    t = gpu.torch_(); tags = read_dng_tags(path); img = gpu.to_gpu_u16(u16); field = None
    if analysis is not None:                                                                                      # analysis_luma from this decode: nothing done that changes sharpness, exposed so the median is 0.18, gamma-encoded
        y = gpu.luma(img); med = max(gpu._median(y[::8, ::8]), 1e-5); analysis['luma'] = gpu.to_numpy(gpu.srgb_oetf(y * (0.18 / med))); del y
    if 'OpcodeList3' in tags:
        for oid, p in parse_opcodes(tags['OpcodeList3']):
            if oid == 9:
                m, plane, planes, _box, _sp, _org = gain_map(p); img, field = gpu.apply_gain_map(img, m, plane, planes)
            elif oid == 1:
                coef, centre = warp_params(p); img = gpu.warp_rectilinear(img, coef, centre)
    img = gpu_colour(img, tags)
    be = float(_rationals(tags['BaselineExposure'])[0]) if 'BaselineExposure' in tags else 0.0
    img = gpu.local_tone(img.clamp(min=0) * 2.0 ** be, recover, HIGHLIGHTS); lut = tone_lut(inputs.tone_curve(tags)); lut_t = t.from_numpy(lut).to(img.device) if lut is not None else None
    sample = gpu.to_numpy(img[:, ::8, ::8]); g = 2.0 ** (auto_ev(sample, lut, stride=1) if ev == 'auto' else float(ev)); img = img * g
    npf = tags.get('NoiseProfile'); npf = (npf[0] * g * 2.0 ** be, npf[1] * (g * 2.0 ** be) ** 2) if npf is not None else None
    sc = 0.0
    if npf is not None and denoise in ('auto', 'chroma'): sc = float(np.clip(visible_noise(gpu.to_numpy(img[:, ::8, ::8]), npf, stride=1) * 0.6, 0.0, 1.5))
    out = gpu_tail(img, lut_t, npf, field, sc, denoise, highlight_contrast); del img
    res = gpu.to_numpy(out); del out; gpu.free(); return res
