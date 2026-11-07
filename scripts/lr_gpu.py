"""The look code of scripts/lr_look.py and the brightness match of scripts/arena_variants.py in PyTorch, for the arena and the Lightroom report (stratastills.gpu conventions: float32, the CPU code is the REFERENCE, a function here is held to its original by tests/unit/test_lr_gpu.py on every device there is). Everything takes and returns what the CPU function does (uint8 BGR numpy pictures, the same `did` dicts), so a caller changes nothing: lr_look.apply_look and arena_variants.match_brightness use this module when `use()` says so (a CUDA or MPS device, and STRATASTILLS_GPU not 0).
Not here on purpose: the pixel pipeline's own code (src/stratastills is hashed into the neutral pictures' cache keys by build_report.code_hash, a new function there would make every neutral picture again). Small scalar decisions (curve knots, slider trust) are the CPU functions themselves; only the per-pixel work is on the device."""
import os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stratastills import gpu
import lr_look as LK
import lr_dataset as D

# the constants of OpenCV's float BGR<->CIELAB (D65, sRGB primaries): cv2.cvtColor on float32 pictures in 0..1
_M = ((0.412453, 0.357580, 0.180423), (0.212671, 0.715160, 0.072169), (0.019334, 0.119193, 0.950227))
_MI = ((3.240479, -1.53715, -0.498535), (-0.969256, 1.875991, 0.041556), (0.055648, -0.204043, 1.057311))
_WHITE = (0.950456, 1.0, 1.088754)


def use():
    """True when the per-pixel work should run on the device: there is one (gpu.available(), so STRATASTILLS_GPU=0 and STRATASTILLS_DEVICE=cpu switch it off) and torch imports."""
    try: return gpu.available()
    except Exception: return False


def up(bgr):
    """A uint8 BGR picture (H, W, 3) as float32 BGR 0..1 on the device (H, W, 3)."""
    return gpu.torch_().from_numpy(np.ascontiguousarray(bgr)).to(gpu.dev()).float() / 255


def down(x):
    """float 0..1 BGR (H, W, 3) -> uint8, rounded like `np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)`."""
    return (x * 255 + 0.5).clamp(0, 255).to(gpu.torch_().uint8).cpu().numpy()


def _mat(m, x):
    t = gpu.torch_(); return x @ t.tensor(m, dtype=t.float32, device=x.device).T


def bgr_to_lab(x):
    """cv2.cvtColor(BGR2LAB) of a float32 picture in 0..1: (H, W, 3) BGR -> (H, W, 3) L (0..100), a, b."""
    t = gpu.torch_(); rgb = x.flip(-1).clamp(0, 1); lin = t.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = _mat(_M, lin) / t.tensor(_WHITE, dtype=t.float32, device=x.device); f = t.where(xyz > 0.008856, xyz.clamp_min(1e-12) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return t.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_bgr(lab):
    """cv2.cvtColor(LAB2BGR) of a float32 picture: (H, W, 3) L, a, b -> BGR 0..1 (clipped, as OpenCV does before the gamma)."""
    t = gpu.torch_(); fy = (lab[..., 0] + 16) / 116; fx = fy + lab[..., 1] / 500; fz = fy - lab[..., 2] / 200; f = t.stack([fx, fy, fz], -1)
    xyz = t.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * t.tensor(_WHITE, dtype=t.float32, device=lab.device); rgb = _mat(_MI, xyz).clamp(0, 1)
    return t.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb.clamp_min(1e-12) ** (1 / 2.4) - 0.055).flip(-1)


LAB = os.environ.get('STRATASTILLS_LR_LAB', 'cv2')           # 'cv2': the colour conversions stay OpenCV's (15 ms a picture; its float BGR2LAB goes through 1024-entry spline tables and is off the exact formula by up to 0.19 in L, which the tone curve (slope up to 4) turns into one grey level of difference from the CPU path); 'torch': bgr_to_lab / lab_to_bgr above (exact formula, on the device)


def lab_in(bgr):
    """uint8 BGR picture -> CIELAB (H, W, 3) tensor on the device."""
    if LAB == 'torch': return bgr_to_lab(up(bgr))
    return gpu.torch_().from_numpy(cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)).to(gpu.dev())


def lab_out(lab):
    """CIELAB (H, W, 3) tensor -> uint8 BGR picture, rounded like the CPU code."""
    if LAB == 'torch': return down(lab_to_bgr(lab))
    return np.clip(cv2.cvtColor(lab.cpu().numpy(), cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)


def percentiles(L, ps):
    """np.percentile(L, ps) (linear interpolation) as a list of floats."""
    t = gpu.torch_(); return [float(v) for v in t.quantile(L.flatten(), t.tensor([p / 100 for p in ps], dtype=t.float32, device=L.device)).cpu()]


def interp(x, xp, fp):
    """np.interp(x, xp, fp) for a tensor `x` and short numpy knots (ends held)."""
    t = gpu.torch_(); xp_ = t.tensor(np.asarray(xp), dtype=t.float32, device=x.device); fp_ = t.tensor(np.asarray(fp), dtype=t.float32, device=x.device)
    xc = x.clamp(float(xp[0]), float(xp[-1])); i = (t.bucketize(xc, xp_, right=True) - 1).clamp(0, len(xp) - 2); x0, x1 = xp_[i], xp_[i + 1]
    return fp_[i] + (fp_[i + 1] - fp_[i]) * (xc - x0) / (x1 - x0)


def _area_weights(n, m, device):
    """(m, n) matrix of cv2.INTER_AREA: output pixel i averages the input span [i * n / m, (i + 1) * n / m) with fractional weights at its two ends (torch's 'area' mode, adaptive pooling, uses whole-pixel windows instead and differs by up to 5 L on a quarter-size copy)."""
    t = gpu.torch_(); sc = n / m; lo = t.arange(m, device=device, dtype=t.float64) * sc; hi = lo + sc; j = t.arange(n, device=device, dtype=t.float64)
    w = (t.minimum(hi[:, None], j[None] + 1) - t.maximum(lo[:, None], j[None])).clamp_min(0); return (w / sc).float()


def resize_area(x, size):
    """cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) of a plane (H, W) to `size` = (h, w), a reduction: two matrix products."""
    return _area_weights(x.shape[0], size[0], x.device) @ x @ _area_weights(x.shape[1], size[1], x.device).T


def _base(L):
    """lr_look._base: the guided-filter base of the lightness plane (0..100)."""
    h, w = L.shape; k = 4; small = resize_area(L / 100, (max(8, h // k), max(8, w // k))); r = max(2, max(small.shape) // 30)
    mean = lambda x: gpu.box(x, r); mi = mean(small); var = mean(small * small) - mi * mi; A = var / (var + LK.GUIDE_EPS); B = mi - A * mi
    return gpu.resize_linear(mean(A) * small + mean(B), (h, w)) * 100


def local_tone(L, shadows, highlights):
    if abs(shadows) < 1 and abs(highlights) < 1: return L
    base = _base(L); t = (base / 100).clamp(0, 1); lift = LK.SHADOW_GAIN * shadows / 100 * (1 - t) ** 2 + LK.HIGHLIGHT_GAIN * highlights / 100 * t ** 2
    return (L + lift).clamp(0, 100)


def clarity(L, amount):
    if abs(amount) < 1: return L
    sigma = max(2.0, 0.015 * max(L.shape)); detail = L - gpu.gaussian(L, sigma); mid = 1 - ((L - 50) / 50) ** 2
    return (L + LK.CLARITY_GAIN * amount / 100 * mid.clamp(0, 1) * detail).clamp(0, 100)


def vignette(L, amount):
    if abs(amount) < 1: return L
    t = gpu.torch_(); h, w = L.shape; y = t.arange(h, device=L.device, dtype=t.float32).view(-1, 1); x = t.arange(w, device=L.device, dtype=t.float32).view(1, -1)
    r2 = (((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / 2; return L * (1 + LK.VIGNETTE_GAIN * amount / 100 * r2)


def _median(x): return float(gpu.torch_().quantile(x.flatten(), 0.5))


def skin_rotation(L, a, b):
    t = gpu.torch_(); h = t.rad2deg(t.atan2(b, a)); C = t.hypot(a, b); m = (h > 25) & (h < 75) & (C > 12) & (C < 45) & (L > 25) & (L < 90)
    if float(m.float().mean()) < 0.03: return 0.0
    return float(np.clip(0.6 * (LK.SKIN_HUE - _median(h[m])), 0, LK.SKIN_TURN))


def _apply_look(bgr, pred, cam, colour=None):
    """lr_look._apply_look on the device."""
    t = gpu.torch_(); lab = lab_in(bgr); L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    L = vignette(local_tone(L, LK.slider(pred, 'Shadows2012'), LK.slider(pred, 'Highlights2012')), LK.slider(pred, 'PostCropVignetteAmount'))
    src = percentiles(L, D.PCTS); dst = list(pred[:len(D.PCTS)]); x, y = LK.curve_knots(np.array(src), np.array(dst)); L2 = interp(L, x, y)
    L2 = clarity(L2, LK.slider(pred, 'Clarity2012'))
    if colour is not None:
        import lr_fit as FIT
        p = t.from_numpy(np.asarray(colour, np.float32).copy())[None]; p[:, :len(FIT.L_KNOTS) - 1] = FIT.init_params(1, 'cpu')[:, :len(FIT.L_KNOTS) - 1]
        with t.no_grad(): out_lab = FIT.transform(t.stack([L2, a, b])[None], p.to(L.device))[0].permute(1, 2, 0)
        return lab_out(out_lab), dict(curve=[(round(float(i), 1), round(float(j), 1)) for i, j in zip(x, y)], chroma_scale='learned', cast=tuple(round(float(v), 2) for v in (colour[-3] * 5, colour[-2] * 5)), sliders={}, source_L=[round(v, 1) for v in src])
    p = dict(zip(LK.TARGETS, pred)); C = t.hypot(a, b); n_chroma, n_a, n_b = float(C.mean()), float(a.mean()), float(b.mean())
    scale = float(np.clip(1 + LK.COLOUR_TRUST * (p['C_mean'] / max(n_chroma, 0.5) - 1), *LK.CHROMA_SCALE))
    da, db = (float(np.clip(LK.CAST_TRUST * (p['a_mean'] - n_a), -LK.CAST_MAX, LK.CAST_MAX)), float(np.clip(LK.CAST_TRUST * (p['b_mean'] - n_b), -LK.CAST_MAX, LK.CAST_MAX)))
    hue = t.rad2deg(t.atan2(b, a)) % 360; skin = t.exp(-0.5 * ((hue - 50) / 22) ** 2)
    vib = float(np.clip(1 + (LK.slider(pred, 'Vibrance') - dict(LK.SLIDERS)['Vibrance']) / 100 * LK.VIBRANCE_GAIN, 0.6, 1.6))
    w = 1 / (1 + (C / LK.VIB_KNEE) ** 2) * (1 - LK.SKIN_HOLD * skin); s = 1 + (scale - 1) * w * vib; wl = (L2 / 20.0).clamp(0, 1)
    a2, b2 = a * s + da * wl, b * s + db * wl; rot = skin_rotation(L2, a2, b2)
    if rot: c, sn = float(np.float32(np.cos(np.radians(rot)))), float(np.float32(np.sin(np.radians(rot)))); a2, b2 = a2 * c - b2 * sn, a2 * sn + b2 * c
    return lab_out(t.stack([L2, a2, b2], -1)), dict(curve=[(round(float(i), 1), round(float(j), 1)) for i, j in zip(x, y)], chroma_scale=round(scale, 3), cast=(round(da, 2), round(db, 2)), sliders={k: round(LK.slider(pred, k), 1) for k, _ in LK.SLIDERS if k in ('Highlights2012', 'Shadows2012', 'PostCropVignetteAmount')}, source_L=[round(v, 1) for v in src])


def anchor_face(bgr, box, target):
    """lr_look.anchor_face on the device."""
    t = gpu.torch_(); lab = lab_in(bgr); L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]; H, W = L.shape; x, y, w, h = box
    sl = (slice(max(0, int((y + h / 4) * H)), int((y + 3 * h / 4) * H)), slice(max(0, int((x + w / 4) * W)), int((x + 3 * w / 4) * W)))
    if L[sl].numel() == 0: return bgr, {}
    fL, fa, fb = _median(L[sl]), _median(a[sl]), _median(b[sl]); fC = float(np.hypot(fa, fb)); fh = float(np.degrees(np.arctan2(fb, fa))); tL, ta, tb = target; tC = float(np.hypot(ta, tb)); th = float(np.degrees(np.arctan2(tb, ta)))
    dL = float(np.clip(tL - fL, -LK.FACE_L_MAX, LK.FACE_L_MAX)); gC = float(np.clip(tC / max(fC, 1.0), LK.FACE_C_RANGE[0], LK.FACE_C_RANGE[1])); rot = float(np.clip(0.5 * LK._ang(th - fh), -LK.FACE_ROT_MAX, LK.FACE_ROT_MAX))
    C = t.hypot(a, b); hue = t.rad2deg(t.atan2(b, a)); d = (hue - fh + 180) % 360 - 180; band = t.exp(-0.5 * (d / 25) ** 2) * (C / 8).clamp(0, 1)
    L2 = (L + dL * t.exp(-0.5 * ((L - fL) / 22) ** 2)).clamp(0, 100); C2 = C * (1 + (gC - 1) * band); h2 = t.deg2rad(hue + rot * band)
    return lab_out(t.stack([L2, C2 * t.cos(h2), C2 * t.sin(h2)], -1)), dict(face_dL=round(dL, 1), face_chroma_gain=round(gC, 2), face_hue_turn=round(rot, 1))


def match_brightness(img, ref):
    """arena_variants.match_brightness on the device: `img` with its mean CIELAB lightness brought to `ref`'s by a gamma on L (bisection on the gamma, as the CPU function)."""
    t = gpu.torch_(); lab = lab_in(img); L = lab[..., 0] / 100; target = float(lab_in(ref)[..., 0].mean()); lo, hi = 0.2, 3.0; Lc = L.clamp(1e-4, 1)
    for _ in range(30):
        g = (lo + hi) / 2
        if float((100 * Lc ** g).mean()) > target: lo = g
        else: hi = g
    lab[..., 0] = 100 * Lc ** ((lo + hi) / 2)
    return lab_out(lab)
