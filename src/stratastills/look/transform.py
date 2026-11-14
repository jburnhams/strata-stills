"""The colour-and-tone transform T(params) the learned look predicts (moved from scripts/lr_fit.py, which keeps the fitting): in CIELAB, (1) a monotone lightness curve through fixed input knots (ends pinned), (2) a chroma gain that depends on the input chroma, separately for skin-like hues and the rest, (3) a cast (a, b shifted, not into black) and (4) a global hue rotation."""

import cv2
import numpy as np
import torch

L_KNOTS = [0, 10, 25, 50, 75, 90, 100]

C_KNOTS = [0, 15, 30, 50, 80]

NP = (len(L_KNOTS) - 1) + 2 * len(C_KNOTS) + 3  # 6 lightness segments + chroma gains (rest, skin) + (da, db, rotation)

NAMES = (
    [f"Lseg{i}" for i in range(len(L_KNOTS) - 1)]
    + [f"gain_rest_C{c}" for c in C_KNOTS]
    + [f"gain_skin_C{c}" for c in C_KNOTS]
    + ["cast_a", "cast_b", "rot_deg"]
)


def to_lab(bgr):
    """(N, H, W, 3) uint8 BGR -> float32 tensor (N, 3, H, W) of CIELAB (L 0-100, a, b)."""
    lab = np.stack([cv2.cvtColor(x.astype(np.float32) / 255, cv2.COLOR_BGR2LAB) for x in bgr])
    return torch.from_numpy(lab).permute(0, 3, 1, 2).contiguous()


def init_params(n, dev):
    """Identity: lightness segments in proportion to their widths, gains 1, no cast, no rotation (the lightness segments and gains are kept as logs/softplus inputs)."""
    w = np.diff(L_KNOTS).astype(np.float32)
    th = np.log(np.expm1(w / w.sum() * 4))
    p = torch.zeros(n, NP, device=dev)
    p[:, : len(w)] = torch.tensor(th, device=dev)
    return p


def _interp(x, xs, ys):
    """Piecewise-linear y(x): x (N, H, W), xs (K,) fixed, ys (N, K) per picture; differentiable in ys."""
    xs_t = torch.tensor(xs, dtype=x.dtype, device=x.device)
    idx = torch.clamp(torch.bucketize(x.contiguous(), xs_t) - 1, 0, len(xs) - 2).flatten(1)
    x0, x1 = xs_t[idx], xs_t[idx + 1]
    y0, y1 = ys.gather(1, idx), ys.gather(1, idx + 1)
    t = (x.flatten(1).clamp(xs[0], xs[-1]) - x0) / (x1 - x0)
    return (y0 + (y1 - y0) * t).view_as(x)


def transform(lab, p):
    """Apply the transform with parameters `p` (N, NP) to CIELAB pictures `lab` (N, 3, H, W) -> CIELAB."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    nl = len(L_KNOTS) - 1
    nc = len(C_KNOTS)
    inc = torch.nn.functional.softplus(p[:, :nl])
    ys = torch.cat(
        [torch.zeros_like(inc[:, :1]), torch.cumsum(inc, 1) / inc.sum(1, keepdim=True) * 100], 1
    )  # monotone, ends pinned at 0 and 100
    L2 = _interp(L, L_KNOTS, ys)
    C = torch.hypot(a, b)
    h = torch.atan2(b, a) * 180 / np.pi % 360
    skin = torch.exp(-0.5 * ((h - 50) / 20) ** 2) * torch.clamp(C / 10, 0, 1)
    g_rest = torch.exp(p[:, nl : nl + nc].clamp(-2, 2))
    g_skin = torch.exp(p[:, nl + nc : nl + 2 * nc].clamp(-2, 2))
    g = _interp(C, C_KNOTS, g_rest) * (1 - skin) + _interp(C, C_KNOTS, g_skin) * skin
    da, db, rot = (
        (p[:, -3] * 5).view(-1, 1, 1),
        (p[:, -2] * 5).view(-1, 1, 1),
        (p[:, -1] * 10 * np.pi / 180).view(-1, 1, 1),
    )
    wl = torch.clamp(L2 / 20, 0, 1)
    a2, b2 = a * g + da * wl, b * g + db * wl
    c, s = torch.cos(rot), torch.sin(rot)
    return torch.stack([L2, a2 * c - b2 * s, a2 * s + b2 * c], 1)


def decode(p):
    """Params as people read them: lightness curve output at the knots, chroma gains, cast (Lab units) and rotation (degrees)."""
    nl = len(L_KNOTS) - 1
    nc = len(C_KNOTS)
    inc = torch.nn.functional.softplus(p[:, :nl])
    ys = torch.cat([torch.zeros_like(inc[:, :1]), torch.cumsum(inc, 1) / inc.sum(1, keepdim=True) * 100], 1)
    return torch.cat(
        [ys[:, 1:-1], torch.exp(p[:, nl : nl + 2 * nc].clamp(-2, 2)), p[:, -3:-2] * 5, p[:, -2:-1] * 5, p[:, -1:] * 10],
        1,
    )
