"""What the person does to a picture, fitted from each pair (scripts/lr_dataset.py thumbs): a small colour-and-tone transform T(params) from the camera's preview to their final, with params found by gradient descent on every pair at once (torch, GPU when there is one). The look model then learns to predict the params, not whole-picture averages, so applying a prediction is just T: no hand-set hold-backs.
  python scripts/lr_fit.py fit [--limit N]   fit every pair in candidates/personal/thumbs -> candidates/personal/fit.npz (params + the pictures' distance to the final before and after)
  python scripts/lr_fit.py check             how well T can reproduce the finals (the ceiling of the approach), and the spread of each parameter
T works in CIELAB: (1) a monotone lightness curve through fixed input knots (ends pinned), (2) a chroma gain that depends on the input chroma, separately for skin-like hues and the rest, (3) a cast (a, b shifted, not into black) and (4) a global hue rotation."""
import argparse, glob, os, sys
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D

L_KNOTS = [0, 10, 25, 50, 75, 90, 100]
C_KNOTS = [0, 15, 30, 50, 80]
NP = (len(L_KNOTS) - 1) + 2 * len(C_KNOTS) + 3             # 6 lightness segments + chroma gains (rest, skin) + (da, db, rotation)
NAMES = [f'Lseg{i}' for i in range(len(L_KNOTS) - 1)] + [f'gain_rest_C{c}' for c in C_KNOTS] + [f'gain_skin_C{c}' for c in C_KNOTS] + ['cast_a', 'cast_b', 'rot_deg']


def to_lab(bgr):
    """(N, H, W, 3) uint8 BGR -> float32 tensor (N, 3, H, W) of CIELAB (L 0-100, a, b)."""
    lab = np.stack([cv2.cvtColor(x.astype(np.float32) / 255, cv2.COLOR_BGR2LAB) for x in bgr]); return torch.from_numpy(lab).permute(0, 3, 1, 2).contiguous()


def init_params(n, dev):
    """Identity: lightness segments in proportion to their widths, gains 1, no cast, no rotation (the lightness segments and gains are kept as logs/softplus inputs)."""
    w = np.diff(L_KNOTS).astype(np.float32); th = np.log(np.expm1(w / w.sum() * 4)); p = torch.zeros(n, NP, device=dev)
    p[:, :len(w)] = torch.tensor(th, device=dev); return p


def _interp(x, xs, ys):
    """Piecewise-linear y(x): x (N, H, W), xs (K,) fixed, ys (N, K) per picture; differentiable in ys."""
    xs_t = torch.tensor(xs, dtype=x.dtype, device=x.device); idx = torch.clamp(torch.bucketize(x.contiguous(), xs_t) - 1, 0, len(xs) - 2).flatten(1)
    x0, x1 = xs_t[idx], xs_t[idx + 1]; y0, y1 = ys.gather(1, idx), ys.gather(1, idx + 1); t = (x.flatten(1).clamp(xs[0], xs[-1]) - x0) / (x1 - x0)
    return (y0 + (y1 - y0) * t).view_as(x)


def transform(lab, p):
    """Apply the transform with parameters `p` (N, NP) to CIELAB pictures `lab` (N, 3, H, W) -> CIELAB."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]; nl = len(L_KNOTS) - 1; nc = len(C_KNOTS)
    inc = torch.nn.functional.softplus(p[:, :nl]); ys = torch.cat([torch.zeros_like(inc[:, :1]), torch.cumsum(inc, 1) / inc.sum(1, keepdim=True) * 100], 1)      # monotone, ends pinned at 0 and 100
    L2 = _interp(L, L_KNOTS, ys)
    C = torch.hypot(a, b); h = torch.atan2(b, a) * 180 / np.pi % 360; skin = torch.exp(-0.5 * ((h - 50) / 20) ** 2) * torch.clamp(C / 10, 0, 1)
    g_rest = torch.exp(p[:, nl:nl + nc].clamp(-2, 2)); g_skin = torch.exp(p[:, nl + nc:nl + 2 * nc].clamp(-2, 2))
    g = _interp(C, C_KNOTS, g_rest) * (1 - skin) + _interp(C, C_KNOTS, g_skin) * skin
    da, db, rot = (p[:, -3] * 5).view(-1, 1, 1), (p[:, -2] * 5).view(-1, 1, 1), (p[:, -1] * 10 * np.pi / 180).view(-1, 1, 1); wl = torch.clamp(L2 / 20, 0, 1)
    a2, b2 = a * g + da * wl, b * g + db * wl; c, s = torch.cos(rot), torch.sin(rot)
    return torch.stack([L2, a2 * c - b2 * s, a2 * s + b2 * c], 1)


def decode(p):
    """Params as people read them: lightness curve output at the knots, chroma gains, cast (Lab units) and rotation (degrees)."""
    nl = len(L_KNOTS) - 1; nc = len(C_KNOTS); inc = torch.nn.functional.softplus(p[:, :nl]); ys = torch.cat([torch.zeros_like(inc[:, :1]), torch.cumsum(inc, 1) / inc.sum(1, keepdim=True) * 100], 1)
    return torch.cat([ys[:, 1:-1], torch.exp(p[:, nl:nl + 2 * nc].clamp(-2, 2)), p[:, -3:-2] * 5, p[:, -2:-1] * 5, p[:, -1:] * 10], 1)


def distance(a, b, blur=1.5):
    """Per-picture mean |difference| in L and mean CIE76 colour distance between two CIELAB batches, after a small blur (so that a pixel of misregistration and noise do not count)."""
    k = int(blur * 3) * 2 + 1; ker = cv2.getGaussianKernel(k, blur).astype(np.float32); K = torch.from_numpy(ker @ ker.T).to(a.device).expand(3, 1, k, k)
    bl = lambda x: torch.nn.functional.conv2d(torch.nn.functional.pad(x, (k // 2,) * 4, mode='replicate'), K, groups=3)
    d = bl(a) - bl(b); return d[:, 0].abs().mean((1, 2)), torch.sqrt(d[:, 1] ** 2 + d[:, 2] ** 2).mean((1, 2))


def fit_batch(cam, fin, steps=400, lr=0.05):
    """Params for each pair in the batch (CIELAB tensors on one device), by Adam on a smooth L1 distance of the blurred pictures."""
    dev = cam.device; p = init_params(len(cam), dev).requires_grad_(True); opt = torch.optim.Adam([p], lr=lr)
    k = 5; ker = cv2.getGaussianKernel(k, 1.2).astype(np.float32); K = torch.from_numpy(ker @ ker.T).to(dev).expand(3, 1, k, k)
    bl = lambda x: torch.nn.functional.conv2d(torch.nn.functional.pad(x, (2,) * 4, mode='replicate'), K, groups=3); tgt = bl(fin)
    for i in range(steps):
        for g in opt.param_groups: g['lr'] = lr * (0.1 + 0.9 * (1 - i / steps))
        opt.zero_grad(); loss = torch.nn.functional.smooth_l1_loss(bl(transform(cam, p)), tgt, beta=2.0); loss.backward(); opt.step()
    return p.detach()


def load_pairs(limit=None):
    parts = sorted(glob.glob(os.path.join(D.OUT, 'thumbs', 't*.npz')))[:limit]; z = [np.load(f) for f in parts]
    return np.concatenate([x['cam'] for x in z]), np.concatenate([x['fin'] for x in z]), np.concatenate([x['idx'] for x in z])


def fit(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; cam, fin, idx = load_pairs(a.limit); n = len(idx); print(f'{n} pairs on {dev}', flush=True); P = np.zeros((n, NP), np.float32); before = np.zeros((n, 2), np.float32); after = np.zeros((n, 2), np.float32)
    for s in range(0, n, a.batch):
        c, f = to_lab(cam[s:s + a.batch]).to(dev), to_lab(fin[s:s + a.batch]).to(dev); p = fit_batch(c, f, a.steps)
        with torch.no_grad(): b0 = distance(c, f); a0 = distance(transform(c, p), f)
        P[s:s + len(p)] = p.cpu().numpy(); before[s:s + len(p)] = torch.stack(b0, 1).cpu().numpy(); after[s:s + len(p)] = torch.stack(a0, 1).cpu().numpy(); print(f'{s + len(p)}/{n}: L {before[:s + len(p), 0].mean():.2f} -> {after[:s + len(p), 0].mean():.2f}, colour {before[:s + len(p), 1].mean():.2f} -> {after[:s + len(p), 1].mean():.2f}', flush=True)
    np.savez(os.path.join(D.OUT, 'fit.npz'), params=P, decoded=decode(torch.from_numpy(P)).numpy(), before=before, after=after, idx=idx, names=np.array(NAMES))


def check(a):
    z = np.load(os.path.join(D.OUT, 'fit.npz')); dec = z['decoded']; b, f = z['before'], z['after']
    print(f"{len(b)} pairs. mean |dL| {b[:, 0].mean():.2f} -> {f[:, 0].mean():.2f}; colour distance {b[:, 1].mean():.2f} -> {f[:, 1].mean():.2f} (camera preview as it is -> after the fitted transform: the ceiling)")
    print('worst 5 percent of pairs after the fit: |dL| %.2f, colour %.2f' % tuple(np.percentile(f, 95, axis=0)))
    names = ['L@10', 'L@25', 'L@50', 'L@75', 'L@90'] + [f'gain rest C{c}' for c in C_KNOTS] + [f'gain skin C{c}' for c in C_KNOTS] + ['cast a', 'cast b', 'rot deg']
    for i, n in enumerate(names): print(f'  {n:16s} p10 {np.percentile(dec[:, i], 10):7.2f}  median {np.median(dec[:, i]):7.2f}  p90 {np.percentile(dec[:, i], 90):7.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('step', choices=['fit', 'check']); ap.add_argument('--limit', type=int); ap.add_argument('--batch', type=int, default=128); ap.add_argument('--steps', type=int, default=400)
    a = ap.parse_args(); {'fit': fit, 'check': check}[a.step](a)
