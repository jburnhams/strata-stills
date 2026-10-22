"""Learned denoising on the GPU: SCUNet (`scunet_color_real_psnr`, Zhang et al., Apache-2.0, loaded with spandrel), the model strata-360 found cleanest for real camera noise (it removes noise and blocky patterns without inventing texture; docs/progress/2026-10-music-upscale.md there).
What 360 learned and this module follows: denoise BEFORE any enlarging, and never at full strength everywhere (SCUNet over-smooths low-contrast, low-light texture). So `denoise` returns the cleaned picture and `adaptive` blends it back into the original by a weight map that comes from the noise itself (develop.py gives it the
noise expected at each pixel from the file's NoiseProfile): full strength where the noise is plainly visible, none where it is below what the eye sees, so texture in clean areas is left exactly as the sensor recorded it.

Runs in tiles of TILE px with OVERLAP px of feathered overlap (a 48 MP picture does not fit the 12 GB card at once), in half precision. Weights: `models/scunet/scunet_color_real_psnr.pth` (gitignored): see `fetch()`."""
import os, urllib.request

import numpy as np

URL = 'https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth'
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'scunet', 'scunet_color_real_psnr.pth')
TILE = 1024
OVERLAP = 64
_model = {}


def available(): return os.path.exists(WEIGHTS)


def fetch():
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    if not available(): urllib.request.urlretrieve(URL, WEIGHTS)
    return WEIGHTS


def _load():
    if 'm' not in _model:
        import torch
        from spandrel import ModelLoader
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = ModelLoader().load_from_file(WEIGHTS).model.eval().to(dev)
        if dev.type == 'cuda': m = m.half()
        _model['m'] = (m, dev)
    return _model['m']


def _feather(n, at_start, at_end, overlap):
    """Blend weights along one side of a tile: 1 in the middle, ramping to ~0 across the overlap where a neighbouring tile takes over (and 1 right to the edge of the picture)."""
    w = np.ones(n, np.float32); k = min(overlap, n // 2)
    if not at_start: w[:k] = (np.arange(k) + 0.5) / k
    if not at_end: w[n - k:] = np.minimum(w[n - k:], (np.arange(k)[::-1] + 0.5) / k)
    return w


def denoise(rgb, tile=TILE, overlap=OVERLAP, need=None):
    """The SCUNet-cleaned copy of `rgb` (float32 HxWx3, 0..1, display-encoded: it is trained on gamma-encoded pictures). `need(y0, y1, x0, x1)` (optional) says whether a tile has to be cleaned at all; a tile that does not is passed through (saves most of the time when only some areas are noisy)."""
    import torch
    m, dev = _load(); h, w = rgb.shape[:2]; out = np.zeros_like(rgb); wsum = np.zeros((h, w, 1), np.float32); step = tile - 2 * overlap
    for y0 in range(0, max(h - overlap, 1), step):
        for x0 in range(0, max(w - overlap, 1), step):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w); y0c, x0c = max(0, y1 - tile), max(0, x1 - tile); t = rgb[y0c:y1, x0c:x1]; th, tw = t.shape[:2]
            wt = np.outer(_feather(th, y0c == 0, y1 == h, overlap), _feather(tw, x0c == 0, x1 == w, overlap))[..., None].astype(np.float32)
            if need is not None and not need(y0c, y1, x0c, x1): out[y0c:y1, x0c:x1] += t * wt; wsum[y0c:y1, x0c:x1] += wt; continue
            ph, pw = (-th) % 64, (-tw) % 64; tp = np.pad(t, ((0, ph), (0, pw), (0, 0)), mode='reflect') if ph or pw else t
            x = torch.from_numpy(np.ascontiguousarray(tp.transpose(2, 0, 1)))[None].to(dev)
            with torch.no_grad(): y = m(x.half() if dev.type == 'cuda' else x).float()[0].cpu().numpy().transpose(1, 2, 0)[:th, :tw]
            out[y0c:y1, x0c:x1] += y * wt; wsum[y0c:y1, x0c:x1] += wt
    return out / np.maximum(wsum, 1e-6)


def weight_map(sigma8, lo=1.5, hi=5.0, top=0.85):
    """How much of the cleaned picture to use at each pixel, from the noise expected there in 8-bit grey levels: nothing below `lo` (invisible: leave the texture), `top` (not 1: a little of the original grain stays, which looks natural) from `hi`."""
    return (np.clip((sigma8 - lo) / (hi - lo), 0, 1) * top).astype(np.float32)


def adaptive(rgb, sigma8):
    """`rgb` cleaned where its noise (`sigma8`, a per-pixel map or a number, in 8-bit grey levels) is visible, left alone where it is not."""
    w = weight_map(np.asarray(sigma8, np.float32)); w = w if w.ndim == 3 else (w[..., None] if w.ndim == 2 else w)
    if float(np.max(w)) <= 0: return rgb
    den = keep_tones(rgb, denoise(rgb, need=lambda y0, y1, x0, x1: float(w[y0:y1, x0:x1].max()) > 0.02))
    return rgb * (1 - w) + den * w


def keep_tones(orig, den, scale=8):
    """The model shifts broad brightness and colour a little (a sky comes out lighter): give `den` back the smooth colour and brightness of `orig`, keeping only what the model changed at fine scale (the difference of the two, smoothed on a 1/`scale` copy, is subtracted)."""
    import cv2
    h, w = orig.shape[:2]; small = lambda a: cv2.resize(a, (max(4, w // scale), max(4, h // scale)), interpolation=cv2.INTER_AREA)
    d = cv2.resize(cv2.GaussianBlur(small(den) - small(orig), (0, 0), 1.5), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(den - d, 0, 1)
