"""Learned denoising on the GPU, with the models strata-360 compared (docs/progress/2026-10-music-upscale.md there), chosen here by eye on drone crops:

  esrgan  Real-ESRGAN `realesr-general-x4v3` (BSD-3-Clause) used as a SUPERSAMPLED cleaner: each tile is enlarged 4x by the model and averaged back down, which removes noise and keeps (the user's words: "left texture in") the fine texture. `strength` blends the weights of the
          general model (0: keeps most grain and texture) with its `wdn` (weak-denoise-noise-trained) sibling (1: cleanest), as the Real-ESRGAN `--denoise_strength` option does. The default.
  scunet  SCUNet `scunet_color_real_psnr` (Apache-2.0): the cleanest, and too smooth for these pictures (it takes roof-tile and brick texture with the grain); kept as an option.

Both are loaded with spandrel and run in tiles on the GPU (in half precision where the card is fast at it, see `_fp16`). `adaptive` blends the cleaned picture into the original by a weight map from the noise expected at each pixel (develop.noise_map8: the file's NoiseProfile), so areas whose noise is below what the eye sees are left exactly as the sensor recorded them,
and the model's slight shifts of broad brightness and colour are taken back (`keep_tones`). Weights live in `models/` (gitignored): `fetch()` downloads what a model needs."""
import os, urllib.request

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
FILES = {
    'scunet': [('scunet/scunet_color_real_psnr.pth', 'https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth')],
    'esrgan': [('realesrgan/realesr-general-x4v3.pth', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'),
               ('realesrgan/realesr-general-wdn-x4v3.pth', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth')],
}
MODELS = tuple(FILES)
DEFAULT = 'esrgan'
DEFAULT_STRENGTH = 0.6          # esrgan: how far towards the stronger-denoise weights (0..1)
TILE = {'scunet': 1024, 'esrgan': 768}      # esrgan: 768 px tiles took 9 % less time than 512 on a GTX 1050 (less overlap to redo); 1024 ran out of its memory, so an out-of-memory halves the tile (`denoise`)
OVERLAP = {'scunet': 64, 'esrgan': 32}
_models = {}


def paths(model): return [os.path.join(ROOT, f) for f, _ in FILES[model]]
def available(model=DEFAULT): return all(os.path.exists(p) for p in paths(model))


def fetch(model=DEFAULT):
    for (f, url), p in zip(FILES[model], paths(model)):
        if not os.path.exists(p): os.makedirs(os.path.dirname(p), exist_ok=True); urllib.request.urlretrieve(url, p)
    return paths(model)


def _device():
    from stratastills import gpu; torch = gpu.torch_()                                                      # the first use of torch settles whether cuDNN works on this card
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _fp16(dev):
    """Whether the network runs in half precision: on a CUDA card with fast half arithmetic (compute capability 7 and up). A Pascal card such as the GTX 10xx runs half at a fraction of its float32 speed (measured on a GTX 1050: 1.39 s per 512 px tile against 1.09 s in float32)."""
    if dev.type != 'cuda': return False
    import torch
    try: return torch.cuda.get_device_capability(dev)[0] >= 7
    except Exception: return True


def _load(model, strength):
    key = (model, round(float(strength), 2) if model == 'esrgan' else 0)
    if key not in _models:
        import torch
        from spandrel import ModelLoader
        dev = _device()
        if model == 'esrgan':
            a, b = (torch.load(p, map_location='cpu') for p in paths(model)); a, b = a.get('params', a), b.get('params', b)
            net = ModelLoader().load_from_state_dict({k: (1 - strength) * a[k] + strength * b[k] for k in a}).model
        else: net = ModelLoader().load_from_file(paths(model)[0]).model
        net = net.eval().to(dev); _models[key] = net.half() if _fp16(dev) else net
    return _models[key]


def _run(net, tile, multiple, post):
    """One tile (float32 HxWx3, 0..1) through the network: padded to a multiple of `multiple` (reflect), run, `post` (network output, padded shape -> cleaned), cropped back."""
    import torch
    dev = next(net.parameters()).device; th, tw = tile.shape[:2]; ph, pw = (-th) % multiple, (-tw) % multiple
    tp = np.pad(tile, ((0, ph), (0, pw), (0, 0)), mode='reflect') if ph or pw else tile
    x = torch.from_numpy(np.ascontiguousarray(tp.transpose(2, 0, 1)))[None].to(dev)
    with torch.no_grad(): y = net(x.to(next(net.parameters()).dtype)).float()[0].cpu().numpy().transpose(1, 2, 0)
    return post(y, tp.shape)[:th, :tw]


def _feather(n, at_start, at_end, overlap):
    """Blend weights along one side of a tile: 1 in the middle, ramping to ~0 across the overlap where a neighbouring tile takes over (and 1 right to the edge of the picture)."""
    w = np.ones(n, np.float32); k = min(overlap, n // 2)
    if not at_start: w[:k] = (np.arange(k) + 0.5) / k
    if not at_end: w[n - k:] = np.minimum(w[n - k:], (np.arange(k)[::-1] + 0.5) / k)
    return w


def denoise(rgb, model=DEFAULT, strength=DEFAULT_STRENGTH, tile=None, overlap=None, need=None):
    """The cleaned copy of `rgb` (float32 HxWx3, 0..1, display-encoded: both models are trained on gamma-encoded pictures). `need(y0, y1, x0, x1)` (optional) says whether a tile has to be cleaned at all; a tile that does not is passed through (saves most of the time when only some areas are noisy)."""
    import cv2
    if model not in MODELS: raise ValueError(f'unknown denoise model {model!r} (one of {", ".join(MODELS)})')
    net = _load(model, strength); tile = tile or TILE[model]; overlap = overlap if overlap is not None else OVERLAP[model]
    if model == 'esrgan': post, mult = (lambda y, shape: cv2.resize(y, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)), 4                      # x4 from the network, averaged back down: the supersampled clean
    else: post, mult = (lambda y, shape: y), 64
    while True:
        try: out, wsum = _tiles(rgb, net, tile, overlap, mult, post, need); break
        except Exception as e:                                                                        # the card (shared with the desktop) cannot hold a tile this size: half the side, and again
            from stratastills import gpu
            if not gpu.is_oom(e) or tile <= 256: raise
            try: gpu.free()
            except RuntimeError: pass                                                                      # no device to free (the model ran on the CPU)
            tile //= 2
    return np.clip(out / np.maximum(wsum, 1e-6), 0, 1)


def _tiles(rgb, net, tile, overlap, mult, post, need):
    h, w = rgb.shape[:2]; out = np.zeros_like(rgb); wsum = np.zeros((h, w, 1), np.float32); step = tile - 2 * overlap
    for y0 in range(0, max(h - overlap, 1), step):
        for x0 in range(0, max(w - overlap, 1), step):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w); y0c, x0c = max(0, y1 - tile), max(0, x1 - tile); t = rgb[y0c:y1, x0c:x1]; th, tw = t.shape[:2]
            wt = np.outer(_feather(th, y0c == 0, y1 == h, overlap), _feather(tw, x0c == 0, x1 == w, overlap))[..., None].astype(np.float32)
            y = t if need is not None and not need(y0c, y1, x0c, x1) else _run(net, t, mult, post)
            out[y0c:y1, x0c:x1] += y * wt; wsum[y0c:y1, x0c:x1] += wt
    return out, wsum


def weight_map(sigma8, lo=1.5, hi=5.0, top=0.9):
    """How much of the cleaned picture to use at each pixel, from the noise expected there in 8-bit grey levels: nothing below `lo` (invisible: leave the texture), `top` (not 1: a little of the original grain stays, which looks natural) from `hi`."""
    return (np.clip((sigma8 - lo) / (hi - lo), 0, 1) * top).astype(np.float32)


def keep_tones(orig, den, scale=8):
    """The models shift broad brightness and colour a little (a sky comes out lighter): give `den` back the smooth colour and brightness of `orig`, keeping only what the model changed at fine scale (the difference of the two, smoothed on a 1/`scale` copy, is subtracted)."""
    import cv2
    h, w = orig.shape[:2]; small = lambda a: cv2.resize(a, (max(4, w // scale), max(4, h // scale)), interpolation=cv2.INTER_AREA)
    d = cv2.resize(cv2.GaussianBlur(small(den) - small(orig), (0, 0), 1.5), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(den - d, 0, 1)


def adaptive(rgb, sigma8, model=DEFAULT, strength=DEFAULT_STRENGTH):
    """`rgb` cleaned where its noise (`sigma8`, a per-pixel map or a number, in 8-bit grey levels) is visible, left alone where it is not."""
    w = weight_map(np.asarray(sigma8, np.float32)); w = w if w.ndim == 3 else (w[..., None] if w.ndim == 2 else w)
    if float(np.max(w)) <= 0: return rgb
    den = keep_tones(rgb, denoise(rgb, model, strength, need=lambda y0, y1, x0, x1: float(w[y0:y1, x0:x1].max()) > 0.02))
    return rgb * (1 - w) + den * w
