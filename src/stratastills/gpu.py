"""The pixel pipeline on the GPU (PyTorch): the same recipes as develop.py and detail.py, written for large tensors, so a 48 MP raw develops in seconds instead of minutes.

Devices, in the order tried: NVIDIA CUDA, Apple silicon MPS (Metal), and otherwise NOTHING: with neither, `available()` is False and the callers use the CPU code (NumPy and OpenCV in develop.py and detail.py), which is the REFERENCE every function here is tested against (tests/unit/test_gpu.py). The tests also run these torch
functions on the torch CPU device (`use('cpu')`), so a machine with no GPU still checks them. STRATASTILLS_DEVICE=cuda|mps|cpu forces a device (cpu = torch on the CPU, for tests); STRATASTILLS_GPU=0 switches the GPU path off.
Conventions: a picture is a float32 tensor (3, H, W) on the device (channels first, so a filter is a convolution over planes), one plane is (H, W); values as in the CPU code. Borders reflect without repeating the edge pixel (OpenCV's default, torch 'reflect').
Memory: a 48 MP plane is 193 MB and a picture 580 MB; the functions keep a handful alive at a time and the long per-pixel table lookups are chunked, so the whole development stays under about 6 GB (a 12 GB card, or an M4's unified memory, is plenty).
Operations a backend lacks are probed once (`cubic_ok`) and have a fallback, so Apple's Metal backend, which has had gaps in torch, degrades to slower rather than failing. Only float32 is used (MPS has no float64)."""
import contextlib, math, os, sys, threading

import numpy as np

_torch = None
_device = None                              # None = not chosen yet, False = no GPU, else a torch.device
_cubic_ok = None
LUMA = (0.2126, 0.7152, 0.0722)


def torch_():
    global _torch
    if _torch is None:
        import torch
        _torch = torch; _cudnn_probe(torch)
    return _torch


def _cudnn_probe(torch):
    """Where cuDNN has no engine for a plain convolution (found on a Pascal card, GTX 1050, with the cuDNN 9.10 that ships with torch 2.9 cu126: even a 1x9 float32 kernel fails), turn cuDNN off for the process: PyTorch's own kernels do the same convolutions with the same result, and every pixel and model step here is a convolution."""
    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled: return
    try:
        for dt in (torch.float32, torch.float16): torch.nn.functional.conv2d(torch.zeros(1, 1, 16, 16, device='cuda', dtype=dt), torch.ones(1, 1, 1, 9, device='cuda', dtype=dt))
        torch.cuda.synchronize()
    except (RuntimeError, AttributeError): torch.backends.cudnn.enabled = False


def _choose():
    t = torch_(); want = os.environ.get('STRATASTILLS_DEVICE', '').lower()
    if want in ('cpu', 'cuda', 'mps'): return t.device(want)
    if os.environ.get('STRATASTILLS_GPU', '1') == '0': return False
    try:
        if t.cuda.is_available(): return t.device('cuda')
        if getattr(t.backends, 'mps', None) is not None and t.backends.mps.is_available(): return t.device('mps')
    except Exception: pass
    return False


def use(name):
    """Force the device ('cuda', 'mps', 'cpu' = torch on the CPU, or None to choose again); used by the tests."""
    global _device, _cubic_ok
    _device = None if name is None else torch_().device(name); _cubic_ok = None


def dev():
    """The torch device in use."""
    global _device
    if _device is None:
        try: _device = _choose()
        except Exception: _device = False
    if _device is False: raise RuntimeError('no GPU device')
    return _device


def allocations():
    """How many buffers have been allocated on the card so far (CUDA; 0 before it is used), for telling whether a stretch of work used the GPU: the number goes up when it did. None where the backend cannot say (MPS)."""
    t = sys.modules.get('torch')
    if t is None: return 0
    try:
        if _device is not False and _device is not None and _device.type == 'mps': return None
        return int(t.cuda.memory_stats().get('allocation.all.allocated', 0)) if t.cuda.is_available() and t.cuda.is_initialized() else 0
    except Exception: return None


def available():
    """True when there is a CUDA or MPS device to use (and the GPU path is not switched off): what the callers ask before choosing this path over the CPU code."""
    global _device
    if _device is None:
        try: _device = _choose()
        except Exception: _device = False
    return _device is not False and _device.type in ('cuda', 'mps')


def cubic_ok():
    """Whether grid_sample does bicubic on this device (CUDA and CPU yes; Metal has lacked it): probed once with a tiny picture; the lens warp falls back to OpenCV on the CPU if not."""
    global _cubic_ok
    if _cubic_ok is None:
        t = torch_()
        try:
            x = t.zeros(1, 1, 4, 4, device=dev()); gr = t.zeros(1, 2, 2, 2, device=dev()); t.nn.functional.grid_sample(x, gr, mode='bicubic', padding_mode='reflection', align_corners=False); _cubic_ok = True
        except Exception: _cubic_ok = False
    return _cubic_ok


PEAK_BYTES = 6.5e9              # what one 48 MP develop needs on the device at its peak (measured 5.7 GB)
_slots = None
PEAK_PIXELS = 48.8e6           # the size of the picture PEAK_BYTES was measured on


TILE_BYTES_PER_PX = 190         # card memory per pixel of a tile with its margin (measured: 2048 px tiles + 128 margin = 0.91 GB, 1024: 0.29, 512: 0.11)
TILE_PEAK_BYTES = 1.2e9         # what one picture needs on the card when it goes through in the largest tiles (0.91 GB measured, with room)
_tiled_batch = False


def _cuda_total():
    """The memory of a CUDA card this program may plan with, or None for another device: the card's total, but no more than what is free plus what this process already holds (a card shared with the desktop or another program has far less free than its total: a 2 GB GTX 1050 with the desktop on it has about 1.6 GB, and plans made on 2 GB ran out of memory)."""
    try:
        d = dev()
        if d.type == 'cuda':
            t = torch_(); free, total = t.cuda.mem_get_info(); return min(total, free + t.cuda.memory_reserved())
    except Exception: pass
    return None


def is_oom(e):
    """Whether the exception is the device running out of memory (CUDA, or the allocator of MPS)."""
    return type(e).__name__ == 'OutOfMemoryError' or (isinstance(e, RuntimeError) and 'out of memory' in str(e).lower())


def whole_slots():
    """How many whole pictures (PEAK_BYTES each) the device holds at once: 90 % of a CUDA card's memory over that, one for MPS (it shares the machine's memory) and CPU; at least 1."""
    total = _cuda_total(); return max(1, int(total * 0.9 // PEAK_BYTES)) if total else 1


def tile_slots():
    """How many pictures may be on the device at once while they go through in tiles: what fits in a CUDA card's memory (90 % of it over TILE_PEAK_BYTES), one for MPS (two threads at once crashed it) and for CPU."""
    total = _cuda_total()
    if total: return max(1, int(total * 0.9 // TILE_PEAK_BYTES))
    return 1                                                                                                           # MPS too: two threads running kernels on it at once abort the process (torch's MPS backend; macOS CI, test_pipeline's batch: 'Fatal Python error: Aborted' in two sharpening threads), so it takes one picture at a time like the CPU


def tile_size(halo=128):
    """The tile side to use on this device: the largest (multiple of 64, 256 to 2048) whose region fits in the card's memory share of one picture in the batch (90 % of it over the slots in use), so a small card gets small tiles (1024 px tiles cost 11 % more time than 2048, 512 80 % more); MPS shares the machine's memory (1024); 2048 otherwise. STRATASTILLS_TILE overrides."""
    from stratastills import resources
    if os.environ.get('STRATASTILLS_TILE'): return max(64, int(os.environ['STRATASTILLS_TILE']))
    total = _cuda_total()
    if total: return resources.tile_for(0, TILE_BYTES_PER_PX, 0.0, halo, budget_bytes=total * 0.9 / max(1, slots()))
    try: return 1024 if dev().type == 'mps' else resources.MAX_TILE
    except Exception: return resources.MAX_TILE


def tiling_pays(workers):
    """Whether a batch of `workers` pictures at once is better made in tiles: when the card cannot hold that many whole pictures (measured on a 12 GB card, 4 at once in tiles was 11 % faster than the best whole-picture batch, 2 at once was not, and a lone picture is 25 % slower in tiles). Needs a device whose memory is the limit (CUDA, MPS)."""
    try: return dev().type in ('cuda', 'mps') and workers > whole_slots() and tile_slots() > whole_slots()
    except Exception: return False


def plan_workers():
    """How many pictures to make at once on this device: as many as the card holds in tiles plus one (the extra one is being decoded or encoded on the CPU while the others are on the card), at most 4, when that beats whole pictures; else 2 (whole pictures, a second one overlapping the first's CPU parts)."""
    n = min(4, tile_slots() + 1)
    return n if tiling_pays(n) else 2


@contextlib.contextmanager
def batch(workers):
    """For a batch of `workers` pictures at once: whether the pictures go through in tiles (tiling_pays) is set for all of them, and the slots are those of the mode."""
    global _tiled_batch
    old = _tiled_batch; _tiled_batch = tiling_pays(workers)
    try: yield _tiled_batch
    finally: _tiled_batch = old


def max_pixels():
    """The largest picture (pixels) one develop does whole on the device (a bigger one is developed in tiles, gpu_tiled.py), or None when the device's memory is not the limit (CPU, MPS): 90 % of a CUDA card's memory over the peak bytes per pixel. In a batch that goes through in tiles (`batch`) every picture counts as too big. STRATASTILLS_MAX_PIXELS overrides (and is the way to measure tiling on pictures that would fit)."""
    if os.environ.get('STRATASTILLS_MAX_PIXELS'): return int(os.environ['STRATASTILLS_MAX_PIXELS'])
    if _tiled_batch: return 1
    total = _cuda_total(); return int(total * 0.9 / PEAK_BYTES * PEAK_PIXELS) if total else None


_slots_lock = threading.Lock()


def _semaphore(tiled=None):
    """The device's slot semaphore for the mode (whole pictures or tiles; the batch's by default), made once per mode (under a lock: several threads ask at the start of a batch, and two semaphores would make a release go to the wrong one)."""
    global _slots
    tiled = _tiled_batch if tiled is None else tiled
    with _slots_lock:
        if _slots is None: _slots = {}
        if tiled not in _slots:
            n = os.environ.get('STRATASTILLS_GPU_SLOTS'); k = max(1, int(n)) if n else tile_slots() if tiled else whole_slots()
            _slots[tiled] = threading.BoundedSemaphore(k); _slots[tiled].n = k
        return _slots[tiled]


def slots():
    """How many pictures may be on the device at once in the current mode (whole pictures or the batch's tiles); STRATASTILLS_GPU_SLOTS overrides."""
    return _semaphore().n


@contextlib.contextmanager
def slot():
    """Hold one of the device's slots while a picture is on it: with several pictures being made at once, their CPU parts (decode, encode, files) overlap and only as many pictures as fit in memory use the card together; the rest wait here."""
    s = _semaphore(); s.acquire()
    try: yield
    finally: s.release()                                                                                              # the same semaphore that was acquired, whatever happens to the module's variable meanwhile


def free():
    t = torch_(); d = dev()
    try:
        if d.type == 'cuda': t.cuda.empty_cache()
        elif d.type == 'mps': t.mps.empty_cache()
    except Exception: pass


def to_gpu(a):
    """A numpy array (H, W, 3) or (H, W) as a float32 tensor (3, H, W) or (H, W) on the device."""
    t = torch_().from_numpy(np.ascontiguousarray(a)).to(dev(), non_blocking=True).float()
    return t.permute(2, 0, 1).contiguous() if t.dim() == 3 else t


def to_gpu_u16(a):
    """A uint16 numpy array (H, W, 3) (what LibRaw gives) as a float32 tensor (3, H, W) on the device scaled to 0..1, without first making a float copy on the CPU."""
    t = torch_(); x = t.from_numpy(np.ascontiguousarray(a).view(np.int16)).to(dev()).float(); x = t.where(x < 0, x + 65536.0, x) / 65535.0
    return x.permute(2, 0, 1).contiguous()


def to_numpy(t):
    """A (3, H, W) tensor as float32 numpy (H, W, 3); (H, W) stays (H, W)."""
    return (t.permute(1, 2, 0) if t.dim() == 3 else t).contiguous().cpu().numpy()


def luma(img):
    """Luminance plane of a (3, H, W) picture."""
    return img[0] * LUMA[0] + img[1] * LUMA[1] + img[2] * LUMA[2]


def smoothstep(x):
    x = x.clamp(0, 1); return x * x * (3 - 2 * x)


# ---- filters --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _pad(y, r):
    """Reflect-pad a (N, C, H, W) tensor by r on every side; where r is not smaller than the picture (a tiny one), repeat the edge instead: torch cannot reflect that far, OpenCV can, and nothing that small matters."""
    F = torch_().nn.functional; return F.pad(y, (r, r, r, r), mode='reflect' if r < min(y.shape[-2:]) else 'replicate')


def box(x, r):
    """Mean over a (2r+1) square of the plane `x` (H, W): two one-dimensional averages, edges reflected (cv2.boxFilter)."""
    F = torch_().nn.functional; k = 2 * r + 1; y = _pad(x[None, None], r); y = F.avg_pool2d(y, (1, k), stride=1); return F.avg_pool2d(y, (k, 1), stride=1)[0, 0]


def gaussian(x, sigma):
    """Gaussian blur of a plane (H, W) or a stack (N, H, W) with standard deviation `sigma` pixels, a kernel of +-4 sigma (what cv2.GaussianBlur uses for float images), edges reflected."""
    t = torch_(); F = t.nn.functional; r = int(math.ceil(sigma * 4)); xs = t.arange(-r, r + 1, device=x.device, dtype=t.float32); k = t.exp(-0.5 * (xs / sigma) ** 2); k = k / k.sum()
    y = x[:, None] if x.dim() == 3 else x[None, None]; y = _pad(y, r); y = conv(y, k.view(1, 1, 1, -1)); y = conv(y, k.view(1, 1, -1, 1))
    return y[:, 0] if x.dim() == 3 else y[0, 0]


CONV_WORKSPACE = 96 << 20       # bytes: what one strip of a convolution may take for its working buffer (below)


def conv(y, k):
    """F.conv2d of the kernel `k` (1, 1, kh, kw) over `y` (N, 1, H, W) (no padding), done in strips of rows (with the kernel's overlap) so that the working buffer stays small: the answer is the same. Where cuDNN is off (a Pascal card, see _choose) a convolution builds a buffer of kernel-size times the input: a sigma 8 blur of a 2304 px tile asked for 1.25 GB, and the 3x3 gradient of a 21 MP picture 0.7 GB, out of memory on a 2 GB card (the whole development then ran twice, and the sharpening went to the CPU)."""
    F = torch_().nn.functional; kh, kw = k.shape[-2:]; out_h = y.shape[2] - kh + 1; step = max(8, CONV_WORKSPACE // max(1, kh * kw * 4 * y.shape[0] * y.shape[3]))
    if step >= out_h: return F.conv2d(y, k)
    return torch_().cat([F.conv2d(y[:, :, i:i + step + kh - 1], k) for i in range(0, out_h, step)], dim=2)


def guided(p, guide, r, eps):
    """He et al.'s guided filter, as develop._guided."""
    mg, mp = box(guide, r), box(p, r); cov = box(guide * p, r) - mg * mp; var = box(guide * guide, r) - mg * mg; a = cov / (var + eps); b = mp - a * mg
    return box(a, r) * guide + box(b, r)


def resize_area(x, size):
    """Plane (H, W) reduced to `size` = (h, w) by area averaging (cv2.INTER_AREA). On MPS the adaptive pooling behind it handles only sizes that divide evenly; for the others the (small) reduction is done on the CPU."""
    F = torch_().nn.functional
    if x.device.type == 'mps' and (x.shape[-2] % size[0] or x.shape[-1] % size[1]): return F.interpolate(x.cpu()[None, None], size=size, mode='area')[0, 0].to(x.device)
    return F.interpolate(x[None, None], size=size, mode='area')[0, 0]


def resize_linear(x, size, mode='bilinear'):
    """Plane (H, W) enlarged to `size` = (h, w): 'bilinear' (cv2.INTER_LINEAR) or 'bicubic' (cv2.INTER_CUBIC), half-pixel centres."""
    return torch_().nn.functional.interpolate(x[None, None], size=size, mode=mode, align_corners=False)[0, 0]


def apply_lut(x, lut, chunk=1 << 24):
    """`x` (any shape, values 0..1) through the lookup table `lut` (n,) with linear interpolation (np.interp over linspace(0, 1, n)); done in pieces so the index tensors stay small."""
    t = torch_(); n = lut.numel(); flat = x.reshape(-1); out = t.empty_like(flat)
    for i in range(0, flat.numel(), chunk):
        v = flat[i:i + chunk] * (n - 1); i0 = v.floor().clamp_(0, n - 2); f = v - i0; i0 = i0.long(); out[i:i + chunk] = lut[i0] * (1 - f) + lut[i0 + 1] * f
    return out.reshape(x.shape)


# ---- the development ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def apply_gain_map(img, m, plane, planes):
    """develop.apply_gain_map: `m` is the map (rows, cols, planes) as numpy; returns (img, field) with the field of the second plane (the green one) as a plane tensor."""
    t = torch_(); h, w = img.shape[1:]; field = None
    for k in range(planes):
        g = resize_linear(t.from_numpy(np.ascontiguousarray(m[:, :, min(k, m.shape[2] - 1)])).to(img.device), (h, w)); img[plane + k] *= g
        if k == min(1, planes - 1): field = g
    return img, field


def warp_rectilinear(img, coef, centre):
    """develop.warp_rectilinear with bicubic sampling by grid_sample (the same kernel as cv2.INTER_CUBIC), per colour plane."""
    t = torch_(); F = t.nn.functional; c3, h, w = img.shape
    if not cubic_ok():                                                                                                # a backend without bicubic grid_sample: the CPU does the warp
        from stratastills import develop
        return gpu_from_numpy_chw(develop.warp_rectilinear(to_numpy(img), coef, centre), img.device)
    cx, cy = centre[0] * (w - 1), centre[1] * (h - 1); mr = float(np.hypot(max(cx, w - 1 - cx), max(cy, h - 1 - cy)))
    yy = t.arange(h, device=img.device, dtype=t.float32).view(h, 1); xx = t.arange(w, device=img.device, dtype=t.float32).view(1, w); x = (xx - cx) / mr; y = (yy - cy) / mr; r2 = x * x + y * y; out = t.empty_like(img)
    for c in range(c3):
        kr0, kr1, kr2, kr3, kt0, kt1 = (float(v) for v in coef[min(c, len(coef) - 1)])
        f = kr0 + r2 * (kr1 + r2 * (kr2 + r2 * kr3)); xs = x * f + kt0 * 2 * x * y + kt1 * (r2 + 2 * x * x); ys = y * f + kt1 * 2 * x * y + kt0 * (r2 + 2 * y * y)
        grid = t.stack([((xs * mr + cx) + 0.5) / w * 2 - 1, ((ys * mr + cy) + 0.5) / h * 2 - 1], dim=-1)[None]; del f, xs, ys
        out[c] = F.grid_sample(img[c][None, None], grid, mode='bicubic', padding_mode='reflection', align_corners=False)[0, 0]; del grid
    return out


def gpu_from_numpy_chw(a, device):
    t = torch_().from_numpy(np.ascontiguousarray(a)).to(device).float(); return t.permute(2, 0, 1).contiguous() if t.dim() == 3 else t


def sample_grid(small, ys, xs, h, w, mode='bilinear'):
    """The plane `small` (sh, sw) enlarged to (h, w) as resize_linear does ('bilinear' or 'bicubic', half-pixel centres, edge values held), evaluated only at the pixel rows `ys` and columns `xs` (index tensors/arrays of the full-size picture): (len(ys), len(xs)). What a tile of the enlarged plane is, without making the whole of it."""
    t = torch_(); F = t.nn.functional; d = small.device; gy = ((t.as_tensor(np.asarray(ys), device=d).float() + 0.5) / h * 2 - 1).view(-1, 1).expand(-1, len(xs)); gx = ((t.as_tensor(np.asarray(xs), device=d).float() + 0.5) / w * 2 - 1).view(1, -1).expand(len(ys), -1)
    return F.grid_sample(small[None, None], t.stack([gx, gy], -1)[None], mode=mode, padding_mode='border', align_corners=False)[0, 0]


def warp_source_box(coef, centre, h, w, box, margin=4):
    """The part of the source picture (y0, y1, x0, x1) that warp_rectilinear_box needs to make the output `box`: the box's sample positions (computed on the card) plus a margin for the bicubic kernel, clipped to the picture."""
    t = torch_(); y0, y1, x0, x1 = box; d = dev(); xs_, ys_ = _warp_coords(coef, centre, h, w, box, d); lo = lambda a: max(0, int(a.min().floor()) - margin); hi = lambda a, n: min(n, int(a.max().ceil()) + margin + 1)
    return min(lo(ys_[c]) for c in range(len(ys_))), max(hi(ys_[c], h) for c in range(len(ys_))), min(lo(xs_[c]) for c in range(len(xs_))), max(hi(xs_[c], w) for c in range(len(xs_)))


def _warp_coords(coef, centre, h, w, box, device, planes=3):
    """The source positions (x, y in pixels of the full picture) each output pixel of `box` samples, per colour plane: (list of xs, list of ys), each (bh, bw): develop.warp_rectilinear's mapping."""
    t = torch_(); y0, y1, x0, x1 = box; cx, cy = centre[0] * (w - 1), centre[1] * (h - 1); mr = float(np.hypot(max(cx, w - 1 - cx), max(cy, h - 1 - cy)))
    yy = t.arange(y0, y1, device=device, dtype=t.float32).view(-1, 1); xx = t.arange(x0, x1, device=device, dtype=t.float32).view(1, -1); x = (xx - cx) / mr; y = (yy - cy) / mr; r2 = x * x + y * y; XS, YS = [], []
    for c in range(planes):
        kr0, kr1, kr2, kr3, kt0, kt1 = (float(v) for v in coef[min(c, len(coef) - 1)]); f = kr0 + r2 * (kr1 + r2 * (kr2 + r2 * kr3))
        XS.append((x * f + kt0 * 2 * x * y + kt1 * (r2 + 2 * x * x)) * mr + cx); YS.append((y * f + kt1 * 2 * x * y + kt0 * (r2 + 2 * y * y)) * mr + cy)
    return XS, YS


def warp_rectilinear_box(src, src_box, coef, centre, h, w, box):
    """warp_rectilinear (bicubic) of the full (h, w) picture, computed only for the output `box`, from `src` (3, sh, sw): the part `src_box` (y0, y1, x0, x1) of the picture (warp_source_box says which). Edge reflection is that of the picture only where src_box reaches its edge, as it does for the pixels near it."""
    t = torch_(); F = t.nn.functional; sy0, sy1, sx0, sx1 = src_box; sh, sw = sy1 - sy0, sx1 - sx0; XS, YS = _warp_coords(coef, centre, h, w, box, src.device); out = t.empty((3, box[1] - box[0], box[3] - box[2]), device=src.device, dtype=t.float32)
    for c in range(3):
        grid = t.stack([((XS[c] - sx0) + 0.5) / sw * 2 - 1, ((YS[c] - sy0) + 0.5) / sh * 2 - 1], dim=-1)[None]; out[c] = F.grid_sample(src[c][None, None], grid, mode='bicubic', padding_mode='reflection', align_corners=False)[0, 0]
    return out


def local_tone(img, strength, highlights, px=1200):
    """develop.local_tone: the edge-preserving base is computed on a reduced copy (the bilateral filter runs on the CPU, it is tiny)."""
    import cv2
    if strength <= 0: return img
    t = torch_(); h, w = img.shape[1:]; y = luma(img).clamp_(min=1e-5); ly = t.log2(y); del y
    k = min(1.0, px / max(h, w)); sz = (max(8, round(w * k)), max(8, round(h * k))); small = resize_area(ly, (sz[1], sz[0])).cpu().numpy()
    base_s = cv2.bilateralFilter(small, 9, 0.5, max(4.0, 0.012 * max(small.shape))); base = resize_linear(t.from_numpy(base_s).to(img.device), (h, w), 'bicubic')
    mid = _median(base[::8, ::8]); kk = t.where(base > mid, 1 - highlights * min(1.0, strength / 0.35), t.full_like(base, 1 - strength)); new = mid + (base - mid) * kk + (ly - base); del kk, base
    return img * t.exp2(new - ly)[None]


def _median(x):
    """Median of a tensor by sorting (torch.median and quantile are not available on every backend); the lower of the two middle values for an even count, like torch.median."""
    f = x.reshape(-1); return float(torch_().sort(f).values[(f.numel() - 1) // 2])


def quantile(x, q):
    """The value at fraction `q` (0..1) of the sorted values of a tensor (nearest rank), by sorting."""
    f = x.reshape(-1); return float(torch_().sort(f).values[min(f.numel() - 1, int(round(q * (f.numel() - 1))))])


def shoulder(img, knee, white):
    t = torch_(); m = img.amax(dim=0, keepdim=True); span = 1.0 - knee; x = (m - knee).clamp_(min=0) / span; w = (white - knee) / span
    new = t.where(m > knee, knee + span * x * (1 + x / (w * w)) / (1 + x), m); return img * (new / m.clamp(min=1e-6))


def srgb_oetf(x):
    t = torch_(); x = x.clamp(0, 1); return t.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1 / 2.4) - 0.055)


def srgb_eotf(v):
    t = torch_(); v = v.clamp(0, 1); return t.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055).pow(2.4))


def finish(img, lut, knee, white, blend=1.0):
    """develop.finish: shoulder, then the profile tone curve (lut a tensor) per channel (`blend` 1), on the brightest channel with the colour ratios kept (0), or a mix; or the sRGB curve."""
    img = shoulder(img.clamp(min=0), knee, white).clamp_(0, 1)
    if lut is None: return srgb_oetf(img)
    if blend >= 1: return apply_lut(img, lut)
    mx = img.amax(dim=0, keepdim=True); mn = img.amin(dim=0, keepdim=True); fx, fn = apply_lut(mx, lut), apply_lut(mn, lut); d = mx - mn
    ratio = fn + (fx - fn) * torch_().where(d > 1e-6, (img - mn) / d.clamp(min=1e-6), torch_().zeros_like(img))
    return ratio if blend <= 0 else blend * apply_lut(img, lut) + (1 - blend) * ratio


def look_to(look, device):
    """A look (develop.tone_parts: (table, hue/saturation table), either maybe None) as tensors on the device; None stays None."""
    if look is None: return None
    t = torch_(); return tuple(None if x is None else t.from_numpy(x).to(device) for x in look)


def hue_look(rgb, table):
    """develop.hue_look on a (3, H, W) tensor with the table as a tensor."""
    from stratastills import develop as D
    t = torch_(); dv = rgb.device; w, e1, e2, cs = (t.from_numpy(x).to(dv) for x in (D._W, D._E1, D._E2, D.LOOK_CHROMA)); h, nc = table.shape[0], len(D.LOOK_CHROMA)
    y = (rgb * w[:, None, None]).sum(0); d = rgb - y[None]; a = (d * e1[:, None, None]).sum(0); b = (d * e2[:, None, None]).sum(0); hue = t.atan2(b, a); ch = t.hypot(a, b)
    fh = (hue / (2 * np.pi)) % 1.0 * h; i0 = fh.floor().long() % h; i1 = (i0 + 1) % h; fh = fh - fh.floor()
    k = t.bucketize(ch, cs[1:-1]); fc = ((ch - cs[k]) / (cs[k + 1] - cs[k])).clamp(0, 1); j0 = k
    q = lambda i, j: table[i, j]; fhh = fh[..., None]; fcc = fc[..., None]
    tt = q(i0, j0) * (1 - fhh) * (1 - fcc) + q(i1, j0) * fhh * (1 - fcc) + q(i0, j0 + 1) * (1 - fhh) * fcc + q(i1, j0 + 1) * fhh * fcc
    ang = hue + tt[..., 0] * (np.pi / 180); r = ch * tt[..., 1]; d2 = (r * t.cos(ang))[None] * e1[:, None, None] + (r * t.sin(ang))[None] * e2[:, None, None]
    return (y[None] + d2).clamp_(0, 1)


def chroma_scale(rgb, look):
    """develop.chroma_scale on a (3, H, W) tensor with the look as tensors (look_to)."""
    if look is None: return rgb
    lut, tab = look
    if lut is not None: y = luma(rgb)[None]; p = y.clamp(0, 1) * (len(lut) - 1); i = p.floor().long().clamp(max=len(lut) - 2); f = p - i; rgb = (y + (rgb - y) * (lut[i] * (1 - f) + lut[i + 1] * f)).clamp_(0, 1)
    return rgb if tab is None else hue_look(rgb, tab)


def limit_chroma(rgb, knee, ceil):
    """develop.limit_chroma on a (3, H, W) tensor."""
    t = torch_()
    if knee <= 0 or ceil <= knee: return rgb
    y = luma(rgb)[None]; d = rgb - y; c = (d * d).sum(0, keepdim=True).sqrt() + 1e-9; room = ceil - knee; ex = (c - knee).clamp(min=0)
    new = t.where(c > knee, knee + room * ex / (ex + room), c); return (y + d * (new / c)).clamp_(0, 1)


def apply_hue_sat_map(rgb, table, rows=512):
    """develop.apply_hue_sat_map on a (3, H, W) tensor with the table as a (hue, saturation, 3) tensor, in bands of `rows` rows."""
    t = torch_(); hd, sd = table.shape[:2]; out = t.empty_like(rgb)
    for r0 in range(0, rgb.shape[1], rows):
        x = rgb[:, r0:r0 + rows].clamp(min=0); mx = x.amax(0); mn = x.amin(0); d = mx - mn; ds = t.where(d > 0, d, t.ones_like(d)); r, g, b = x[0], x[1], x[2]
        h = t.where(mx == r, ((g - b) / ds) % 6.0, t.where(mx == g, (b - r) / ds + 2.0, (r - g) / ds + 4.0)); h = t.where(d > 0, h, t.zeros_like(h)); s = t.where(mx > 0, d / t.where(mx > 0, mx, t.ones_like(mx)), t.zeros_like(mx))
        hs = h * (hd / 6.0); fl = hs.floor(); i0 = fl.long() % hd; fh = (hs - fl).unsqueeze(-1); i1 = (i0 + 1) % hd
        ss = s.clamp(0, 1) * (sd - 1); j0 = ss.floor().clamp(max=sd - 2).long(); fs = (ss - j0).unsqueeze(-1); j1 = j0 + 1
        dd = table[i0, j0] * (1 - fh) * (1 - fs) + table[i1, j0] * fh * (1 - fs) + table[i0, j1] * (1 - fh) * fs + table[i1, j1] * fh * fs
        h = (h + dd[..., 0] * (6.0 / 360.0)) % 6.0; s = (s * dd[..., 1]).clamp(max=1.0); v = mx * dd[..., 2]
        c = v * s; xx = c * (1 - (h % 2.0 - 1).abs()); m = v - c; k = h.floor().long() % 6; z = t.zeros_like(c)
        cand = t.stack([t.stack([c, xx, z]), t.stack([xx, c, z]), t.stack([z, c, xx]), t.stack([z, xx, c]), t.stack([xx, z, c]), t.stack([c, z, xx])])        # (6, 3, rows, W)
        out[:, r0:r0 + rows] = cand.gather(0, k[None, None].expand(1, 3, *k.shape))[0] + m
    return out


def blind_noise(rgb):
    """develop.blind_noise on a (3, H, W) tensor (a float: the noise in 8-bit grey levels)."""
    t = torch_(); F = t.nn.functional; y = (luma(rgb) * 255)[None, None]; k = t.tensor([[1., -2, 1], [-2, 4, -2], [1, -2, 1]], device=rgb.device)[None, None]
    pad = lambda a: F.pad(a[None], (1, 1, 1, 1), mode='replicate'); r = conv(pad(y[0]), k)[0, 0, 2:-2, 2:-2].abs(); s = gaussian(y[0], 2.0)
    sx = conv(pad(s), t.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=rgb.device)[None, None])[0, 0]; sy = conv(pad(s), t.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]], device=rgb.device)[None, None])[0, 0]
    g = t.hypot(sx, sy)[2:-2, 2:-2]; flat = g <= _quantile(g[::4, ::4], 0.25); return float(_quantile(r[flat], 0.5) / 0.6745 / 6)


def _quantile(x, q):
    x = x.reshape(-1); k = max(1, min(x.numel(), int(round(q * (x.numel() - 1))) + 1)); return x.kthvalue(k).values


def noise_map8(img, noise_profile, blur=8.0, gain=None):
    """develop.noise_map8: the noise expected at each pixel in 8-bit grey levels, as a plane. `gain` (a plane, the lens-shading field) multiplies the pixel's signal AND its noise beyond the global exposure gain the profile is scaled for: variance = S G v + O G^2."""
    t = torch_(); h, w = img.shape[1:]; sz = (max(8, round(h / 8)), max(8, round(w / 8))); v = gaussian(resize_area(luma(img), sz), blur / 8 * 2).clamp_(min=1e-5)
    slope = t.where(v < 0.0031308, t.full_like(v, 12.92), (1.055 / 2.4) * v.pow(1 / 2.4 - 1))
    if gain is None: s8 = 255.0 * t.sqrt(noise_profile[0] * v + noise_profile[1]) * slope
    else: gs = gaussian(resize_area(gain, sz), blur / 8 * 2); s8 = 255.0 * t.sqrt(noise_profile[0] * gs * v + noise_profile[1] * gs * gs) * slope
    return resize_linear(s8, (h, w))


def clean_colour(rgb, sig8, desat_from=0.86):
    """develop.clean_colour on a (3, H, W) picture with the noise map `sig8` (H, W)."""
    t = torch_(); y = luma(rgb); cb, cr = rgb[2] - y, rgb[0] - y; sg = sig8 / 255.0
    w = [((sg * 255 - lo) / (hi - lo)).clamp_(0, 1) for lo, hi in ((2.0, 6.0), (5.0, 11.0))]; out = []
    for c in (cb, cr):
        f = guided(c, y, 2, 4e-4) * (1 - w[0]) + (guided(c, y, 6, 1.5e-3) * (1 - w[1]) + guided(c, y, 14, 4e-3) * w[1]) * w[0]
        out.append(t.sign(f) * (f.abs() - 0.3 * sg).clamp_(min=0))
    roll = 1.0 - ((y - desat_from) / (1.0 - desat_from)).clamp_(0, 1) ** 2 * 0.9
    r = y + out[1] * roll; b = y + out[0] * roll; g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return t.stack([r, g, b]).clamp_(0, 1)


def core_luma(rgb, sig8, band_core, k=1.0):
    """develop.core_luma: the luminance grain cored band by band, thresholds following the noise map and the local brightness."""
    t = torch_(); y = luma(rgb); n = sig8 / 255.0; b = gaussian(y, 8.0); n = n * (0.7 + 0.4 * smoothstep((b - 0.55) / 0.30)); del b
    g = [y] + [gaussian(y, s) for s in (1.0, 2.0, 4.0)]; out = g[3].clone()
    for band, a in zip((g[0] - g[1], g[1] - g[2], g[2] - g[3]), band_core): out += t.sign(band) * (band.abs() - a * k * n).clamp_(min=0)
    return (rgb + (out - y)[None]).clamp_(0, 1)


def highlight_detail(rgb, amount, lo=0.62, hi=0.92):
    """develop.highlight_detail."""
    y = luma(rgb); b2 = guided(y, y, 2, 2e-4); b8 = guided(y, y, 8, 1e-3); f = 1 + amount * smoothstep((b8 - lo) / (hi - lo)); new = b8 + (b2 - b8) * f + (y - b2) * f
    return (rgb + (new - y)[None]).clamp_(0, 1)


# ---- sharpening (detail.py) -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _interp(x, xp, fp, left):
    """np.interp on tensors: `x` any shape, xp ascending (n,), fp (n,); below xp[0] the value `left`, above xp[-1] fp[-1]."""
    t = torch_(); idx = t.bucketize(x, xp).clamp(1, xp.numel() - 1); x0, x1, y0, y1 = xp[idx - 1], xp[idx], fp[idx - 1], fp[idx]
    v = y0 + ((x - x0) / (x1 - x0)).clamp(0, 1) * (y1 - y0); return t.where(x < xp[0], t.full_like(v, left), v)


def sharpen_gain(freq, mtf, shape, scale, noise, max_gain, amount, rolloff):
    """detail.sharpen_gain as the half spectrum (H, W//2 + 1) that rfft2 gives: the Wiener filter of the measured blur, normalised, held back at the top frequencies."""
    t = torch_(); d = dev(); h, w = shape; fy = t.fft.fftfreq(h, device=d).view(h, 1); fx = t.fft.rfftfreq(w, device=d).view(1, -1); r = t.sqrt(fx * fx + fy * fy)
    xp = t.from_numpy(np.asarray(freq, np.float32)).to(d); fp = t.from_numpy(np.asarray(mtf, np.float32)).to(d)
    m = _interp(r.clamp(max=0.5) * scale, xp, fp, 1.0).clamp_(0.02, 1.0); gain = (((1 + noise) * m / (m * m + noise)) ** amount).clamp_(max=max_gain); gain = t.where(r * scale < xp[0], t.ones_like(gain), gain)
    if rolloff: gain = 1 + (gain - 1) * (1 - (r - rolloff[0]) / (rolloff[1] - rolloff[0])).clamp_(0, 1)
    return gain


def activity(y, mask):
    """detail.activity on a (H, W) luminance tensor (None: everywhere)."""
    if mask is None: return torch_().ones_like(y)
    t = torch_(); F = t.nn.functional; s = gaussian(y[None] * 255, 1.0)[0]; kx = t.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=y.device) / 8
    gx = conv(F.pad(s[None, None], (1, 1, 1, 1), mode='replicate'), kx[None, None])[0, 0]; gy = conv(F.pad(s[None, None], (1, 1, 1, 1), mode='replicate'), kx.t()[None, None])[0, 0]
    return gaussian(((t.hypot(gx, gy) - mask[0]) / (mask[1] - mask[0])).clamp(0, 1)[None], 1.5)[0]


def sharpen_delta(rgb, gain, overshoot=0.05, mask=None):
    """The change detail.sharpen makes to the luminance, (H, W), for a (3, H, W) picture and a half-spectrum gain: the luminance (BT.601 weights, as there) is filtered by FFT and the change held within the local range of the original plus `overshoot`. Each intermediate is freed as soon as it is not needed (the peak is about 5 planes: 21 MP in 0.45 GB, where holding them all took 1.3 GB and ran a 2 GB card out of memory)."""
    t = torch_(); F = t.nn.functional; h, w = rgb.shape[1:]; y = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
    f = t.fft.rfft2(y); f *= gain; d = t.fft.irfft2(f, s=(h, w)); del f; d -= y; d *= activity(y, mask); d += y                          # d is now the filtered, masked luminance
    pad = lambda a: F.pad(a[None, None], (2, 2, 2, 2), mode='replicate')
    d = t.minimum(d, F.max_pool2d(pad(y), 5, stride=1)[0, 0] + overshoot); d = t.maximum(d, -F.max_pool2d(pad(-y), 5, stride=1)[0, 0] - overshoot); d -= y
    return d


def sharpen(rgb, gain, overshoot=0.05, mask=None):
    """detail.sharpen on a (3, H, W) picture with a half-spectrum gain: the change of `sharpen_delta` added equally to the three channels."""
    return (rgb + sharpen_delta(rgb, gain, overshoot, mask)[None]).clamp_(0, 1)