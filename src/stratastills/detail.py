"""How much real detail a picture holds, and so the largest size worth exporting it at.

A 48 MP sensor does not resolve 48 MP: the lens, the quad-Bayer re-mosaic, the demosaic and the maker's processing blur it. Pixels beyond what it resolves only enlarge the blur (bigger files, nothing more to see).
The measure is the sharpness (MTF, the contrast that is left at each spatial frequency) of the picture itself, found from the power spectrum of the textured parts of the luminance, which needs no test chart:

  1  tiles of TILE px with plenty of structure and no clipping are windowed and Fourier transformed; their power is averaged by radius, in cycles per pixel (0.5 = the finest the pixels can hold, Nyquist)
  2  natural scenes have power that falls as a power law, 1/f^a, so a straight line fitted to the low, un-blurred frequencies (FIT_BAND) predicts the spectrum of a perfectly sharp picture
  3  the noise floor (the flat level the spectrum comes down to at the highest frequencies) is taken off, and the ratio of what is left to the prediction is the MTF squared
  4  f50, f30 and f20 are the frequencies where the MTF falls to 50 %, 30 % and 20 %. The size at which f30 reaches the new Nyquist is 2 x f30 of the native size: that is the recommendation (RECOMMEND); the others are reported for judgement.

`effective_resolution(luma)` returns that as numbers (and the curve for plotting). Checked on synthetic pictures of known blur (tests/unit/test_detail.py)."""
import numpy as np

TILE = 256
FIT_BAND = (0.008, 0.035)
MAX_TILES = 240
MTF_LIMITS = (0.5, 0.3, 0.2)
RECOMMEND = 'scale30'            # the size to recommend: where the contrast has fallen to 30 %. At 20 % the faint tail near Nyquist is counted, which in a 100 % view is not visible detail (checked by eye on the Hazelmere DNGs: 0.75 x is indistinguishable from native); at 50 % the sharpness limit would throw real detail away.


def _tiles(y):
    h, w = y.shape; out = []
    for r in range(0, h - TILE + 1, TILE):
        for c in range(0, w - TILE + 1, TILE):
            t = y[r:r + TILE, c:c + TILE]; lo, hi = float(t.min()), float(t.max())
            if hi > 0.985 or lo < 0.015 or t.mean() < 0.08 or hi - lo < 0.04: continue
            gy, gx = np.gradient(t); out.append((float(np.mean(gx * gx + gy * gy)), r, c))
    out.sort(reverse=True)
    keep = out[:max(8, min(MAX_TILES, len(out) // 3))]
    return [y[r:r + TILE, c:c + TILE] for _, r, c in keep]


def radial_power(y):
    """(frequencies in cycles/px, mean power) of the textured tiles of `y` (2D float, gamma-encoded), or (None, None) if there are too few usable tiles."""
    ts = _tiles(y)
    if len(ts) < 6: return None, None
    win = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float32); norm = float((win ** 2).sum())
    fx = np.fft.fftfreq(TILE); r = np.hypot(*np.meshgrid(fx, fx)); nb = TILE // 2; idx = np.minimum((r / 0.5 * nb).astype(int), nb)
    keep = (idx < nb).ravel(); acc = np.zeros(nb); cnt = np.bincount(idx.ravel()[keep], minlength=nb)[:nb].astype(float)
    for t in ts:
        f = np.fft.fft2((t - t.mean()) * win); p = (np.abs(f) ** 2 / norm).ravel()
        acc += np.bincount(idx.ravel()[keep], weights=p[keep], minlength=nb)[:nb]
    freq = (np.arange(nb) + 0.5) / nb * 0.5
    return freq, acc / (len(ts) * np.maximum(cnt, 1))


def effective_resolution(y):
    """{ok, f50, f30, f20, scale50, scale30, scale20, alpha, noise_share, tiles, freq, mtf}: `scale30` is the fraction of the native size worth keeping (1.0 = the pixels really are all used). `y` is a 2D float luminance array, gamma-encoded, 0..1.
    ok is False (and the rest absent) when the picture has too little texture to measure (a clear sky, a blank wall)."""
    freq, p = radial_power(y)
    if freq is None: return dict(ok=False, reason='too little texture to measure')
    k = (freq >= FIT_BAND[0]) & (freq <= FIT_BAND[1]); a, b = np.polyfit(np.log(freq[k]), np.log(p[k]), 1)
    pred = np.exp(b) * freq ** a
    sm = np.convolve(np.pad(p, 2, mode='edge'), np.ones(5) / 5, mode='valid')                                    # smooth a little so the noise floor is not one lucky bin
    noise = float(sm[freq >= 0.3].min()); ratio = np.maximum(sm - noise, 1e-12 * pred.max()) / pred
    m = np.sqrt(ratio / max(float(ratio[k].mean()), 1e-12))                                                        # the fitted band is by definition not blurred: MTF 1 there
    lo = int(np.argmax(freq >= FIT_BAND[0])); m[:lo] = 1.0; m = np.minimum.accumulate(np.minimum(m, 1.0))              # contrast only falls with frequency (below the fitted band it is taken as 1)
    out = dict(ok=True, alpha=float(-a), tiles=int(len(_tiles(y))), noise_share=float(noise / max(float(sm[k].mean()), 1e-12)))
    for lim in MTF_LIMITS:
        below = np.nonzero(m < lim)[0]; f = float(np.interp(lim, m[below[0] - 1:below[0] + 1][::-1], freq[below[0] - 1:below[0] + 1][::-1])) if len(below) and below[0] > 0 else 0.5
        out[f'f{int(lim * 100)}'] = f; out[f'scale{int(lim * 100)}'] = float(min(1.0, 2 * f))
    out['freq'] = freq.tolist(); out['mtf'] = m.tolist()
    return out


def recommended_size(res, long_side, round_to=64):
    """The long side (pixels) to export at: the measured detail limit of the native `long_side`, rounded up to a multiple of `round_to` and never above native. None if it could not be measured."""
    if not res.get('ok'): return None
    return int(min(long_side, int(np.ceil(res[RECOMMEND] * long_side / round_to)) * round_to))


NOISE_TERM = 0.04               # the Wiener noise term: at 0.01 the gain climbed to its cap at the highest frequencies, where there is mostly noise (measured: finest-scale grain 2 to 3 times the unsharpened picture's and the camera JPEG's)
ROLLOFF = (0.2, 0.5)            # the boost fades to nothing between these OUTPUT frequencies (cycles per pixel; 0.5 is Nyquist). Was (0.12, 0.30); widened with the overshoot clamp at 0 (sharpen_sweep.py: DJI sharpness 47.3 -> 52.0 and A7C 17.2 -> 18.9 against RawTherapee's 51.6 and 15.9, flat-area noise unchanged because the activity mask keeps the boost out of flat areas)


_UNSET = object()               # defaults looked up when called, so that scripts/sharpen_sweep.py --patch detail.NOISE_TERM=... reaches them


def sharpen_gain(mtf_freq, mtf, shape, scale, noise=_UNSET, max_gain=4.0, amount=1.0, rolloff=_UNSET):
    """The Wiener filter (2D, for an image of `shape` = (h, w)) that undoes the measured blur: gain (1 + noise) m/(m^2 + noise) at each frequency (scaled so that it is exactly 1 where nothing is blurred, m = 1: a larger noise term must not take contrast off the broad tones), where m is the measured MTF at that frequency of the NATIVE picture (an output pixel is 1/`scale` native pixels, so output frequency f is native frequency f * scale), limited to `max_gain`. `amount` 1 restores the measured blur and no more; above 1 goes past it (the gain is raised to that power), which adds the local contrast the eye reads as crispness, as a camera's own JPEG sharpening does (at the cost of more of the noise).
    Above the measured detail limit the gain falls back towards 1 instead of boosting what is only noise, and the boost is faded out over the top frequencies (`rolloff`): the sharpening was found to lift the finest-scale grain 2 to 3 times, tile detail needing none of that. Isotropic: the lens blur seen from a drone is close to it."""
    noise = NOISE_TERM if noise is _UNSET else noise; rolloff = ROLLOFF if rolloff is _UNSET else rolloff
    h, w = shape; r = np.hypot(*np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(h)))
    m = np.interp(np.minimum(r, 0.5) * scale, mtf_freq, mtf, left=1.0, right=mtf[-1]); m = np.clip(m, 0.02, 1.0)
    g = np.minimum(((1 + noise) * m / (m * m + noise)) ** amount, max_gain); g = np.where(r * scale < mtf_freq[0], 1.0, g)
    if rolloff: g = 1 + (g - 1) * np.clip(1 - (r - rolloff[0]) / (rolloff[1] - rolloff[0]), 0, 1)
    return g.astype(np.float32)


ROLLOFF_BRIGHT = (0.1, 0.3)     # a tighter roll-off for bright areas (walls, render), where the same boost leaves the grain about a third above the camera JPEG's (was (0.06, 0.16), retuned with ROLLOFF)
BRIGHT_FROM, BRIGHT_SPAN = 0.60, 0.28


def sharpen_picture(rgb, mtf_freq, mtf, scale, amount=1.0, backend='auto'):
    """The whole sharpening of an export: `sharpen` with the Wiener filter of the measured blur, but with the tighter roll-off (ROLLOFF_BRIGHT) blended in where the picture is bright (smoothed luminance from BRIGHT_FROM up over BRIGHT_SPAN, smoothstep), so the grain of bright walls is not lifted as it is in the dark and mid tones.
    `rgb` float32 HxWx3 0..1; `scale` is the output size over the native size (see sharpen_gain). Runs on the GPU (gpu.sharpen) when there is one and `backend` is 'auto' or 'gpu' (about 20 times faster); this CPU code is the reference."""
    import cv2
    from stratastills import gpu
    if backend == 'gpu' or (backend == 'auto' and gpu.available()):
        try: return _sharpen_picture_gpu(rgb, mtf_freq, mtf, scale, amount)
        except Exception as e:
            if backend == 'gpu' or not gpu.is_oom(e): raise
        gpu.free()                                                                                                    # the card ran out of memory: the CPU does it
    h, w = rgb.shape[:2]; mask = sharpen_mask(mtf_freq, mtf); base = sharpen(rgb, sharpen_gain(mtf_freq, mtf, (h, w), scale, amount=amount), overshoot=OVERSHOOT, mask=mask, knee=KNEE); y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    b = cv2.GaussianBlur(y, (0, 0), 6.0); k = np.clip((b - BRIGHT_FROM) / BRIGHT_SPAN, 0, 1); k = (k * k * (3 - 2 * k))[..., None]
    if float(k.max()) <= 0: return base
    alt = sharpen(rgb, sharpen_gain(mtf_freq, mtf, (h, w), scale, amount=amount, rolloff=ROLLOFF_BRIGHT), overshoot=OVERSHOOT, mask=mask, knee=KNEE)
    return base * (1 - k) + alt * k


def _sharpen_picture_gpu(rgb, mtf_freq, mtf, scale, amount):
    """sharpen_picture on the device: both filters and the brightness blend, the picture going over once and coming back once."""
    from stratastills import gpu
    with gpu.slot():                                                                                       # one device slot while the picture is on the card (several pictures may be in flight)
        x = gpu.to_gpu(rgb); h, w = x.shape[1:]; g = lambda ro: gpu.sharpen_gain(mtf_freq, mtf, (h, w), scale, NOISE_TERM, 4.0, amount, ro)
        mask = sharpen_mask(mtf_freq, mtf); d1 = gpu.sharpen_delta(x, g(ROLLOFF), OVERSHOOT, mask=mask, knee=KNEE); y = gpu.luma(x); k = gpu.smoothstep((gpu.gaussian(y, 6.0) - BRIGHT_FROM) / BRIGHT_SPAN); del y
        d2 = gpu.sharpen_delta(x, g(ROLLOFF_BRIGHT), OVERSHOOT, mask=mask, knee=KNEE) if float(k.max()) > 0 else None                  # the two filters' results, blended by brightness, one channel at a time (the whole picture of each would not fit a small card)
        for c in range(3):
            x[c] = (x[c] + d1).clamp_(0, 1) if d2 is None else (x[c] + d1).clamp_(0, 1) * (1 - k) + (x[c] + d2).clamp_(0, 1) * k
        out = gpu.to_numpy(x); del x, d1, d2, k; gpu.free(); return out


KNEE = 0.03                    # 0: the sharpened luminance is clipped to the local range (+ OVERSHOOT); above 0 it is taken there smoothly, never more than KNEE beyond it (`limit`)
OVERSHOOT = 0.0                # how far (0..1 of the range) the sharpening may take a pixel beyond the local minimum and maximum of the original (5x5): the halos beside edges. Was 0.05; measured on the A7C set the overshoot metric falls 19.5 -> 17.0 (RawTherapee 18.5; the unsharpened pictures are at 17.7) for 1 level of sharpness
SHARPEN_MASK = (2.0, 6.0)      # the sharpening goes only where there is structure: its weight rises from 0 to 1 as the gradient of the (1 px blurred) luminance goes from the first to the second number, in 8-bit levels per pixel; in the flat areas the boost would only lift the noise (measured: flat-area noise 1.17 -> 1.53 levels with the filter on everywhere)

MASK_REF_FREQ = 0.2             # the native frequency (cycles per pixel) at which the measured MTF decides how far the mask is raised
MASK_REF_MTF = 0.45             # a picture whose MTF there is this or more keeps SHARPEN_MASK (the A7C pictures of the candidates measure 0.5-0.85, DJI 0.2-0.7)
MASK_MAX_FACTOR = 3.5           # the most the mask thresholds are multiplied by


def sharpen_mask(mtf_freq, mtf):
    """SHARPEN_MASK raised for a picture that measures soft (its MTF at MASK_REF_FREQ under MASK_REF_MTF): what the measurement describes there is mostly a soft scene (or a soft lens), the Wiener gain sits at its cap over a wide band, and boosting the low-contrast mottling of the texture makes blobs (a smooth ground comes out as cobbles, edges as scallops: seen on NEX-5N pictures, f50 0.06). Only structure strong enough to be real edges is sharpened then. Factor = MASK_REF_MTF / MTF, 1 to MASK_MAX_FACTOR."""
    if SHARPEN_MASK is None: return None
    m = float(np.interp(MASK_REF_FREQ, mtf_freq, mtf)); k = float(np.clip(MASK_REF_MTF / max(m, 1e-3), 1.0, MASK_MAX_FACTOR)); return (SHARPEN_MASK[0] * k, SHARPEN_MASK[1] * k)


def activity(y, mask=SHARPEN_MASK):
    """The weight (0..1, float32, same shape as the luminance `y` in 0..1) of the sharpening at each pixel: the gradient of `y` blurred by 1 px, in 8-bit levels per pixel, through a ramp (`mask`), then blurred by 1.5 px so that the edge of the mask does not show. None: everywhere."""
    import cv2
    if mask is None: return np.ones_like(y)
    s = cv2.GaussianBlur(y * 255, (0, 0), 1.0); g = np.hypot(cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)) / 8
    return cv2.GaussianBlur(np.clip((g - mask[0]) / (mask[1] - mask[0]), 0, 1).astype(np.float32), (0, 0), 1.5)


def limit(v, lo, hi, knee):
    """`v` held to [lo, hi]: clipped when `knee` is 0, else taken smoothly towards lo - knee and hi + knee (tanh) so that a pixel beyond the range keeps some of its excess and neighbouring pixels do not all land on the same value."""
    if not knee: return np.clip(v, lo, hi)
    return np.where(v > hi, hi + knee * np.tanh((v - hi) / knee), np.where(v < lo, lo - knee * np.tanh((lo - v) / knee), v)).astype(np.float32)


def sharpen(rgb, gain, overshoot=0.05, mask=None, knee=0.0):
    """Apply a sharpen_gain filter to the luminance of `rgb` (float32 HxWx3, 0..1): the change is added equally to the three channels (colour is left alone), weighted by the activity of the picture there (`mask`, see `activity`; None: everywhere), and the result held to the local range of the original widened by `overshoot` (`limit` with `knee`; `overshoot` None: not held)."""
    import cv2
    y = rgb @ np.array([0.299, 0.587, 0.114], np.float32); ys = np.fft.ifft2(np.fft.fft2(y) * gain).real.astype(np.float32); d = (ys - y) * activity(y, mask)
    if overshoot is not None: k = np.ones((5, 5), np.uint8); d = limit(y + d, cv2.erode(y, k) - overshoot, cv2.dilate(y, k) + overshoot, knee) - y
    return np.clip(rgb + d[..., None], 0, 1)
