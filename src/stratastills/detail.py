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
ROLLOFF = (0.12, 0.30)          # the boost fades to nothing between these OUTPUT frequencies (cycles per pixel; 0.5 is Nyquist): tile courses, brick lines and edges live below 0.3, the grain above it


def sharpen_gain(mtf_freq, mtf, shape, scale, noise=NOISE_TERM, max_gain=4.0, amount=1.0, rolloff=ROLLOFF):
    """The Wiener filter (2D, for an image of `shape` = (h, w)) that undoes the measured blur: gain (1 + noise) m/(m^2 + noise) at each frequency (scaled so that it is exactly 1 where nothing is blurred, m = 1: a larger noise term must not take contrast off the broad tones), where m is the measured MTF at that frequency of the NATIVE picture (an output pixel is 1/`scale` native pixels, so output frequency f is native frequency f * scale), limited to `max_gain`. `amount` 1 restores the measured blur and no more; above 1 goes past it (the gain is raised to that power), which adds the local contrast the eye reads as crispness, as a camera's own JPEG sharpening does (at the cost of more of the noise).
    Above the measured detail limit the gain falls back towards 1 instead of boosting what is only noise, and the boost is faded out over the top frequencies (`rolloff`): the sharpening was found to lift the finest-scale grain 2 to 3 times, tile detail needing none of that. Isotropic: the lens blur seen from a drone is close to it."""
    h, w = shape; r = np.hypot(*np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(h)))
    m = np.interp(np.minimum(r, 0.5) * scale, mtf_freq, mtf, left=1.0, right=mtf[-1]); m = np.clip(m, 0.02, 1.0)
    g = np.minimum(((1 + noise) * m / (m * m + noise)) ** amount, max_gain); g = np.where(r * scale < mtf_freq[0], 1.0, g)
    if rolloff: g = 1 + (g - 1) * np.clip(1 - (r - rolloff[0]) / (rolloff[1] - rolloff[0]), 0, 1)
    return g.astype(np.float32)


ROLLOFF_BRIGHT = (0.06, 0.16)   # a tighter roll-off for bright areas (walls, render), where the same boost leaves the grain about a third above the camera JPEG's; measured: bright-wall grain 4.06 -> 3.58 with window and moulding edges and roof-tile detail unchanged
BRIGHT_FROM, BRIGHT_SPAN = 0.60, 0.28


def sharpen_picture(rgb, mtf_freq, mtf, scale, amount=1.0, backend='auto'):
    """The whole sharpening of an export: `sharpen` with the Wiener filter of the measured blur, but with the tighter roll-off (ROLLOFF_BRIGHT) blended in where the picture is bright (smoothed luminance from BRIGHT_FROM up over BRIGHT_SPAN, smoothstep), so the grain of bright walls is not lifted as it is in the dark and mid tones.
    `rgb` float32 HxWx3 0..1; `scale` is the output size over the native size (see sharpen_gain). Runs on the GPU (gpu.sharpen) when there is one and `backend` is 'auto' or 'gpu' (about 20 times faster); this CPU code is the reference."""
    import cv2
    from stratastills import gpu
    if backend == 'gpu' or (backend == 'auto' and gpu.available()): return _sharpen_picture_gpu(rgb, mtf_freq, mtf, scale, amount)
    h, w = rgb.shape[:2]; base = sharpen(rgb, sharpen_gain(mtf_freq, mtf, (h, w), scale, amount=amount)); y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    b = cv2.GaussianBlur(y, (0, 0), 6.0); k = np.clip((b - BRIGHT_FROM) / BRIGHT_SPAN, 0, 1); k = (k * k * (3 - 2 * k))[..., None]
    if float(k.max()) <= 0: return base
    alt = sharpen(rgb, sharpen_gain(mtf_freq, mtf, (h, w), scale, amount=amount, rolloff=ROLLOFF_BRIGHT))
    return base * (1 - k) + alt * k


def _sharpen_picture_gpu(rgb, mtf_freq, mtf, scale, amount):
    """sharpen_picture on the device: both filters and the brightness blend, the picture going over once and coming back once."""
    from stratastills import gpu
    with gpu.slot():                                                                                       # one device slot while the picture is on the card (several pictures may be in flight)
        x = gpu.to_gpu(rgb); h, w = x.shape[1:]; g = lambda ro: gpu.sharpen_gain(mtf_freq, mtf, (h, w), scale, NOISE_TERM, 4.0, amount, ro)
        base = gpu.sharpen(x, g(ROLLOFF)); y = gpu.luma(x); k = gpu.smoothstep((gpu.gaussian(y, 6.0) - BRIGHT_FROM) / BRIGHT_SPAN)[None]; del y
        if float(k.max()) > 0: base = base * (1 - k) + gpu.sharpen(x, g(ROLLOFF_BRIGHT)) * k
        out = gpu.to_numpy(base); del x, base, k; gpu.free(); return out


def sharpen(rgb, gain, overshoot=0.05):
    """Apply a sharpen_gain filter to the luminance of `rgb` (float32 HxWx3, 0..1): the change is added equally to the three channels (colour is left alone), and the result may not go beyond the local range of the original by more than `overshoot` (no halos)."""
    import cv2
    y = rgb @ np.array([0.299, 0.587, 0.114], np.float32); ys = np.fft.ifft2(np.fft.fft2(y) * gain).real.astype(np.float32); d = ys - y
    k = np.ones((5, 5), np.uint8); lo, hi = cv2.erode(y, k) - overshoot, cv2.dilate(y, k) + overshoot; d = np.clip(y + d, lo, hi) - y
    return np.clip(rgb + d[..., None], 0, 1)
