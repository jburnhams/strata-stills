"""The GPU development of a picture too big for the card, in tiles: develop.py's `_develop_gpu_on_device` with the same recipe step for step, but the picture lives in system memory and goes through the card a tile at a time.
It is possible because almost nothing in the recipe looks far: the lens corrections and colour are per pixel (or per pixel from the pixels around the one it samples), the clean-up filters reach at most about 90 pixels (`develop.gpu_tail`), and what is global is a handful of numbers (the local-tone base on a 1200 px copy, its median, the exposure, the visible-noise estimate) that are found once from a reduced copy of the whole. So:
  pass A  per tile: the lens shading gain and the lens warp (the part of the source the tile samples), the colour -> exposed linear picture, stored in system memory (float32, 12 bytes a pixel)
  global  the local-tone base from a reduced copy, the exposure and the noise estimate from a sample every 8th pixel (as the whole-picture version does)
  pass B  per tile with a margin of HALO pixels: local tone, exposure, then `develop.gpu_tail`; the margin is cut off and the tile written into the result
tests/unit/test_gpu_tiled.py holds the result to that of the whole-picture version. Operations the tiling cannot do (an opcode list in an order it does not handle) raise Unsupported and the caller falls back to developing a reduced picture."""
import cv2, numpy as np

from stratastills import denoise_ai, develop as D, gpu

TILE = 2048
HALO = 128                      # the widest reach of the clean-up (guided filters, grain bands, the noise map's blur at 1/8 scale: about 90) with room to spare; a multiple of 8, which the noise map's reduction wants


class Unsupported(Exception): pass


def boxes(h, w, tile):
    """The core boxes (y0, y1, x0, x1) of the tiles, aligned to multiples of 8."""
    tile = max(8, tile // 8 * 8)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile): yield y0, min(h, y0 + tile), x0, min(w, x0 + tile)


def _opcodes(tags):
    gains, warp = [], None
    if 'OpcodeList3' in tags:
        for oid, p in D.parse_opcodes(tags['OpcodeList3']):
            if oid == 9:
                if warp is not None: raise Unsupported('a lens shading map after the lens warp')
                m, plane, planes, _b, _s, _o = D.gain_map(p, tags.get('Flip', 0)); gains.append((m, plane, planes))
            elif oid == 1:
                if warp is not None: raise Unsupported('two lens warps')
                warp = D.warp_params(p, tags.get('Flip', 0))
    return gains, warp


def _gain_plane(m, k, shape, ys, xs):
    t = gpu.torch_(); return gpu.sample_grid(t.from_numpy(np.ascontiguousarray(m[:, :, min(k, m.shape[2] - 1)])).to(gpu.dev()), ys, xs, shape[0], shape[1])


def _analysis_luma(u16, rows=1024):
    """D.analysis_luma's result from this decode (nothing done that changes sharpness, the median exposed to 0.18, gamma-encoded), in bands of rows."""
    w = np.array([0.2126, 0.7152, 0.0722], np.float32); sample = u16[::8, ::8].astype(np.float32) / 65535.0 @ w; med = max(float(np.sort(sample.reshape(-1))[(sample.size - 1) // 2]), 1e-5); out = np.empty(u16.shape[:2], np.float32)
    for r in range(0, u16.shape[0], rows):
        y = gpu.luma(gpu.to_gpu_u16(u16[r:r + rows])); out[r:r + rows] = gpu.to_numpy(gpu.srgb_oetf(y * (0.18 / med)))
    return out


def develop_tiled(path, u16, ev, denoise, recover, highlight_contrast, analysis=None, tile=None, halo=HALO, tone=None, lens='auto'):
    """develop._develop_gpu_on_device for a picture too big for the card: the same result (float32 RGB (H, W, 3), 0..1), made in tiles. `u16`: the decoded raw (H, W, 3) uint16."""
    tile = tile or gpu.tile_size(halo); t = gpu.torch_(); tags = D.read_dng_tags(path, lens); H, W = u16.shape[:2]; dv = gpu.dev(); gains, warp = _opcodes(tags); field_map = None
    if analysis is not None: analysis['luma'] = _analysis_luma(u16)
    if warp is not None and not gpu.cubic_ok(): raise Unsupported('this device has no bicubic sampling')
    be = float(D._rationals(tags['BaselineExposure'])[0]) if 'BaselineExposure' in tags else 0.0
    if gains: field_map = (gains[-1][0], min(1, gains[-1][2] - 1))                                                      # the second plane's map of the last shading opcode is the field the noise map counts (as gpu.apply_gain_map returns it)
    # ---- pass A
    lin = np.empty((3, H, W), np.float32)
    for box in boxes(H, W, tile):
        y0, y1, x0, x1 = box; sb = gpu.warp_source_box(warp[0], warp[1], H, W, box) if warp is not None else box; sy0, sy1, sx0, sx1 = sb
        img = gpu.to_gpu_u16(u16[sy0:sy1, sx0:sx1]); ys, xs = np.arange(sy0, sy1), np.arange(sx0, sx1)
        for m, plane, planes in gains:
            for k in range(planes): img[plane + k] *= _gain_plane(m, k, (H, W), ys, xs)
        if warp is not None: img = gpu.warp_rectilinear_box(img, sb, warp[0], warp[1], H, W, box)
        img = D.gpu_colour(img, tags).clamp(min=0) * 2.0 ** be; lin[:, y0:y1, x0:x1] = img.cpu().numpy(); del img
    gpu.free()
    # ---- global: the local-tone base, the exposure, the noise estimate
    curve, chroma = D.tone_parts(tone, tags); lut = D.tone_lut(curve); lut_t = t.from_numpy(lut).to(dv) if lut is not None else None; chroma_t = gpu.look_to(chroma, dv); sy = np.arange(0, H, 8); sx = np.arange(0, W, 8); base_s = mid = None
    wts = np.array([0.2126, 0.7152, 0.0722], np.float32)
    if recover > 0:
        ly = np.log2(np.maximum(lin[0] * wts[0] + lin[1] * wts[1] + lin[2] * wts[2], 1e-5)); k = min(1.0, 1200 / max(H, W)); sz = (max(8, round(W * k)), max(8, round(H * k)))
        small = cv2.resize(ly, sz, interpolation=cv2.INTER_AREA); del ly; base_s = t.from_numpy(cv2.bilateralFilter(small, 9, 0.5, max(4.0, 0.012 * max(small.shape)))).to(dv)
        mid = gpu._median(gpu.sample_grid(base_s, sy, sx, H, W, 'bicubic'))

    def local_tone(region, ys, xs):
        """gpu.local_tone on a region of the picture (3, rh, rw) with the global base and median."""
        if recover <= 0: return region
        ly = t.log2(gpu.luma(region).clamp_(min=1e-5)); base = gpu.sample_grid(base_s, ys, xs, H, W, 'bicubic'); kk = t.where(base > mid, 1 - D.HIGHLIGHTS * min(1.0, recover / 0.35), t.full_like(base, 1 - recover)); new = mid + (base - mid) * kk + (ly - base)
        return region * t.exp2(new - ly)[None]
    sample = local_tone(t.from_numpy(np.ascontiguousarray(lin[:, ::8, ::8])).to(dv), sy, sx); g = 2.0 ** (D.auto_ev(gpu.to_numpy(sample), lut, stride=1) if ev == 'auto' else D.ev_stops(ev))
    npf = tags.get('NoiseProfile'); npf = (npf[0] * g * 2.0 ** be, npf[1] * (g * 2.0 ** be) ** 2) if npf is not None else None; sc = 0.0
    if npf is not None and denoise in ('auto', 'ai', 'chroma'): sc = float(np.clip(D.visible_noise(gpu.to_numpy(sample * g), npf, stride=1) * 0.6, 0.0, 1.5))
    del sample
    # ---- pass B
    out = np.empty((H, W, 3), np.float32); sig8 = np.empty((H, W), np.float32) if npf is not None else None; blinds = []
    for box in boxes(H, W, tile):
        y0, y1, x0, x1 = box; ry0, ry1, rx0, rx1 = max(0, y0 - halo), min(H, y1 + halo), max(0, x0 - halo), min(W, x1 + halo); ys, xs = np.arange(ry0, ry1), np.arange(rx0, rx1)
        img = local_tone(t.from_numpy(np.ascontiguousarray(lin[:, ry0:ry1, rx0:rx1])).to(dv), ys, xs) * g
        field = _gain_plane(field_map[0], field_map[1], (H, W), ys, xs) if field_map is not None else None
        ex = {}; res = D.gpu_tail(img, lut_t, npf, field, sc, 'auto' if denoise == 'ai' else denoise, highlight_contrast, chroma_t, extras=ex); del img
        out[y0:y1, x0:x1] = gpu.to_numpy(res[:, y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0]); del res
        if sig8 is not None: sig8[y0:y1, x0:x1] = gpu.to_numpy(ex['sig8'][y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0]); blinds.append(float(ex['blind']))
    gpu.free()
    blind = float(np.median(blinds)) if blinds else 0.0                                                                  # the whole picture's blind noise: the median of the tiles' (each from its flattest quarter)
    ai = npf is not None and (denoise == 'ai' or (denoise == 'auto' and D._auto_ai(blind, lambda: D.blind_noise(out)))) and denoise_ai.available()
    if analysis is not None: analysis['ai'] = bool(ai); analysis['blind_noise'] = blind
    return denoise_ai.adaptive(out, sig8, denoise_ai.DEFAULT, denoise_ai.DEFAULT_STRENGTH) if ai else out
