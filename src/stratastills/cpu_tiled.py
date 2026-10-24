"""The NumPy development (the CPU reference, develop.py) in tiles, for machines without a GPU and not enough memory for a whole picture: the whole-picture code peaks at about 145 bytes a pixel (7 GB for 48 MP, 17 GB for a 116 MP panorama); this keeps 36 bytes a pixel for the picture and a few hundred for the tile in hand, and the tile is as big as the picture's share of memory allows (resources.cpu_tile).
The same plan as gpu_tiled.py, step for step, using the NumPy functions: pass A per tile (shading gain, the lens warp of the part of the source the tile samples, colour) into a float32 array; the few global numbers (local-tone base on a 1200 px copy, its median, exposure, the noise estimate) from a reduced copy and a sample every 8th pixel; pass B per tile with a margin of HALO pixels (the widest reach of the clean-up is about 90): local tone, exposure, noise map, tone curve and clean-up; the margin is cut off. tests/unit/test_cpu_tiled.py holds the result to the whole-picture one."""
import cv2, numpy as np

from stratastills import develop as D
from stratastills.gpu_tiled import Unsupported, boxes, _opcodes

HALO = 128
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _maps(ys, xs, hh, ww, sh, sw):
    """cv2.remap maps (x, y) that stretch an (sh, sw) plane over an (hh, ww) picture as cv2.resize does (half-pixel centres), for the rows `ys` and columns `xs` of the picture."""
    mx = ((np.asarray(xs, np.float32) + 0.5) * (sw / ww) - 0.5)[None, :].repeat(len(ys), 0); my = ((np.asarray(ys, np.float32) + 0.5) * (sh / hh) - 0.5)[:, None].repeat(len(xs), 1); return mx, my


def _sample(plane, ys, xs, hh, ww, interp):
    """The plane enlarged to (hh, ww) (cv2.resize with `interp`), evaluated only at rows `ys` and columns `xs`."""
    mx, my = _maps(ys, xs, hh, ww, *plane.shape); return cv2.remap(np.ascontiguousarray(plane, np.float32), mx, my, interp, borderMode=cv2.BORDER_REPLICATE)


def _warp_coords(coef, centre, h, w, box, planes=3):
    y0, y1, x0, x1 = box; cx, cy = centre[0] * (w - 1), centre[1] * (h - 1); mr = np.hypot(max(cx, w - 1 - cx), max(cy, h - 1 - cy))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32); x = (xx - cx) / mr; y = (yy - cy) / mr; r2 = x * x + y * y; XS, YS = [], []
    for c in range(planes):
        kr0, kr1, kr2, kr3, kt0, kt1 = coef[min(c, len(coef) - 1)]; f = kr0 + r2 * (kr1 + r2 * (kr2 + r2 * kr3))
        XS.append(((x * f + kt0 * 2 * x * y + kt1 * (r2 + 2 * x * x)) * mr + cx).astype(np.float32)); YS.append(((y * f + kt1 * 2 * x * y + kt0 * (r2 + 2 * y * y)) * mr + cy).astype(np.float32))
    return XS, YS


def _analysis_luma(u16, rows=1024):
    sample = u16[::8, ::8].astype(np.float32) / 65535.0 @ LUMA; med = max(float(np.sort(sample.reshape(-1))[(sample.size - 1) // 2]), 1e-5); out = np.empty(u16.shape[:2], np.float32)
    for r in range(0, u16.shape[0], rows): out[r:r + rows] = D.srgb_oetf(np.clip((u16[r:r + rows].astype(np.float32) / 65535.0 @ LUMA) * (0.18 / med), 0, 1))
    return out


def develop_tiled(path, u16, ev, denoise, recover, highlight_contrast, analysis=None, tile=2048, halo=HALO, tone=None, lens='auto'):
    """develop.develop (backend 'cpu', denoise 'auto', 'chroma' or 'off') for a picture too big for the memory it may use: the same result (float32 RGB (H, W, 3), 0..1), made in tiles of `tile` pixels. `u16`: the decoded raw (H, W, 3) uint16."""
    tags = D.read_dng_tags(path, lens); H, W = u16.shape[:2]; gains, warp = _opcodes(tags)
    if analysis is not None: analysis['luma'] = _analysis_luma(u16)
    be = float(D._rationals(tags['BaselineExposure'])[0]) if 'BaselineExposure' in tags else 0.0; field_src = (gains[-1][0], min(1, gains[-1][2] - 1)) if gains else None
    # ---- pass A
    lin = np.empty((H, W, 3), np.float32)
    for box in boxes(H, W, tile):
        y0, y1, x0, x1 = box; sy0, sy1, sx0, sx1 = box
        if warp is not None:
            XS, YS = _warp_coords(warp[0], warp[1], H, W, box); sy0 = max(0, int(min(a.min() for a in YS)) - 4); sy1 = min(H, int(max(a.max() for a in YS)) + 6); sx0 = max(0, int(min(a.min() for a in XS)) - 4); sx1 = min(W, int(max(a.max() for a in XS)) + 6)
        img = u16[sy0:sy1, sx0:sx1].astype(np.float32) / 65535.0; ys, xs = np.arange(sy0, sy1), np.arange(sx0, sx1)
        for m, plane, planes in gains:
            for k in range(planes): img[:, :, plane + k] *= _sample(m[:, :, min(k, m.shape[2] - 1)], ys, xs, H, W, cv2.INTER_LINEAR)
        if warp is not None:
            out = np.empty((y1 - y0, x1 - x0, 3), np.float32)
            for c in range(3): out[:, :, c] = cv2.remap(np.ascontiguousarray(img[:, :, c]), XS[c] - sx0, YS[c] - sy0, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            img = out; del XS, YS
        lin[y0:y1, x0:x1] = np.clip(D.cpu_colour(img, tags), 0, None) * 2.0 ** be; del img
    # ---- global: the local-tone base, the exposure, the noise estimate
    curve, chroma = D.tone_parts(tone, tags); lut = D.tone_lut(curve); sy = np.arange(0, H, 8); sx = np.arange(0, W, 8); base_s = mid = None
    if recover > 0:
        ly = np.log2(np.maximum(lin @ LUMA, 1e-5)); k = min(1.0, 1200 / max(H, W)); small = cv2.resize(ly, (max(8, round(W * k)), max(8, round(H * k))), interpolation=cv2.INTER_AREA); del ly
        base_s = cv2.bilateralFilter(small, 9, 0.5, max(4.0, 0.012 * max(small.shape))); mid = float(np.sort(_sample(base_s, sy, sx, H, W, cv2.INTER_CUBIC).reshape(-1))[(len(sy) * len(sx) - 1) // 2])

    def local_tone(region, ys, xs):
        if recover <= 0: return region
        ly = np.log2(np.maximum(region @ LUMA, 1e-5)); base = _sample(base_s, ys, xs, H, W, cv2.INTER_CUBIC); kk = np.where(base > mid, 1 - D.HIGHLIGHTS * min(1.0, recover / 0.35), 1 - recover); new = mid + (base - mid) * kk + (ly - base)
        return region * (2.0 ** (new - ly))[..., None]
    sample = local_tone(np.ascontiguousarray(lin[::8, ::8]), sy, sx); g = 2.0 ** (D.auto_ev(sample, lut, stride=1) if ev == 'auto' else D.ev_stops(ev))
    npf = tags.get('NoiseProfile'); npf = (npf[0] * g * 2.0 ** be, npf[1] * (g * 2.0 ** be) ** 2) if npf is not None else None; sc = 0.0
    if npf is not None and denoise == 'auto': sc = float(np.clip(D.visible_noise(sample * g, npf, stride=1) * 0.6, 0.0, 1.5))
    del sample; classical = denoise in ('auto', 'chroma') and npf is not None
    # ---- pass B
    out = np.empty((H, W, 3), np.float32)
    for box in boxes(H, W, tile):
        y0, y1, x0, x1 = box; ry0, ry1, rx0, rx1 = max(0, y0 - halo), min(H, y1 + halo), max(0, x0 - halo), min(W, x1 + halo); ys, xs = np.arange(ry0, ry1), np.arange(rx0, rx1)
        img = local_tone(np.ascontiguousarray(lin[ry0:ry1, rx0:rx1]), ys, xs) * g
        if sc > 0.3: y = img @ LUMA; c = cv2.GaussianBlur(img - y[..., None], (0, 0), sc); img = np.clip(y[..., None] + c, 0, None)
        field = _sample(field_src[0][:, :, min(field_src[1], field_src[0].shape[2] - 1)], ys, xs, H, W, cv2.INTER_LINEAR) if field_src is not None else None
        sig8 = D.noise_map8(img, npf, gain=field) if npf is not None else None; res = D.limit_chroma(D.chroma_scale(D.finish(img, lut), chroma), None if chroma is None else 0.0); del img
        if classical:
            res = D.clean_colour(res, sig8)
            if denoise == 'auto': res = D.core_luma(res, sig8)
        if highlight_contrast > 0: res = D.highlight_detail(res, highlight_contrast)
        out[y0:y1, x0:x1] = res[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0]; del res, sig8
    return out
