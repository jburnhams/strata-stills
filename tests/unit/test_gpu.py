"""Each torch function in gpu.py is held to the CPU original in develop.py / detail.py (the reference), on random pictures with structure and noise. The tests run on every device there is: torch on the CPU always (so a machine with no GPU still checks the code), CUDA and Apple's MPS where present."""
import cv2, numpy as np, pytest

pytest.importorskip('torch')
from stratastills import develop as D, gpu


@pytest.fixture(params=['cpu', 'cuda', 'mps'], autouse=True)
def device(request):
    t = gpu.torch_()
    if request.param == 'cuda' and not t.cuda.is_available(): pytest.skip('no CUDA')
    if request.param == 'mps' and not (getattr(t.backends, 'mps', None) is not None and t.backends.mps.is_available()): pytest.skip('no MPS')
    gpu.use(request.param); yield request.param; gpu.use(None)
H, W = 240, 320
RNG = np.random.default_rng(0)


def picture(seed=0, scale=1.0, noise=0.02):
    """A smooth, edgy, noisy float32 picture (H, W, 3) in 0..scale."""
    rng = np.random.default_rng(seed); base = cv2.GaussianBlur(rng.random((H, W, 3)).astype(np.float32), (0, 0), 6) * 3 - 0.8
    base[:, W // 2:] += 0.3; base[H // 2:, :W // 4] *= 0.4; return np.clip(base + rng.normal(0, noise, base.shape).astype(np.float32), 0, 1) * scale


def g(a): return gpu.to_gpu(a)
def c(t): return gpu.to_numpy(t)


def inner(a, m=20): return a[m:-m, m:-m]


@pytest.mark.parametrize('r', [1, 2, 6, 14])
def test_box_filter_matches_opencv(r):
    p = picture()[..., 0]; ref = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT_101); out = c(gpu.box(g(p), r))
    assert np.abs(out - ref).max() < 1e-5


@pytest.mark.parametrize('sigma', [1.0, 2.0, 4.0, 8.0, 1.5])
def test_gaussian_matches_opencv(sigma):
    p = picture()[..., 1]; ref = cv2.GaussianBlur(p, (0, 0), sigma); out = c(gpu.gaussian(g(p), sigma)); assert np.abs(out - ref).max() < 2e-4
    st = gpu.gaussian(g(picture()).contiguous(), sigma); assert st.shape == (3, H, W)                                    # a stack of planes at once


def test_guided_filter_matches_the_cpu_one_away_from_the_border():
    p = picture()[..., 0]; q = picture(1)[..., 1]
    for r, eps in ((2, 4e-4), (6, 1.5e-3), (14, 4e-3)):
        ref = D._guided(p, q, r, eps); out = c(gpu.guided(g(p), g(q), r, eps)); assert np.abs(inner(out, 2 * r + 3) - inner(ref, 2 * r + 3)).max() < 5e-4 and np.abs(out - ref).mean() < 5e-4


def test_lut_matches_np_interp():
    lut = np.cumsum(RNG.random(4096)).astype(np.float32); lut /= lut[-1]; x = RNG.random((3, 50, 60)).astype(np.float32); x[0, 0, :3] = (0.0, 1.0, 0.5)
    ref = np.interp(x, np.linspace(0, 1, 4096), lut); out = gpu.apply_lut(gpu.torch_().from_numpy(x).to(gpu.dev()), gpu.torch_().from_numpy(lut).to(gpu.dev()), chunk=1000).cpu().numpy(); assert np.abs(out - ref).max() < 1e-6


def test_shoulder_and_finish_match():
    img = picture(2, scale=5.0); lut = np.cumsum(RNG.random(4096)).astype(np.float32); lut /= lut[-1]; lt = gpu.torch_().from_numpy(lut).to(gpu.dev())
    assert np.abs(c(gpu.shoulder(g(img), D.KNEE, D.WHITE)) - D.shoulder(img)).max() < 1e-4
    assert np.abs(c(gpu.finish(g(img), lt, D.KNEE, D.WHITE)) - D.finish(img, lut)).max() < 5e-4
    assert np.abs(c(gpu.finish(g(img), None, D.KNEE, D.WHITE)) - D.finish(img, None)).max() < 5e-4


def test_noise_map_matches():
    img = picture(3, scale=0.5); npf = (0.004, 1e-6); ref = D.noise_map8(img, npf); out = c(gpu.noise_map8(g(img), npf)); assert np.abs(out - ref).max() < 0.05 * ref.max() and np.abs(out - ref).mean() < 0.01 * ref.mean()


def test_clean_colour_matches():
    img = picture(4); sig = np.full((H, W), 8.0, np.float32); sig[:, W // 2:] = 3.0; ref = D.clean_colour(img, sig); out = c(gpu.clean_colour(g(img), g(sig)))
    assert np.abs(inner(out) - inner(ref)).max() < 2e-3 and np.abs(out - ref).mean() < 2e-4


def test_core_luma_matches():
    img = picture(5); sig = np.full((H, W), 8.0, np.float32); ref = D.core_luma(img, sig); out = c(gpu.core_luma(g(img), g(sig), D.BAND_CORE)); assert np.abs(inner(out) - inner(ref)).max() < 2e-3 and np.abs(out - ref).mean() < 2e-4


def test_highlight_detail_matches():
    img = picture(6) * 0.4 + 0.55; ref = D.highlight_detail(np.clip(img, 0, 1), 0.8); out = c(gpu.highlight_detail(g(np.clip(img, 0, 1)), 0.8)); assert np.abs(inner(out) - inner(ref)).max() < 3e-3 and np.abs(out - ref).mean() < 3e-4


def test_gain_map_and_warp_match():
    img = picture(7); m = (1 + RNG.random((8, 8, 3)) * 2).astype(np.float32); ref, rf = D.apply_gain_map(img.copy(), m, 0, 3); out, of = gpu.apply_gain_map(g(img), m, 0, 3)
    assert np.abs(c(out) - ref).max() < 2e-5 and np.abs(c(of) - rf).max() < 2e-5
    coef = np.array([[0.953, -4e-5, 6e-4, -4e-4, 0, 0], [0.953, 0, 0, 0, 0, 0], [0.9532, -6.6e-4, 2.8e-4, 1.2e-4, 0, 0]]); a = D.warp_rectilinear(img, coef, (0.5, 0.5)); b = c(gpu.warp_rectilinear(g(img), coef, (0.5, 0.5)))
    assert np.abs(inner(a, 10) - inner(b, 10)).max() < 0.02 and np.abs(a - b).mean() < 1.5e-3


def test_local_tone_matches():
    img = picture(8, scale=0.3) + 0.01; ref = D.local_tone(img, 0.35, 1200, 0.55); out = c(gpu.local_tone(g(img), 0.35, 0.55)); rel = np.abs(out - ref) / (ref + 1e-3)
    assert rel.mean() < 0.01 and np.percentile(rel, 99) < 0.05 and gpu.local_tone(g(img), 0, 0.55).shape == (3, H, W)


def test_median_and_quantile_by_sorting():
    x = gpu.torch_().from_numpy(RNG.random((9, 11)).astype(np.float32)).to(gpu.dev()); v = np.sort(x.cpu().numpy().ravel())
    assert gpu._median(x) == pytest.approx(float(v[(v.size - 1) // 2])) and gpu.quantile(x, 0.995) == pytest.approx(float(v[round(0.995 * (v.size - 1))])) and gpu.quantile(x, 0.0) == pytest.approx(float(v[0]))


def test_the_lens_warp_falls_back_to_the_cpu_where_bicubic_sampling_is_missing(monkeypatch):
    img = picture(7); coef = np.array([[0.953, -4e-5, 6e-4, -4e-4, 0, 0]] * 3); monkeypatch.setattr(gpu, '_cubic_ok', False)
    out = c(gpu.warp_rectilinear(g(img), coef, (0.5, 0.5))); assert np.abs(out - D.warp_rectilinear(img, coef, (0.5, 0.5))).max() < 1e-6                            # exactly the CPU result


def test_device_choice(monkeypatch):
    monkeypatch.setenv('STRATASTILLS_DEVICE', 'cpu'); gpu.use(None); assert gpu.dev().type == 'cpu' and not gpu.available()                                        # torch on the CPU is for tests: callers use the CPU code, not this
    monkeypatch.delenv('STRATASTILLS_DEVICE'); monkeypatch.setenv('STRATASTILLS_GPU', '0'); gpu.use(None); assert not gpu.available()
    with pytest.raises(RuntimeError): gpu.dev()
    monkeypatch.delenv('STRATASTILLS_GPU'); gpu.use('cpu'); assert gpu.dev().type == 'cpu' and gpu.cubic_ok()


def test_a_uint16_raw_goes_to_the_device_as_floats():
    a = RNG.integers(0, 65536, (6, 8, 3), dtype=np.uint16); a[0, 0] = (0, 65535, 40000); out = c(gpu.to_gpu_u16(a).permute(1, 2, 0)) if False else gpu.to_gpu_u16(a).permute(1, 2, 0).cpu().numpy()
    assert np.abs(out - a.astype(np.float32) / 65535.0).max() < 1e-6 and out[0, 0, 1] == 1.0


def synthetic_dng(monkeypatch, h=240, w=320, seed=3):
    """A fake DNG for D.develop: a LibRaw-like uint16 picture and the tags of a real Mini 4 Pro file (colour matrices, noise profile, tone curve, a gain map and a warp)."""
    import struct
    rng = np.random.default_rng(seed); base = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 5) * 2.2 - 0.4; base[:, w // 2:] *= 1.6; raw = (np.clip(base * 0.06 + 0.01 + rng.normal(0, 0.002, base.shape), 0, 1) * 65535).astype(np.uint16)
    cm = tuple(v for x in ((12758, 10000), (-6160, 10000), (916, 10000), (-5992, 10000), (14682, 10000), (5340, 10000), (-251, 10000), (792, 10000), (9511, 10000)) for v in x)
    gm = (1 + np.linspace(0, 1, 8 * 8 * 3).reshape(8, 8, 3) * 3).astype('>f4'); gp = struct.pack('>10I', 0, 0, h, w, 0, 3, 1, 1, 8, 8) + struct.pack('>4d', 1 / 7, 1 / 7, 0.0, 0.0) + struct.pack('>I', 3) + gm.tobytes()
    wp = struct.pack('>I', 3) + b''.join(struct.pack('>6d', *cf) for cf in ((0.953, -4e-5, 6e-4, -4e-4, 0, 0), (0.953, 0, 0, 0, 0, 0), (0.9532, -6.6e-4, 2.8e-4, 1.2e-4, 0, 0))) + struct.pack('>2d', 0.5, 0.5)
    ops = struct.pack('>I', 2) + struct.pack('>IIII', 9, 0x01030000, 0, len(gp)) + gp + struct.pack('>IIII', 1, 0x01030000, 0, len(wp)) + wp
    tags = dict(ColorMatrix1=cm, ColorMatrix2=cm, AsShotNeutral=(4096, 9776, 4096, 4096, 4096, 6393), BaselineExposure=(19, 100), NoiseProfile=(0.0004, 1e-7), ProfileToneCurve=tuple(v for x in np.linspace(0, 1, 64) for v in (x, x ** 0.6)), OpcodeList3=ops)
    monkeypatch.setattr(D, 'read_dng_tags', lambda p: tags); monkeypatch.setattr(D, '_raw_u16', lambda p, half: raw.copy()); monkeypatch.setattr(D.denoise_ai, 'available', lambda m=None: False)
    return raw


@pytest.mark.parametrize('denoise', ['auto', 'chroma', 'off'])
def test_the_whole_development_agrees_with_the_cpu_one(monkeypatch, denoise):
    synthetic_dng(monkeypatch); cpu = D.develop('x.dng', denoise=denoise, backend='cpu'); out = D.develop('x.dng', denoise=denoise, backend='gpu')
    assert out.shape == cpu.shape and out.dtype == np.float32 and np.isfinite(out).all()
    d = np.abs(out - cpu); assert d.mean() < 2.5e-3 and np.percentile(d, 99.9) < 0.05, (d.mean(), np.percentile(d, 99.9))


def test_develop_picks_the_gpu_only_when_there_is_one_and_the_option_is_one_it_does(monkeypatch):
    synthetic_dng(monkeypatch); calls = []; monkeypatch.setattr(D, '_develop_gpu', lambda *a: calls.append('gpu') or np.zeros((2, 2, 3), np.float32)); monkeypatch.setattr(D, 'develop_parts', lambda *a, **k: calls.append('cpu') or (_ for _ in ()).throw(StopIteration()))
    monkeypatch.setattr(gpu, 'available', lambda: True); D.develop('x.dng'); D.develop('x.dng', denoise='chroma'); assert calls == ['gpu', 'gpu']
    for kw in (dict(denoise=3.0), dict(denoise='ai'), dict(backend='cpu')):
        with pytest.raises(StopIteration): D.develop('x.dng', **kw)
    monkeypatch.setattr(gpu, 'available', lambda: False)
    with pytest.raises(StopIteration): D.develop('x.dng')
    assert D.develop('x.dng', backend='gpu').shape == (2, 2, 3)


def test_sharpening_matches_the_cpu_filter_and_picture():
    from stratastills import detail
    rng = np.random.default_rng(2); f = np.linspace(0.002, 0.498, 128); m = np.exp(-2 * np.pi ** 2 * 1.6 ** 2 * f ** 2); img = np.clip(picture(9) * 0.6 + 0.2, 0, 1)
    for scale in (1.0, 0.746):
        g_cpu = detail.sharpen_gain(f, m, (H, W), scale, amount=1.5); g_dev = gpu.sharpen_gain(f, m, (H, W), scale, detail.NOISE_TERM, 4.0, 1.5, detail.ROLLOFF).cpu().numpy(); assert np.abs(g_dev - g_cpu[:, :W // 2 + 1]).max() < 2e-4
    ref = detail.sharpen(img, detail.sharpen_gain(f, m, (H, W), 1.0, amount=1.5)); out = c(gpu.sharpen(g(img), gpu.sharpen_gain(f, m, (H, W), 1.0, detail.NOISE_TERM, 4.0, 1.5, detail.ROLLOFF))); assert np.abs(out - ref).max() < 2e-3
    a = detail.sharpen_picture(img, f, m, 0.746, 1.5, backend='cpu'); b = detail._sharpen_picture_gpu(img, f, m, 0.746, 1.5); assert np.abs(a - b).max() < 3e-3 and np.abs(a - b).mean() < 1e-4
    flat = np.full((64, 64, 3), 0.4, np.float32); assert np.allclose(detail._sharpen_picture_gpu(flat, f, m, 1.0, 1.5), 0.4, atol=1e-3)


@pytest.mark.parametrize('blend', [0.0, 0.5, 1.0])
def test_finish_with_the_hue_preserving_tone_curve_matches(blend):
    img = picture(11, scale=3.0); lut = np.cumsum(RNG.random(4096)).astype(np.float32); lut /= lut[-1]; lut = np.sqrt(lut); lt = gpu.torch_().from_numpy(lut).to(gpu.dev())
    assert np.abs(c(gpu.finish(g(img), lt, D.KNEE, D.WHITE, blend)) - D.finish(img, lut, blend)).max() < 1e-3


def test_hue_sat_map_and_chroma_limit_match():
    x = (RNG.random((50, 60, 3)) * 1.5).astype(np.float32); tb = np.stack([RNG.normal(0, 8, (36, 10)), 1 + RNG.normal(0, .1, (36, 10)), 1 + RNG.normal(0, .05, (36, 10))], -1).astype(np.float32)
    assert np.abs(c(gpu.apply_hue_sat_map(g(x), gpu.torch_().from_numpy(tb).to(gpu.dev()), rows=16)) - D.apply_hue_sat_map(x, tb)).max() < 1e-4
    y = RNG.random((50, 60, 3)).astype(np.float32); assert np.abs(c(gpu.limit_chroma(g(y), 0.2, 0.45)) - D.limit_chroma(y, 0.2, 0.45)).max() < 1e-4


def test_the_tiled_development_agrees_with_the_whole_picture_one(monkeypatch):
    from stratastills import gpu_tiled as T
    synthetic_dng(monkeypatch, h=520, w=680); u16 = D._raw_u16('x.dng', False)
    whole = D._develop_gpu_on_device('x.dng', u16, False, 'auto', 'auto', 0.35, D.HIGHLIGHT_CONTRAST, {}); a, b = {}, {}
    D._develop_gpu_on_device('x.dng', u16, False, 'auto', 'auto', 0.35, D.HIGHLIGHT_CONTRAST, a); tiled = T.develop_tiled('x.dng', u16, 'auto', 'auto', 0.35, D.HIGHLIGHT_CONTRAST, b, tile=192, halo=T.HALO)
    assert tiled.shape == whole.shape and tiled.dtype == np.float32 and np.isfinite(tiled).all() and np.abs(a['luma'] - b['luma']).max() < 1e-6
    d = np.abs(tiled - whole); assert d.mean() < 1.5e-3 and np.percentile(d, 99.9) < 0.04, (d.mean(), np.percentile(d, 99.9))
    seam = np.zeros(d.shape[:2], bool); seam[190:194] = True; seam[:, 190:194] = True; assert d[seam].mean() < 3 * d.mean() + 1e-3                              # no seams where the tiles meet


def test_a_picture_too_big_for_the_card_is_developed_in_tiles_and_otherwise_not(monkeypatch):
    from stratastills import gpu_tiled as T
    synthetic_dng(monkeypatch, h=120, w=160); calls = []; monkeypatch.setattr(T, 'develop_tiled', lambda *a, **k: calls.append('tiled') or np.zeros((120, 160, 3), np.float32))
    monkeypatch.setattr(gpu, 'max_pixels', lambda: 1000); D.develop('x.dng', backend='gpu'); assert calls == ['tiled']
    calls.clear(); monkeypatch.setattr(gpu, 'max_pixels', lambda: None); D.develop('x.dng', backend='gpu'); assert calls == []
    monkeypatch.setattr(gpu, 'max_pixels', lambda: 4800); monkeypatch.setattr(T, 'develop_tiled', lambda *a, **k: (_ for _ in ()).throw(T.Unsupported('x'))); out = D.develop('x.dng', backend='gpu'); assert out.shape[0] * out.shape[1] <= 4800     # falls back to a reduced copy


def test_the_cpu_tiled_development_agrees_with_the_whole_picture_one(monkeypatch):
    from stratastills import cpu_tiled
    synthetic_dng(monkeypatch, h=520, w=680); u16 = D._raw_u16('x.dng', False); a, b = {}, {}
    for denoise in ('auto', 'chroma', 'off'):
        whole = D.develop('x.dng', denoise=denoise, backend='cpu', analysis=a); tiled = cpu_tiled.develop_tiled('x.dng', u16, 'auto', denoise, 0.35, D.HIGHLIGHT_CONTRAST, b, tile=192)
        assert tiled.shape == whole.shape and tiled.dtype == np.float32 and np.isfinite(tiled).all() and np.abs(a['luma'] - b['luma']).max() < 1e-5
        d = np.abs(tiled - whole); assert d.mean() < 1.5e-3 and np.percentile(d, 99.9) < 0.04, (denoise, d.mean(), np.percentile(d, 99.9))
        seam = np.zeros(d.shape[:2], bool); seam[190:194] = True; seam[:, 190:194] = True; assert d[seam].mean() < 3 * d.mean() + 1e-3


def test_the_cpu_development_goes_through_tiles_when_memory_is_short(monkeypatch):
    from stratastills import cpu_tiled, resources
    synthetic_dng(monkeypatch, h=120, w=160); calls = []; monkeypatch.setattr(cpu_tiled, 'develop_tiled', lambda *a, **k: calls.append(a[-1] if len(a) > 7 else k.get('tile')) or np.zeros((120, 160, 3), np.float32))
    monkeypatch.setattr(resources, 'cpu_tile', lambda px: 512); D.develop('x.dng', backend='cpu'); assert calls == [512]
    calls.clear(); monkeypatch.setattr(resources, 'cpu_tile', lambda px: None); D.develop('x.dng', backend='cpu'); assert calls == []
    monkeypatch.setattr(resources, 'cpu_tile', lambda px: 512); D.develop('x.dng', backend='cpu', denoise=4.0); assert calls == []                                          # a numeric denoise is not tiled
