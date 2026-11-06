import numpy as np, pytest
from stratastills import denoise_ai as A, develop as D


def test_weight_map_is_zero_when_noise_is_invisible_and_capped_when_strong():
    w = A.weight_map(np.array([0.0, 1.5, 3.25, 5.0, 30.0], np.float32)); assert w[0] == 0 and w[1] == 0 and 0.3 < w[2] < 0.5 and w[3] == pytest.approx(0.9) and w[4] == pytest.approx(0.9)


def test_feather_weights_ramp_only_at_tile_borders_that_have_a_neighbour():
    f = A._feather(200, False, False, 20); assert f[0] < 0.05 and f[-1] < 0.05 and f[100] == 1.0
    g = A._feather(200, True, True, 20); assert (g == 1).all()
    assert A._feather(200, True, False, 20)[0] == 1 and A._feather(200, True, False, 20)[-1] < 0.05


def test_keep_tones_gives_back_the_broad_colour_and_keeps_fine_changes():
    rng = np.random.default_rng(0); orig = np.full((128, 192, 3), 0.4, np.float32); fine = rng.normal(0, 0.02, orig.shape).astype(np.float32)
    den = np.clip(orig + 0.05 + fine, 0, 1)                                           # the model lifted everything by 0.05 and added fine detail
    out = A.keep_tones(orig, den); assert abs(float(out.mean()) - 0.4) < 0.005 and float(out.std()) > 0.015


def test_adaptive_leaves_a_clean_picture_untouched_without_running_the_model(monkeypatch):
    monkeypatch.setattr(A, 'denoise', lambda *a, **k: pytest.fail('the model must not run when no noise is visible'))
    x = np.random.default_rng(1).random((64, 64, 3)).astype(np.float32); assert A.adaptive(x, 0.5) is x


def test_adaptive_blends_by_the_noise_map_and_only_asks_for_noisy_tiles(monkeypatch):
    asked = []; checker = np.indices((64, 64)).sum(0) % 2 * 2.0 - 1.0                                                                    # +-1 per pixel: a fine change that keep_tones must leave alone
    def fake(rgb, model=None, strength=None, need=None): asked.append(need(0, 64, 0, 64)); return (rgb + 0.2 * checker[..., None]).astype(np.float32)
    monkeypatch.setattr(A, 'denoise', fake); x = np.full((64, 64, 3), 0.5, np.float32); s = np.zeros((64, 64), np.float32); s[:, 32:] = 10.0
    out = A.adaptive(x, s); assert asked == [True] and np.abs(out[:, :16] - 0.5).max() < 1e-6 and np.abs(out[:, 40:] - 0.5).max() > 0.1             # untouched where the noise is invisible, the cleaned picture where it is plain


def test_develop_uses_the_model_where_the_profile_says_noise_is_visible(monkeypatch):
    rng = np.random.default_rng(5); raw = (rng.random((60, 80, 3)) * 0.03 + 0.01).astype(np.float32); calls = []
    tags = dict(NoiseProfile=(0.0004, 1e-7)); monkeypatch.setattr(D, 'read_dng_tags', lambda p, lens='auto': tags); monkeypatch.setattr(D, '_raw_u16', lambda p, half: (raw * 65535).astype(np.uint16))
    monkeypatch.setattr(A, 'available', lambda m=None: True); monkeypatch.setattr(A, 'adaptive', lambda rgb, s, m=None, st=None: calls.append(s.shape) or rgb)
    out = D.develop('x.dng', half=True, denoise='ai'); assert calls == [(60, 80)] and out.shape == raw.shape
    calls.clear(); D.develop('x.dng', half=True); assert calls == []                                                                              # the default is the classical clean-up: no model
    calls.clear(); D.develop('x.dng', half=True, denoise='chroma'); D.develop('x.dng', half=True, denoise='off'); assert calls == []
    monkeypatch.setattr(A, 'available', lambda m=None: False); D.develop('x.dng', half=True); assert calls == []                                          # no model installed: the light corner pass is used instead


def test_noise_map_follows_brightness_and_profile():
    dark = np.full((64, 64, 3), 0.01, np.float32); bright = np.full((64, 64, 3), 0.5, np.float32); n = D.noise_map8(dark, (0.001, 1e-6))
    assert n.shape == (64, 64) and float(n.mean()) > float(D.noise_map8(bright, (0.001, 1e-6)).mean()) and float(D.noise_map8(dark, (0.002, 1e-6)).mean()) > float(n.mean())


@pytest.mark.parametrize('model', A.MODELS)
def test_models_remove_noise_in_a_real_run(model):
    if not A.available(model): pytest.skip(f'{model} weights not downloaded (models/)')
    pytest.importorskip('torch'); pytest.importorskip('spandrel')
    rng = np.random.default_rng(2); clean = np.clip(0.4 + 0.2 * np.sin(np.linspace(0, 12, 256))[None, :, None] * np.ones((192, 1, 3)), 0, 1).astype(np.float32)
    noisy = np.clip(clean + rng.normal(0, 0.04, clean.shape), 0, 1).astype(np.float32); out = A.denoise(noisy, model, 1.0, tile=128, overlap=16)
    assert out.shape == noisy.shape and np.abs(out - clean).mean() < np.abs(noisy - clean).mean() * 0.7


def test_unknown_model_is_refused():
    with pytest.raises(ValueError): A.denoise(np.zeros((8, 8, 3), np.float32), 'nope')


def test_an_out_of_memory_halves_the_tile_and_goes_on(monkeypatch):
    pytest.importorskip("torch")
    seen = []

    def run(net, t, mult, post):
        seen.append(t.shape[0])
        if t.shape[0] > 300: raise torch.OutOfMemoryError('CUDA out of memory')
        return t
    monkeypatch.setattr(A, '_load', lambda m, s: object()); monkeypatch.setattr(A, '_run', run)
    x = np.full((600, 600, 3), 0.4, np.float32); out = A.denoise(x, tile=512, overlap=8)
    assert seen[0] == 512 and max(seen[1:]) <= 256 and np.allclose(out, x)

def test_fetch_downloads_models(monkeypatch):
    import os, urllib.request
    downloaded = []
    monkeypatch.setattr(os.path, 'exists', lambda p: False)
    monkeypatch.setattr(os, 'makedirs', lambda d, exist_ok: None)
    monkeypatch.setattr(urllib.request, 'urlretrieve', lambda url, p: downloaded.append(url))
    A.fetch('esrgan')
    assert len(downloaded) == 2
    assert 'realesr-general-x4v3.pth' in downloaded[0]

def test_fp16_returns_true_for_capable_cuda_and_false_for_rocm(monkeypatch):
    import sys
    class FakeDev:
        def __init__(self, typ): self.type = typ

    assert A._fp16(FakeDev('cpu')) is False

    class FakeTorch:
        class version:
            hip = True
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch)
    assert A._fp16(FakeDev('cuda')) is False

    class FakeTorchCuda:
        class version:
            hip = False
        class cuda:
            @staticmethod
            def get_device_capability(dev): return (7, 0)
    monkeypatch.setitem(sys.modules, 'torch', FakeTorchCuda)
    assert A._fp16(FakeDev('cuda')) is True

    class FakeTorchCudaOld:
        class version:
            hip = False
        class cuda:
            @staticmethod
            def get_device_capability(dev): return (6, 1)
    monkeypatch.setitem(sys.modules, 'torch', FakeTorchCudaOld)
    assert A._fp16(FakeDev('cuda')) is False

    class FakeTorchCudaErr:
        class version:
            hip = False
        class cuda:
            @staticmethod
            def get_device_capability(dev): raise Exception()
    monkeypatch.setitem(sys.modules, 'torch', FakeTorchCudaErr)
    assert A._fp16(FakeDev('cuda')) is True

def test_device_returns_cuda_if_available(monkeypatch):
    import sys
    from stratastills import gpu
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available(): return True
        @staticmethod
        def device(d): return d
    monkeypatch.setattr(gpu, 'torch_', lambda: FakeTorch)
    assert A._device() == 'cuda'

def test_load_models_without_real_torch(monkeypatch):
    import sys
    class FakeDev:
        type = 'cpu'
    monkeypatch.setattr(A, '_device', lambda: FakeDev())
    monkeypatch.setattr(A, '_fp16', lambda dev: False)
    class FakeTorch:
        @staticmethod
        def load(p, map_location):
            if 'wdn' in p: return {'params': {'a': 2.0}}
            return {'params': {'a': 1.0}}
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch)
    class FakeModel:
        def __init__(self, val=None): self.val = val
        def eval(self): return self
        def to(self, dev): return self
    class FakeModelLoader:
        def load_from_state_dict(self, d): return type('T', (), {'model': FakeModel(d)})()
        def load_from_file(self, p): return type('T', (), {'model': FakeModel(p)})()
    monkeypatch.setitem(sys.modules, 'spandrel', type('M', (), {'ModelLoader': FakeModelLoader})())

    A._models.clear()
    esrgan_net = A._load('esrgan', 0.6)
    assert esrgan_net.val['a'] == pytest.approx(1.6)

    scunet_net = A._load('scunet', 0.6)
    assert 'scunet' in scunet_net.val

def test_run_pad_and_unpad(monkeypatch):
    import sys
    class FakeTensor:
        def __init__(self, arr):
            self.arr = arr
            self.dtype = 'float32'
        def to(self, *a, **k): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return self.arr
        def __getitem__(self, idx): return self

    class FakeNet:
        def __init__(self): pass
        def parameters(self): yield type('P', (), {'device': 'cpu', 'dtype': 'float32'})()
        def __call__(self, x): return FakeTensor(x.arr)

    class FakeTorch:
        @staticmethod
        def no_grad(): return type('Ctx', (), {'__enter__': lambda s: None, '__exit__': lambda s, *a: None})()
        @staticmethod
        def from_numpy(a): return FakeTensor(a)
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch)

    tile = np.zeros((10, 10, 3), np.float32)
    # mult = 8 -> padded shape 16x16
    out = A._run(FakeNet(), tile, 8, lambda y, shape: y)
    assert out.shape == (10, 10, 3)

def test_denoise_oom_loop(monkeypatch):
    from stratastills import gpu
    monkeypatch.setattr(A, '_load', lambda m, s: object())

    called_tiles = []
    def fake_tiles(rgb, net, tile, overlap, mult, post, need):
        called_tiles.append(tile)
        if tile > 256:
            raise RuntimeError('CUDA out of memory')
        return rgb, np.ones_like(rgb)
    monkeypatch.setattr(A, '_tiles', fake_tiles)
    monkeypatch.setattr(gpu, 'is_oom', lambda e: 'memory' in str(e))
    monkeypatch.setattr(gpu, 'free', lambda: None)

    rgb = np.zeros((16, 16, 3), np.float32)
    out = A.denoise(rgb, model='esrgan', tile=1024)
    assert out.shape == (16, 16, 3)
    assert called_tiles == [1024, 512, 256]

def test_denoise_oom_fail_if_not_oom(monkeypatch):
    from stratastills import gpu
    monkeypatch.setattr(A, '_load', lambda m, s: object())

    def fake_tiles(rgb, net, tile, overlap, mult, post, need):
        raise RuntimeError('other error')
    monkeypatch.setattr(A, '_tiles', fake_tiles)
    monkeypatch.setattr(gpu, 'is_oom', lambda e: 'memory' in str(e))
    monkeypatch.setattr(gpu, 'free', lambda: None)

    rgb = np.zeros((16, 16, 3), np.float32)
    with pytest.raises(RuntimeError):
        A.denoise(rgb, model='esrgan', tile=1024)

def test_tiles_logic(monkeypatch):
    monkeypatch.setattr(A, '_run', lambda net, t, mult, post: t * 2)
    rgb = np.ones((100, 100, 3), np.float32)
    # tile 64, overlap 16
    out, wsum = A._tiles(rgb, None, 64, 16, 4, lambda y, shape: y, need=None)
    assert out.shape == (100, 100, 3)
    assert wsum.shape == (100, 100, 1)

    # Check need branch
    called = []
    def fake_need(y0, y1, x0, x1):
        called.append(True)
        return False
    out, wsum = A._tiles(rgb, None, 64, 16, 4, lambda y, shape: y, need=fake_need)
    assert out.shape == (100, 100, 3)
    assert len(called) > 0

def test_denoise_scunet_branch(monkeypatch):
    monkeypatch.setattr(A, '_load', lambda m, s: object())
    monkeypatch.setattr(A, '_tiles', lambda *a, **k: (np.zeros((10, 10, 3)), np.ones((10, 10, 3))))
    out = A.denoise(np.zeros((10, 10, 3)), model='scunet')
    assert out.shape == (10, 10, 3)

def test_denoise_oom_no_device_to_free(monkeypatch):
    from stratastills import gpu
    monkeypatch.setattr(A, '_load', lambda m, s: object())

    called_tiles = []
    def fake_tiles(rgb, net, tile, overlap, mult, post, need):
        called_tiles.append(tile)
        if tile > 256:
            raise RuntimeError('CUDA out of memory')
        return rgb, np.ones_like(rgb)
    monkeypatch.setattr(A, '_tiles', fake_tiles)
    monkeypatch.setattr(gpu, 'is_oom', lambda e: 'memory' in str(e))
    monkeypatch.setattr(gpu, 'free', lambda: (_ for _ in ()).throw(RuntimeError('no device')))

    rgb = np.zeros((16, 16, 3), np.float32)
    out = A.denoise(rgb, model='esrgan', tile=1024)
    assert out.shape == (16, 16, 3)
    assert called_tiles == [1024, 512, 256]
