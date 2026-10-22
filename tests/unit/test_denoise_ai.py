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
    tags = dict(NoiseProfile=(0.0004, 1e-7)); monkeypatch.setattr(D, 'read_dng_tags', lambda p: tags); monkeypatch.setattr(D, '_raw_linear', lambda p, half: raw.copy())
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
