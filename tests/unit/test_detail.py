import cv2, numpy as np, pytest
from stratastills import detail


def texture(w=2400, h=1800, seed=1):
    """A sharp natural-like picture: a random field with a 1/f power spectrum (alpha 2), scaled to 0.1..0.9."""
    rng = np.random.default_rng(seed); f = np.fft.fftfreq(w)[None, :], np.fft.fftfreq(h)[:, None]
    r = np.hypot(*np.broadcast_arrays(*f)); r[0, 0] = 1
    z = np.fft.ifft2(np.fft.fft2(rng.standard_normal((h, w))) / r ** 1.0).real
    z = (z - z.min()) / (z.max() - z.min()); return (0.1 + 0.8 * z).astype(np.float32)


def test_sharp_picture_keeps_all_its_pixels():
    r = detail.effective_resolution(texture())
    assert r['ok'] and r['scale20'] > 0.9 and r['alpha'] == pytest.approx(2.0, abs=0.5)


@pytest.mark.parametrize('sigma', [1.0, 1.5, 2.5])
def test_blur_lowers_the_effective_size_as_theory_says(sigma):
    y = cv2.GaussianBlur(texture(), (0, 0), sigma); r = detail.effective_resolution(y)
    f20 = np.sqrt(np.log(5) / (2 * np.pi ** 2 * sigma ** 2))                    # Gaussian MTF exp(-2 pi^2 s^2 f^2) = 0.2
    assert r['f20'] == pytest.approx(f20, rel=0.2) and r['scale20'] == pytest.approx(2 * f20, rel=0.2) and r['f50'] < r['f30'] < r['f20']


def test_noise_floor_does_not_fool_it():
    y = cv2.GaussianBlur(texture(), (0, 0), 2.0); y = y + np.random.default_rng(3).normal(0, 0.004, y.shape).astype(np.float32)
    r = detail.effective_resolution(np.clip(y, 0, 1)); assert r['scale20'] == pytest.approx(2 * np.sqrt(np.log(5) / (2 * np.pi ** 2 * 4)), rel=0.25)


def test_flat_picture_cannot_be_measured():
    r = detail.effective_resolution(np.full((1024, 1024), 0.5, np.float32)); assert not r['ok']
    assert detail.recommended_size(r, 8000) is None


def test_recommended_size_rounds_up_and_never_exceeds_native():
    assert detail.recommended_size(dict(ok=True, scale30=0.58), 8064) == 4736 and detail.recommended_size(dict(ok=True, scale30=1.0), 8064) == 8064


def test_sharpening_restores_a_known_blur_without_halos():
    sharp = texture(1200, 900); blurred = cv2.GaussianBlur(sharp, (0, 0), 1.5); r = detail.effective_resolution(blurred)
    g = detail.sharpen_gain(np.array(r['freq']), np.array(r['mtf']), blurred.shape, 1.0)
    out = detail.sharpen(np.repeat(blurred[..., None], 3, 2), g)[..., 0]
    assert np.abs(out - sharp).mean() < np.abs(blurred - sharp).mean() * 0.85                  # closer to the truth than the blurred picture
    assert out.min() >= blurred.min() - 0.06 and out.max() <= blurred.max() + 0.06            # no overshoot beyond the local range
    flat = np.full((64, 64, 3), 0.4, np.float32); assert np.allclose(detail.sharpen(flat, detail.sharpen_gain(np.array(r['freq']), np.array(r['mtf']), (64, 64), 1.0)), 0.4, atol=1e-3)


def test_sharpen_amount_goes_past_restoration():
    f = np.linspace(0.002, 0.498, 128); m = np.exp(-2 * np.pi ** 2 * 1.5 ** 2 * f ** 2); g1 = detail.sharpen_gain(f, m, (64, 64), 1.0); g2 = detail.sharpen_gain(f, m, (64, 64), 1.0, amount=1.6)
    assert g2.max() >= g1.max() and (g2 >= g1 - 1e-6).all() and g2.max() <= 4.0 + 1e-6
