import numpy as np, pytest
from stratastills import jpeg_quality as J
from test_detail import texture


def u8(a): return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def picture(noise, w=1536, h=1024, seed=3):
    """Smooth light and shade with a little detail, plus noise of `noise` (0..1): a photograph of a plain wall, where the picture's own noise is all there is to hide JPEG error."""
    import cv2
    base = cv2.GaussianBlur(texture(w, h), (0, 0), 14) * 1.6 - 0.3; base = np.clip(base, 0.05, 0.95); rng = np.random.default_rng(seed); n = rng.normal(0, noise, (h, w)).astype(np.float32)
    return u8(np.dstack([base + n, base * 0.9 + n, base * 0.8 + n]))


def test_noise_sigma_recovers_the_noise_of_a_smooth_tile():
    rng = np.random.default_rng(0); a = np.clip(0.5 + rng.normal(0, 4 / 255, (256, 256, 1)), 0, 1) * np.ones(3); assert J.noise_sigma(u8(a)) == pytest.approx(4.0, abs=0.7)


def test_a_grainy_picture_needs_less_quality_than_a_clean_one():
    clean_q, ci = J.choose_quality(picture(0.003)); noisy_q, ni = J.choose_quality(picture(0.03))
    assert noisy_q < clean_q and J.MIN_Q <= noisy_q and clean_q <= J.MAX_Q and ni['sigma'] > ci['sigma']
    assert ni['rms'] <= ni['target'] + 0.15 and J.FLOOR <= ci['target'] <= J.CEIL


def test_the_chosen_quality_is_the_lowest_that_meets_the_target():
    img = picture(0.01); q, info = J.choose_quality(img); tiles, _ = J.pick_tiles(img)
    assert J._rms(img, tiles, q) <= info['target'] + 1e-9 and (q == J.MIN_Q or J._rms(img, tiles, q - 1) > info['target'])


def test_tiny_and_flat_pictures_are_handled():
    assert J.choose_quality(np.zeros((16, 16, 3), np.uint8))[0] == J.MAX_Q
    q, info = J.choose_quality(np.full((600, 800, 3), 128, np.uint8)); assert J.MIN_Q <= q <= J.MAX_Q and info['target'] == J.FLOOR
    small, _ = J.pick_tiles(np.zeros((300, 400, 3), np.uint8)); assert small[0][2] == 288 and small[0][2] % 16 == 0
