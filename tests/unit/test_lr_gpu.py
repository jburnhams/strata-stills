"""look/gpu.py (the Lightroom-look code and the arena's brightness match in PyTorch) held to its CPU originals in look/apply.py and scripts/arena_variants.py, on every device there is (torch-CPU always, CUDA / MPS where present), the way test_gpu.py holds gpu.py to develop.py. The scripts are not a package: the folder goes on sys.path."""

import os
import sys

import cv2
import numpy as np
import pytest
import torch  # noqa: F401  (a missing torch fails the test: CI installs it, see pyproject.toml)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import arena_variants as AV

from stratastills import gpu
from stratastills.look import apply as LK
from stratastills.look import features as D
from stratastills.look import gpu as G
from stratastills.look import transform as FIT

H, W = 233, 350  # not a multiple of 4: the quarter-size copy of the guided filter has fractional INTER_AREA weights


@pytest.fixture(params=["cpu", "cuda", "mps"], autouse=True)
def device(request):
    t = gpu.torch_()
    if request.param == "cuda" and not t.cuda.is_available():
        pytest.skip("no CUDA")
    if request.param == "mps" and not (getattr(t.backends, "mps", None) is not None and t.backends.mps.is_available()):
        pytest.skip("no MPS")
    gpu.use(request.param)
    yield request.param
    gpu.use(None)


def photo(seed=0):
    """A smooth, edgy, noisy uint8 BGR picture with dark and bright parts, a skin-coloured patch and some saturated colour."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((H, W, 3)).astype(np.float32), (0, 0), 5) * 3 - 0.9
    base[:, W // 2 :] += 0.3
    base[H // 2 :, : W // 4] *= 0.3
    img = np.clip(base + rng.normal(0, 0.02, base.shape).astype(np.float32), 0, 1)
    img[20:90, 30:110] = np.array([0.45, 0.6, 0.8], np.float32)
    return (img * 255 + 0.5).astype(np.uint8)


def pred():
    p = dict(zip(LK.TARGETS, [0.0] * len(LK.TARGETS)))
    for k, v in zip(D.FINAL_L, list(np.linspace(8, 92, len(D.PCTS))) + [48]):
        p[k] = float(v)
    p.update(a_mean=3.0, b_mean=9.0, C_p50=20.0, C_p90=45.0, C_mean=26.0, C_p99=60.0)
    p.update({k: d + o for (k, d), o in zip(LK.SLIDERS, [10, -20, 5, 5, 0, 25, 15, -15])})
    return np.array([p[k] for k in LK.TARGETS])


def colour():
    rng = np.random.default_rng(3)
    c = FIT.init_params(1, "cpu")[0].numpy().copy()
    return (c + rng.normal(0, 0.15, c.shape)).astype(np.float32)


def c(t):
    return t.cpu().numpy()


def g(a):
    return gpu.torch_().from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(gpu.dev())


def close_u8(a, b, mean=0.05, mx=10):
    d = np.abs(a.astype(int) - b.astype(int))
    assert d.mean() < mean and d.max() <= mx, (d.mean(), d.max())


def test_the_torch_colour_conversion_matches_opencv():
    im = photo()
    x = im.astype(np.float32) / 255
    ref = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    out = c(G.bgr_to_lab(G.up(im)))
    assert (
        np.abs(out - ref).max(axis=(0, 1)) < [0.25, 0.5, 0.35]
    ).all()  # OpenCV's float path is a spline table: off the exact formula by up to 0.19 in L
    lab = ref.copy()
    lab[..., 1:] *= 1.7
    back = c(G.lab_to_bgr(g(lab)))
    assert np.abs(back - cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)).max() < 1e-3


def test_both_colour_routes_give_the_same_picture(monkeypatch):
    im = photo()
    lab = G.lab_in(im)
    for mode in ("cv2", "torch"):
        monkeypatch.setattr(G, "LAB", mode)
        close_u8(G.lab_out(G.lab_in(im)), im, mean=0.1, mx=2)
        assert lab.shape == (H, W, 3)


@pytest.mark.parametrize("size", [(59, 88), (58, 87), (233, 175), (100, 350)])
def test_area_resize_is_opencv_inter_area(size):
    L = cv2.cvtColor(photo().astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0]
    assert (
        np.abs(c(G.resize_area(g(L), size)) - cv2.resize(L, (size[1], size[0]), interpolation=cv2.INTER_AREA)).max()
        < 1e-3
    )


def test_percentiles_and_interp_match_numpy():
    L = cv2.cvtColor(photo().astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0]
    assert np.abs(np.array(G.percentiles(g(L), D.PCTS)) - np.percentile(L, D.PCTS)).max() < 1e-3
    xp, fp = np.array([0, 20, 55, 100.0]), np.array([0, 30, 50, 100.0])
    assert np.abs(c(G.interp(g(L), xp, fp)) - np.interp(L, xp, fp)).max() < 1e-3


def test_the_lightness_steps_match():
    L = cv2.cvtColor(photo().astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0]
    t = g(L)
    assert np.abs(c(G._base(t)) - LK._base(L)).max() < 0.01
    for fn, args in ((LK.local_tone, (45, -60)), (LK.clarity, (35,)), (LK.vignette, (-25,))):
        assert np.abs(c(getattr(G, fn.__name__)(t, *args)) - fn(L, *args)).max() < 0.02, fn.__name__


def test_skin_rotation_matches():
    lab = cv2.cvtColor(photo().astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    L, a, b = (lab[..., i] for i in range(3))
    assert abs(G.skin_rotation(g(L), g(a), g(b)) - LK.skin_rotation(L, a, b)) < 0.05


@pytest.mark.parametrize("with_colour", [False, True])
def test_apply_look_matches(with_colour):
    im = photo(1)
    col = colour() if with_colour else None
    ref, did_ref = LK._apply_look(im, pred(), None, col)
    out, did = G._apply_look(im, pred(), None, col)
    close_u8(out, ref)
    assert did.keys() == did_ref.keys()
    if not with_colour:
        assert abs(did["chroma_scale"] - did_ref["chroma_scale"]) < 0.01


def test_anchor_face_matches():
    im = photo(2)
    box, target = (0.1, 0.1, 0.25, 0.3), (60.0, 15.0, 22.0)
    ref, did_ref = LK.anchor_face(im, box, target)
    out, did = G.anchor_face(im, box, target)
    close_u8(out, ref)
    assert did == did_ref


def test_apply_look_dispatches_on_the_device_and_off_it(monkeypatch):
    im = photo(3)
    face = dict(box=(0.1, 0.1, 0.25, 0.3), target=(60.0, 15.0, 22.0))
    a, _ = LK.apply_look(im, pred(), None, face=face, gpu=False)
    b, _ = LK.apply_look(im, pred(), None, face=face, gpu=True)
    close_u8(a, b)
    monkeypatch.setenv("STRATASTILLS_GPU", "0")
    gpu.use(None)
    assert not G.use()


def test_match_brightness_matches():
    im = photo(4)
    ref = (np.clip(im.astype(float) * 0.7 + 20, 0, 255)).astype(np.uint8)
    a = AV.match_brightness(im, ref, gpu=False)
    b = AV.match_brightness(im, ref, gpu=True)
    close_u8(a, b, mean=0.05, mx=3)
    assert (
        abs(
            float(cv2.cvtColor(b.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0].mean())
            - float(cv2.cvtColor(ref.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0].mean())
        )
        < 0.3
    )
