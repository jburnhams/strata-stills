"""The JPEG written by the card's encoder (export.encode_jpeg_gpu) and the picture that stays on the card between resize, sharpen and encode (pipeline.State)."""
import io

import numpy as np
import pytest
from PIL import Image

from stratastills import export as X, gpu, pipeline as PL


def rgb(h=96, w=128, seed=0):
    rng = np.random.default_rng(seed); x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]; y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return np.clip(0.2 + 0.5 * x * y + 0.05 * rng.standard_normal((h, w, 3)).astype(np.float32), 0, 1).astype(np.float32)


def test_exif_and_icc_segments_survive_in_a_jpeg_made_without_them():
    base = io.BytesIO(); Image.fromarray(X.to_u8(rgb())).save(base, 'JPEG', quality=90)
    data = base.getvalue(); i = 4 + int.from_bytes(data[4:6], 'big'); icc = X._icc(); ex = Image.new('RGB', (2, 2)).getexif(); ex[0x010F] = 'Test camera'; exif = ex.tobytes()
    out = data[:i] + X._app_segments(exif, icc) + data[i:]; im = Image.open(io.BytesIO(out)); im.load()
    assert im.info['icc_profile'] == icc and im.getexif()[0x010F] == 'Test camera' and im.size == (128, 96)


def test_a_large_icc_profile_is_split_into_chunks_and_read_back():
    icc = bytes(range(256)) * 600                                                                                          # 153 600 bytes: three APP2 segments
    base = io.BytesIO(); Image.fromarray(X.to_u8(rgb())).save(base, 'JPEG'); data = base.getvalue(); i = 4 + int.from_bytes(data[4:6], 'big')
    segs = X._app_segments(None, icc); assert segs.count(b'ICC_PROFILE\0') == 3
    assert Image.open(io.BytesIO(data[:i] + segs + data[i:])).info['icc_profile'] == icc


def test_small_pictures_and_the_off_switch_use_pillow(tmp_path, monkeypatch):
    assert X.encode_jpeg_gpu(rgb(), 90) is None                                                                            # under NVJPEG_MIN_PIXELS
    monkeypatch.setattr(X, 'NVJPEG_MIN_PIXELS', 0); monkeypatch.setenv('STRATASTILLS_NVJPEG', '0'); assert X.encode_jpeg_gpu(rgb(), 90) is None
    p = str(tmp_path / 'a.jpg'); X.write_picture(p, rgb(), 'jpeg', 90); assert Image.open(p).size == (128, 96)


def test_the_card_encoder_writes_a_jpeg_close_to_pillows(tmp_path, monkeypatch):
    t = gpu.torch_()
    if not t.cuda.is_available(): pytest.skip('no CUDA')
    pytest.importorskip('torchvision'); gpu.use('cuda')
    try:
        monkeypatch.setattr(X, 'NVJPEG_MIN_PIXELS', 0); img = rgb(256, 320, 3); a = str(tmp_path / 'gpu.jpg'); X.write_picture(a, img, 'jpeg', 95)
        monkeypatch.setenv('STRATASTILLS_NVJPEG', '0'); b = str(tmp_path / 'cpu.jpg'); X.write_picture(b, img, 'jpeg', 95)
        ia, ib = Image.open(a), Image.open(b); assert ia.info['icc_profile'][:24] == ib.info['icc_profile'][:24]                  # the profile's date differs
        d = np.abs(np.asarray(ia.convert('RGB')).astype(int) - np.asarray(ib.convert('RGB')).astype(int)); assert d.mean() < 1.0 and d.max() <= 16                 # white noise of sigma 13 levels is the worst case for a DCT codec: real pictures differ by 0.25 on average
        src = X.to_u8(img).astype(float); err = lambda im: np.sqrt(((np.asarray(im.convert('RGB')).astype(float) - src) ** 2).mean()); assert abs(err(ia) - err(ib)) < 0.1
        tensor = gpu.to_gpu(img); monkeypatch.delenv('STRATASTILLS_NVJPEG'); c = str(tmp_path / 'dev.jpg'); X.write_picture(c, tensor, 'jpeg', 95); assert open(c, 'rb').read() == open(a, 'rb').read()   # from the card's copy: the same bytes
    finally: gpu.use(None)


def state():
    return PL.State('p', 'i', {}, {'path': 'x.arw'}, lambda m: None)


def test_the_picture_may_live_on_the_card_and_comes_back_when_asked():
    gpu.use('cpu')
    try:
        st = state(); a = rgb(); st.rgb = a; assert st.dev is None and st.hw() == (96, 128)
        st.set_dev(gpu.to_gpu(a)); assert st._rgb is None and st.hw() == (96, 128) and np.array_equal(st.rgb, a)                  # downloaded on demand
        st.rgb = a * 0.5; assert st.dev is None                                                                                  # assigning an array replaces the card's copy
        st.set_dev(gpu.to_gpu(a)); st.final_hw = st.hw(); st.dev = None; st._rgb = None; assert st.hw() == (96, 128)               # after encode the size is still known
    finally: gpu.use(None)


def test_resize_on_the_card_equals_opencv_and_the_cpu_route_is_the_default():
    import cv2
    gpu.use('cpu')
    try:
        a = rgb(120, 160); st = state(); st.rgb = a; st.scale = 0.75; PL._resize(st, 120, 90); assert st.dev is None and st.rgb.shape == (90, 120, 3)   # on the host: OpenCV
        ref = cv2.resize(a, (120, 90), interpolation=cv2.INTER_LANCZOS4); assert np.array_equal(st.rgb, ref)
        st2 = state(); st2.set_dev(gpu.to_gpu(a)); st2.scale = 0.75; PL._resize(st2, 120, 90); assert st2.dev is not None and np.abs(st2.rgb - ref).max() < 2e-5
        st3 = state(); st3.set_dev(gpu.to_gpu(a)); st3.scale = 0.4; PL._resize(st3, 64, 48); assert np.abs(st3.rgb - cv2.resize(a, (64, 48), interpolation=cv2.INTER_AREA)).max() < 2e-5
    finally: gpu.use(None)


def test_stage_versions_changed_with_the_encoder_and_the_resize_path():
    assert PL.version('encode') >= 3 and PL.version('resize') % 1000 >= 3
