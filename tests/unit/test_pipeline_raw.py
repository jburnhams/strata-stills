"""Generating from a camera raw: the raw is decoded ONCE (the detail measurement and the tilt come from the luminance the develop pass already made)."""
import numpy as np, pytest
from PIL import Image

pytest.importorskip('torch')
from stratastills import develop as D, gpu, pipeline as PL, projects as P
from test_gpu import synthetic_dng


@pytest.fixture
def raw_project(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects')); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'home'))
    src = tmp_path / 'src'; src.mkdir(); (src / 'R_0001.DNG').write_bytes(b'not read: the decode is faked'); Image.fromarray((np.random.default_rng(0).random((240, 320, 3)) * 255).astype('uint8')).save(src / 'R_0001.JPG')
    d = P.create('R', [str(src)]); return d['id'], P.item_id(P.source_summaries(d['id'])[0]['id'], 'R_0001')


@pytest.mark.parametrize('path', ['gpu', 'cpu'])
def test_the_raw_is_decoded_once_and_the_picture_is_made(monkeypatch, raw_project, path):
    pid, iid = raw_project; synthetic_dng(monkeypatch); calls = []; real = D._raw_u16
    monkeypatch.setattr(D, '_raw_u16', lambda p, half: calls.append(half) or real(p, half))
    if path == 'gpu': gpu.use('cpu'); monkeypatch.setattr(gpu, 'available', lambda: True)                       # the GPU code path, run by torch on the CPU
    else: monkeypatch.setattr(gpu, 'available', lambda: False)                                                  # the NumPy / OpenCV reference
    try: r = PL.generate(pid, iid, 'raw', options=dict(size='native'))
    finally: gpu.use(None)
    assert r['steps'][0] == 'developed from the raw' and r['output'] == [320, 240] and 'detail' in r and 'level' in r
    assert calls == [False]                                                                                         # one decode on either path (the CPU path used to decode again for the analysis)
