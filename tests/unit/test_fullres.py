"""fullres.py: the background full-size copy for the loupe: status, building (with the fallbacks) and the memory-mapped read."""
import cv2, numpy as np, pytest

from stratastills import fullres, images


@pytest.fixture(autouse=True)
def _own_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(fullres, '_jobs', {})
    monkeypatch.setattr(fullres, 'cache_file', lambda pid, path: str(tmp_path / 'full.npy'))


def _jpg(tmp_path, h=12, w=20):
    p = str(tmp_path / 'a.jpg'); cv2.imwrite(p, np.full((h, w, 3), 200, np.uint8)); return p


def test_a_picture_nobody_asked_for_is_idle_and_one_that_is_made_is_ready(tmp_path):
    assert fullres.status('p', 'x.jpg') == dict(state='idle', text='') and fullres.get('p', 'x.jpg') is None
    p = _jpg(tmp_path); fullres._build('p', p, fullres.cache_file('p', p))
    assert fullres.status('p', p) == dict(state='ready', text='')
    a = fullres.get('p', p)
    assert a.shape == (12, 20, 3) and a.dtype == np.uint8 and not a.flags.writeable                      # a read-only memory map, not a copy


def test_a_build_that_fails_is_reported_as_failed_with_the_reason_and_leaves_no_file(tmp_path):
    f = fullres.cache_file('p', 'nope.jpg'); fullres._build('p', str(tmp_path / 'nope.jpg'), f)
    s = fullres.status('p', 'nope.jpg')
    assert s['state'] == 'failed' and 'cannot read nope.jpg' in s['text'] and fullres.get('p', 'nope.jpg') is None


def test_a_raw_that_will_not_develop_falls_back_to_its_biggest_embedded_level(monkeypatch, tmp_path):
    def boom(*a, **k): raise RuntimeError('LibRaw cannot')
    monkeypatch.setattr(fullres.develop, 'develop', boom); monkeypatch.setattr(images, 'load', lambda path, px: np.full((3, 4, 3), 9, np.uint8))
    f = fullres.cache_file('p', 'lr.dng'); fullres._build('p', 'lr.dng', f)
    assert np.load(f).shape == (3, 4, 3)


def test_a_developed_raw_is_stored_as_8_bit_bgr(monkeypatch):
    monkeypatch.setattr(fullres.develop, 'develop', lambda path, half: np.dstack([np.ones((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))]))   # RGB floats, red
    f = fullres.cache_file('p', 'a.arw'); fullres._build('p', 'a.arw', f)
    assert np.load(f)[0, 0].tolist() == [0, 0, 255]


def test_a_damaged_copy_reads_as_not_made(tmp_path):
    (tmp_path / 'full.npy').write_bytes(b'garbage')
    assert fullres.get('p', 'x.jpg') is None


def test_prepare_starts_one_job_and_does_not_start_a_second_while_it_runs(monkeypatch, tmp_path):
    started = []
    class T:
        def __init__(self, target, args, daemon): started.append(args)
        def start(self): pass
    monkeypatch.setattr(fullres.threading, 'Thread', T)
    s = fullres.prepare('p', 'x.jpg'); assert s['state'] == 'running' and 'decoding' in s['text']
    assert fullres.prepare('p', 'x.jpg')['state'] == 'running' and len(started) == 1
    (tmp_path / 'full.npy').write_bytes(b'x'); assert fullres.prepare('p', 'x.jpg') == dict(state='ready', text='')


def test_the_text_of_a_job_says_what_is_slow(monkeypatch):
    monkeypatch.setattr(fullres.threading, 'Thread', lambda target, args, daemon: type('T', (), {'start': lambda self: None})())
    assert 'developing the raw' in fullres.prepare('p', 'y.arw')['text']


def test_prepare_tries_a_failed_job_again(monkeypatch):
    started = []
    monkeypatch.setattr(fullres.threading, 'Thread', lambda target, args, daemon: type('T', (), {'start': lambda self: started.append(1)})())
    fullres._jobs[fullres.cache_file('p', 'x.jpg')] = dict(state='failed', text='boom')
    assert fullres.prepare('p', 'x.jpg')['state'] == 'running' and started == [1]


def test_dims_is_the_size_of_the_file(tmp_path):
    assert fullres.dims(_jpg(tmp_path, 7, 11)) == (11, 7)
