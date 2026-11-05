"""The nudity filter (nsfw.py) with a fake detector: flagged pictures leave `ingest.scan`, so every consumer skips them; threshold, cache, log, switch."""
import json, os

import pytest
from photos import make_jpeg
from stratastills import ingest, nsfw


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_NSFW', '1'); nsfw.reset(); nsfw.set_detector(None)
    d = tmp_path / 'shoot'; d.mkdir()
    for i, n in enumerate(('A', 'B', 'C')): make_jpeg(str(d / f'{n}.jpg'), w=200, h=150, seed=i + 1)
    yield str(d)
    nsfw.set_detector(None); nsfw.reset()


def fake(scores):
    """A detector whose answer depends on the picture size seed: `scores` is consumed per call."""
    calls = []
    def det(bgr):
        calls.append(1); return [dict(scores[(len(calls) - 1) % len(scores)])]
    det.calls = calls; return det


def test_flagged_picture_is_not_listed(folder):
    nsfw.set_detector(fake([dict(**{'class': 'FEMALE_BREAST_EXPOSED', 'score': 0.9}), dict(**{'class': 'FACE_FEMALE', 'score': 0.99}), dict(**{'class': 'BUTTOCKS_COVERED', 'score': 0.8})]))
    assert [s.stem for s in ingest.scan(folder)] == ['B', 'C']                                  # only exposed classes count


def test_threshold_is_configurable(folder, monkeypatch):
    nsfw.set_detector(fake([{'class': 'ANUS_EXPOSED', 'score': 0.6}]))
    monkeypatch.setenv('STRATASTILLS_NSFW_THRESHOLD', '0.7'); assert len(ingest.scan(folder)) == 3
    monkeypatch.setenv('STRATASTILLS_NSFW_THRESHOLD', '0.5'); assert ingest.scan(folder) == []


def test_off_switch_lists_everything_and_never_loads_a_detector(folder, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_NSFW', '0'); nsfw.set_detector(lambda b: 1 / 0); assert len(ingest.scan(folder)) == 3


def test_result_is_cached_and_skips_are_logged(folder):
    det = fake([{'class': 'MALE_GENITALIA_EXPOSED', 'score': 0.8}]); nsfw.set_detector(det)
    ingest.scan(folder); n = len(det.calls); nsfw.reset(); ingest.scan(folder)
    assert n == 3 and len(det.calls) == 3                                                       # second scan read the cache file
    rows = [json.loads(l) for l in open(os.path.join(nsfw.home(), 'nsfw_skipped.jsonl'))]
    assert {r['cls'] for r in rows} == {'MALE_GENITALIA_EXPOSED'} and all(r['reason'] == 'nudity' for r in rows)


def test_missing_detector_fails_loudly(folder, monkeypatch):
    import builtins
    real = builtins.__import__
    monkeypatch.setattr(builtins, '__import__', lambda n, *a, **k: (_ for _ in ()).throw(ImportError()) if n == 'nudenet' else real(n, *a, **k))
    with pytest.raises(nsfw.NsfwUnavailable): ingest.scan(folder)


def test_concurrent_scores_do_not_corrupt_the_cache(folder):
    from concurrent.futures import ThreadPoolExecutor
    nsfw.set_detector(fake([{'class': 'FACE_FEMALE', 'score': 0.9}]))
    paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
    with ThreadPoolExecutor(6) as ex: assert len(list(ex.map(nsfw.score, paths * 4))) == 12
    nsfw.reset(); assert len(json.load(open(os.path.join(nsfw.home(), 'nsfw_cache.json')))) == 3
