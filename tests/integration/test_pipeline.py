"""Pipelines end to end on a synthetic drone shoot: a JPEG shot, a Lightroom-style linear-DNG panorama, the rules that apply a preset to a source folder, and the tiled developers against the whole-picture ones."""
import json, os

import cv2, numpy as np
import pytest

from stratastills import develop as D, gpu, pipeline as PL, projects as P

NADIR, OBLIQUE, PANO = 'DJI_20250402190111_0001_D', 'DJI_20250402190211_0002_D', 'DJI_20250402190111_0001_D-Pano'


def read(path): return cv2.imread(path)


def test_a_shot_is_generated_from_its_jpeg_and_is_found_again(project):
    pid, sid = project; iid = P.item_id(sid, NADIR); r = PL.generate(pid, iid, options=dict(size='native', quality=90))
    img = read(r['file']); assert not r['existing'] and r['output'] == [800, 600] and img.shape[:2] == (600, 800) and r['format'] == 'jpeg' and r['quality'] == 90 and r['bytes'] > 10_000
    assert r['level']['applied'] is False and r['summary'].startswith('JPEG q90')                                                  # a straight-down shot is not levelled
    side = json.load(open(os.path.splitext(r['file'])[0] + '.json')); assert side['key'] == r['key'] and side['source'].endswith('0001_D.JPG') and side['input']['kind'] == 'rendered'
    again = PL.generate(pid, iid, options=dict(size='native', quality=90)); assert again['existing'] and again['key'] == r['key']
    other = PL.generate(pid, iid, options=dict(size='native', quality=80)); assert other['key'] != r['key'] and len([v for v in P.get_item(pid, iid)['variants'] if v['kind'] == 'generated']) == 2


def test_an_oblique_shot_is_levelled_when_the_lines_agree(project):
    pid, sid = project; r = PL.generate(pid, P.item_id(sid, OBLIQUE), options=dict(size='native', sharpen=False))
    assert 'level' in r and (r['level']['applied'] is True and abs(r['level']['angle']) > 1 or r['level']['reason'])                  # levelled, or the reason it was not


def test_a_linear_dng_panorama_is_developed_and_matched_to_its_frames(project):
    pid, sid = project; r = PL.generate(pid, P.item_id(sid, PANO), options=dict(size='native', sharpen=False, level=False))
    assert r['steps'][0] == 'developed from the raw' and r['input'] == dict(kind='linear-dng', make='dji', model='fc8482', profile='DJI Mini 4 Pro') and r['native'] == [320, 240] and r['output'] == [320, 240]
    assert 'match' in r and (r['match']['applied'] or 'reason' in r['match']) and read(r['file']).shape[:2] == (240, 320)
    off = PL.generate(pid, P.item_id(sid, PANO), options=dict(size='native', sharpen=False, level=False, match='off')); assert off['key'] != r['key'] and 'match' not in off


def test_rules_apply_a_preset_to_every_picture_of_a_source_and_only_once(project):
    pid, sid = project; PL.set_rules(pid, [dict(source=sid, pipeline='web')]); assert len(PL.pending(pid)) == 3                         # two shots and the panorama (the video is left out)
    r = PL.apply_rules(pid, workers=2); assert r['done'] == 3 and r['failed'] == [] and r['total'] == 3 and PL.pending(pid) == []
    assert PL.apply_rules(pid)['done'] == 0
    for stem in (NADIR, OBLIQUE, PANO): assert len([v for v in P.get_item(pid, P.item_id(sid, stem))['variants'] if v['kind'] == 'generated']) == 1


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_the_tiled_development_gives_the_whole_picture_result(shoot, monkeypatch, backend):
    """The same linear DNG developed whole and in tiles (forced by the tuning variables) on the CPU code, and on the GPU code run by torch on the CPU."""
    path = os.path.join(shoot, 'DJI_20250402190111_0001_D-Pano.dng')
    if backend == 'gpu':
        pytest.importorskip('torch'); gpu.use('cpu'); monkeypatch.setattr(gpu, 'available', lambda: True)
    try:
        whole = D.develop(path, backend=backend, highlight_contrast=0.8)
        monkeypatch.setenv('STRATASTILLS_TILE', '128'); monkeypatch.setenv('STRATASTILLS_MAX_PIXELS', '1'); tiled = D.develop(path, backend=backend, highlight_contrast=0.8)
    finally: gpu.use(None)
    assert tiled.shape == whole.shape == (240, 320, 3) and tiled.dtype == np.float32 and np.isfinite(tiled).all()
    d = np.abs(tiled - whole); assert d.mean() < 2e-3 and np.percentile(d, 99.9) < 0.06, (d.mean(), np.percentile(d, 99.9))
