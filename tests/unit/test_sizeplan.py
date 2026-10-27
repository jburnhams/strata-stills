"""sizeplan.py: one output size for a folder of pictures (from a sample of them), and the sizes that keep a clean aspect ratio."""
import json, types

import numpy as np
import pytest

from stratastills import projects as P, sizeplan as S


def test_a_request_at_or_above_native_gives_the_native_size():
    assert S.output_dims(4000, 3000, 4000) == (4000, 3000) and S.output_dims(4000, 3000, 9000) == (4000, 3000)


@pytest.mark.parametrize('w,h,ask,m', [(4000, 3000, 2000, 16), (3000, 4000, 2000, 16), (6000, 4000, 3000, 16), (1001, 777, 500, 2)])
def test_the_size_keeps_the_aspect_ratio_with_both_sides_on_the_widest_grid_it_allows(w, h, ask, m):
    ow, oh = S.output_dims(w, h, ask)
    assert max(ow, oh) <= max(w, h) and abs(max(ow, oh) - ask) <= 64 and ow % m == 0 and oh % m == 0
    assert abs(ow / oh - w / h) / (w / h) < 0.02                                                         # portrait stays portrait, wide stays wide
    assert (ow > oh) == (w > h)


def test_a_merged_panorama_keeps_more_of_its_detail_than_a_single_frame():
    assert S.min_share('linear-dng') == S.MIN_SHARE_MERGED > S.min_share('raw') == S.MIN_SHARE


def test_the_sample_of_a_big_folder_is_spread_evenly_including_both_ends():
    ids = list(range(100)); s = S._sample(ids)
    assert len(s) == S.SAMPLE and s[0] == 0 and s[-1] == 99 and s == sorted(s) and len(set(s)) == S.SAMPLE
    assert S._sample([3, 1]) == [3, 1]                                                                   # a small folder is all of it, in order


@pytest.fixture
def folder(monkeypatch, tmp_path):
    items = [dict(id='s1~a', kind='photo'), dict(id='s1~b', kind='photo'), dict(id='s1~v', kind='video'), dict(id='s1~bad', kind='photo'), dict(id='s1~small', kind='photo')]
    def source_items(pid, sid): return items
    monkeypatch.setattr(S, 'P', types.SimpleNamespace(source_items=source_items, folder=lambda pid: str(tmp_path), ProjectError=P.ProjectError))
    dims = {'/p/a': (4000, 3000), '/p/b': (4000, 3000), '/p/small': (1000, 750)}
    monkeypatch.setattr(S.images, 'dims', lambda path: dims[path])
    def pick(it):
        if it['id'] == 's1~bad': raise P.ProjectError('no variant')
        return dict(path='/p/' + it['id'].partition('~')[2])
    calls = []
    def measure(path): calls.append(path); return {'ok': True, S.detail.RECOMMEND: 0.6}
    return types.SimpleNamespace(item=items[0], pick=pick, measure=measure, calls=calls, items=items)


def test_the_folder_size_is_the_median_detail_rounded_up_to_64_and_measured_once(folder):
    size = S.folder_size('p', folder.item, (4000, 3000), folder.measure, folder.pick)
    assert size == 2432                                                                                  # ceil(0.6 * 4000 / 64) * 64
    assert sorted(folder.calls) == ['/p/a', '/p/b']                                                      # the video, the unreadable one and the other native size are not sampled
    assert S.folder_size('p', folder.item, (4000, 3000), folder.measure, folder.pick) == 2432 and len(folder.calls) == 2         # the second time it comes from the cache


def test_the_folder_size_never_goes_below_the_minimum_share(folder):
    folder.measure = lambda path: {'ok': True, S.detail.RECOMMEND: 0.1}
    assert S.folder_size('p', folder.item, (4000, 3000), folder.measure, folder.pick) == 2048            # half of 4000, rounded up to a multiple of 64


def test_the_folder_size_never_goes_above_native(folder):
    folder.measure = lambda path: {'ok': True, S.detail.RECOMMEND: 5.0}
    assert S.folder_size('p', folder.item, (4000, 3000), folder.measure, folder.pick) == 4000


def test_nothing_is_measured_for_a_picture_that_is_not_in_the_group_or_has_no_usable_measurement(folder):
    assert S.folder_size('p', dict(id='s1~small', kind='photo'), (4000, 3000), folder.measure, folder.pick) is None              # its own size is another group
    folder.measure = lambda path: {'ok': False}; assert S.folder_size('p', folder.item, (4000, 3000), folder.measure, folder.pick) is None


def test_a_damaged_cache_file_is_measured_again(folder, tmp_path):
    S.folder_size('p', folder.item, (4000, 3000), folder.measure, folder.pick)
    for f in (tmp_path / 'cache').glob('*.json'): f.write_text('{broken')
    assert S.folder_size('p', folder.item, (4000, 3000), folder.measure, folder.pick) == 2432 and len(folder.calls) == 4
