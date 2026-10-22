import os

import cv2, numpy as np, pytest
from PIL import Image
from stratastills import pipeline as PL, projects as P
from test_detail import texture


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects')); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'home'))
    src = tmp_path / 'src'; src.mkdir(); y = texture(1600, 1200)
    for x in range(0, 1600, 100): cv2.line(y, (x, 0), (x, 1200), 0.05, 4)
    for r in range(0, 1200, 100): cv2.line(y, (0, r), (1600, r), 0.05, 4)
    y = cv2.GaussianBlur(y, (0, 0), 1.6); img = (np.dstack([y, y * 0.9, y * 0.8]) * 255).astype(np.uint8); img = cv2.warpAffine(img, cv2.getRotationMatrix2D((800, 600), 4, 1.05), (1600, 1200), borderMode=cv2.BORDER_REFLECT)
    Image.fromarray(img).save(src / 'S_0001.JPG', exif=Image.Exif().tobytes()); Image.fromarray(img[:600, :800].copy()).save(src / 'S_0002.JPG'); (src / 'S_0003.MP4').write_bytes(b'x')
    d = P.create('E', [str(src)]); sid = P.source_summaries(d['id'])[0]['id']; return d['id'], sid, P.item_id(sid, 'S_0001')


def generated(pid, iid): return [v for v in P.get_item(pid, iid)['variants'] if v['kind'] == 'generated']


def test_presets_are_data_and_can_be_saved_and_deleted():
    names = [p['name'] for p in PL.list_presets()]; assert names[:3] == ['best', 'archive', 'web'] and PL.get_preset('best')['builtin']
    p = PL.save_preset('mine', 'My crisp web', PL.apply_options(PL.get_preset('web')['stages'], dict(sharpen_amount=2.0)))
    assert not p['builtin'] and p['label'] == 'My crisp web' and 'mine' in [q['name'] for q in PL.list_presets()]
    with pytest.raises(P.ProjectError): PL.save_preset('best', 'x', PL.BEST)
    with pytest.raises(P.ProjectError): PL.save_preset('Bad Name', 'x', PL.BEST)
    PL.delete_preset('mine'); assert 'mine' not in [q['name'] for q in PL.list_presets()]
    for bad in ('best', 'nope'):
        with pytest.raises(P.ProjectError): PL.delete_preset(bad)
    with pytest.raises(P.ProjectError): PL.get_preset('nope')


def test_normalize_orders_and_checks_the_stages():
    n = PL.normalize([dict(stage='develop'), dict(stage='encode')]); assert n[0]['highlight_contrast'] == 0.8 and n[1] == dict(stage='encode', format='jpeg', quality=90)
    for bad in ([], [dict(stage='encode')], [dict(stage='develop')], [dict(stage='develop'), dict(stage='sharpen'), dict(stage='level'), dict(stage='encode')], [dict(stage='develop'), dict(stage='develop'), dict(stage='encode')],
                [dict(stage='nope')], [dict(stage='develop', bogus=1), dict(stage='encode')], [dict(stage='develop'), dict(stage='resize', size='10'), dict(stage='encode')], [dict(stage='develop'), dict(stage='encode', quality=10)],
                [dict(stage='develop'), dict(stage='sharpen', amount=9), dict(stage='encode')], [dict(stage='develop', highlight_contrast='x'), dict(stage='encode')]):
        with pytest.raises(P.ProjectError): PL.normalize(bad)
    assert PL.normalize([dict(stage='develop'), dict(stage='encode', format='tiff', quality=70)])[1] == dict(stage='encode', format='tiff', quality='auto')          # a TIFF has no quality


def test_flat_options_become_stage_settings():
    b = PL.get_preset('best')['stages']; o = PL.apply_options(b, dict(size='native', level=False, sharpen_amount=2, format='jpeg', quality=95, highlight_contrast=1.5)); d = {s['stage']: s for s in o}
    assert 'level' not in d and d['resize']['size'] == 'native' and d['sharpen']['amount'] == 2 and d['encode']['quality'] == 95 and d['develop']['highlight_contrast'] == 1.5
    assert 'sharpen' not in [s['stage'] for s in PL.apply_options(b, dict(sharpen=False))] and 'level' in [s['stage'] for s in PL.apply_options(PL.apply_options(b, dict(level=False)), dict(level=True))]
    assert 'sharpen' in [s['stage'] for s in PL.apply_options([s for s in b if s['stage'] != 'sharpen'], dict(sharpen=True))]
    with pytest.raises(P.ProjectError): PL.apply_options(b, dict(nope=1))
    assert PL.resolve(None, None, None) == PL.normalize(PL.BEST) and PL.resolve(stages=PL.BEST) == PL.normalize(PL.BEST)


def test_key_depends_on_what_is_made_and_summary_reads_well():
    b = PL.normalize(PL.BEST); k = PL.stage_key('a~x', 'raw', b)
    assert k == PL.stage_key('a~x', 'raw', b) and k != PL.stage_key('a~y', 'raw', b) and k != PL.stage_key('a~x', 'jpeg', b) and k != PL.stage_key('a~x', 'raw', PL.apply_options(b, dict(size='native')))
    assert PL.stage_key('a~x', 'raw', PL.apply_options(b, dict(format='tiff', quality=70))) == PL.stage_key('a~x', 'raw', PL.apply_options(b, dict(format='tiff', quality=80)))
    assert PL.summary(b) == 'JPEG q90 · size auto · levelled · sharpened 1.5' and PL.summary(PL.apply_options(b, dict(quality='auto'))).startswith('JPEG auto quality') and 'TIFF 16 bit' in PL.summary(PL.get_preset('archive')['stages']) and '2048 px' in PL.summary(PL.get_preset('web')['stages'])
    assert 'highlight contrast 1.5' in PL.summary(PL.apply_options(b, dict(highlight_contrast=1.5)))


def test_generate_makes_a_new_variant_once_and_reports_it_exists_the_second_time(proj):
    pid, sid, iid = proj; steps = []
    key, have = PL.find_existing(pid, iid); assert have is None
    r = PL.generate(pid, iid, progress=steps.append)
    assert not r['existing'] and r['output'][0] % 64 == 0 and os.path.exists(r['file']) and r['file'].endswith(f'{key}.jpg') and os.path.dirname(r['file']) == P.generated_dir(pid, iid) and steps[0].startswith('developing') and 'writing' in steps
    assert r['level']['applied'] and abs(r['level']['angle'] + 4) < 1.0 and r['quality'] == 90 and 'JPEG quality 90 as asked' in ' '.join(r['steps']) and r['summary'].startswith('JPEG q90')
    im = Image.open(r['file']); assert im.size == tuple(r['output']) and im.info.get('icc_profile') and im.getexif().get(0x0112) in (None, 1)
    g = generated(pid, iid); assert len(g) == 1 and g[0]['id'] == 'gen-' + key and g[0]['stages'] and g[0]['quality'] == r['quality'] and g[0]['label'].startswith('Generated')
    k2, have = PL.find_existing(pid, iid); assert k2 == key and have['id'] == 'gen-' + key
    again = PL.generate(pid, iid); assert again['existing'] and again['file'] == r['file'] and len(generated(pid, iid)) == 1
    other = PL.generate(pid, iid, options=dict(size='native', quality=90)); assert not other['existing'] and other['output'] == [1600, 1200] and other['quality'] == 90 and len(generated(pid, iid)) == 2


def test_automatic_quality_is_still_available(proj):
    pid, sid, iid = proj; r = PL.generate(pid, iid, options=dict(size='native', quality='auto', sharpen=False, level=False))
    assert 88 <= r['quality'] <= 99 and 'chosen automatically' in ' '.join(r['steps']) and r['quality_info']['target'] >= 0.45 and r['summary'].startswith('JPEG auto quality')


def test_a_generated_variant_can_be_primary_but_is_never_a_source(proj):
    pid, sid, iid = proj; r = PL.generate(pid, iid, options=dict(size='native', sharpen=False, level=False)); vid = 'gen-' + r['key']
    assert P.set_primary(pid, iid, vid)['primary'] == vid
    assert PL.source_variant(P.get_item(pid, iid))['id'] == 'jpeg'                                    # with a generated primary, the next generation still starts from a real source file
    with pytest.raises(P.ProjectError): PL.generate(pid, iid, variant=vid)
    with pytest.raises(P.ProjectError): PL.generate(pid, iid, variant='raw')


def test_tiff_archive_preset_and_web_preset(proj):
    import tifffile
    pid, sid, iid = proj; r = PL.generate(pid, iid, preset='archive'); a = tifffile.imread(r['file'])
    assert r['file'].endswith('.tif') and a.dtype == np.uint16 and a.shape == (1200, 1600, 3) and r['quality'] is None and not any('sharpened' in s for s in r['steps'])
    w = PL.generate(pid, iid, preset='web'); assert max(w['output']) == 1600                          # 2048 asked, never above native: nothing is invented


def test_flat_picture_is_made_at_native_size_and_unsharpened(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'p')); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'h')); s = tmp_path / 's'; s.mkdir(); Image.new('RGB', (900, 700), 'gray').save(s / 'F_1.JPG')
    d = P.create('F', [str(s)]); iid = P.item_id(P.source_summaries(d['id'])[0]['id'], 'F_1'); r = PL.generate(d['id'], iid)
    assert r['output'] == [900, 700] and not r['detail']['ok'] and 'could not be measured' in r['size']['why'] and any('not sharpened' in x for x in r['steps'])


def test_rules_decide_what_is_pending_and_apply_rules_makes_it(proj):
    pid, sid, iid = proj
    assert PL.rules(pid) == [] and PL.pending(pid) == []
    assert PL.set_rules(pid, [dict(source=sid, pipeline='web')]) == [dict(source=sid, pipeline='web')] and len(PL.pending(pid)) == 2                      # two stills; the video is left alone
    for bad in ([dict(source='zzz', pipeline='web')], [dict(source=sid, pipeline='nope')]):
        with pytest.raises(P.ProjectError): PL.set_rules(pid, bad)
    log = []; r = PL.apply_rules(pid, lambda i, n, t: log.append((i, n)))
    assert r['done'] == 2 and r['failed'] == [] and r['total'] == 2 and PL.pending(pid) == [] and log[-1] == (2, 2)
    assert PL.apply_rules(pid)['total'] == 0
    PL.set_rules(pid, [dict(source='*', pipeline='web'), dict(source=sid, pipeline='archive', variant='jpeg')]); assert len(PL.pending(pid)) == 2                  # the archive rule adds the TIFFs of both
    assert PL.apply_rules(pid, stop=lambda: True)['done'] == 0


def test_apply_rules_reports_a_failure_and_goes_on(proj, monkeypatch):
    pid, sid, iid = proj; PL.set_rules(pid, [dict(source=sid, pipeline='web')]); real = PL.generate

    def flaky(p, i, *a, **k):
        if i.endswith('S_0001'): raise RuntimeError('boom')
        return real(p, i, *a, **k)
    monkeypatch.setattr(PL, 'generate', flaky); r = PL.apply_rules(pid); assert r['done'] == 1 and r['failed'] == [(iid, 'boom')]


def test_apply_rules_makes_several_pictures_at_once_without_mixing_them_up(proj, tmp_path):
    import threading
    pid, sid, iid = proj; src = tmp_path / 'src'
    for k in range(3, 8): Image.fromarray((np.random.default_rng(k).random((120, 160, 3)) * 255).astype('uint8')).save(src / f'S_{k:04d}.JPG')
    PL.set_rules(pid, [dict(source=sid, pipeline='web')]); todo = PL.pending(pid); assert len(todo) == 7
    seen = []; lock = threading.Lock(); real = PL.generate; running = [0]; peak = [0]

    def spy(p, i, *a, **k):
        with lock: running[0] += 1; peak[0] = max(peak[0], running[0])
        try: return real(p, i, *a, **k)
        finally:
            with lock: running[0] -= 1; seen.append(i)
    import pytest as _p; mp = _p.MonkeyPatch(); mp.setattr(PL, 'generate', spy)
    try: r = PL.apply_rules(pid, workers=3)
    finally: mp.undo()
    assert r['failed'] == [], r['failed']
    assert r['done'] == 7 and r['total'] == 7 and r['workers'] == 3 and sorted(seen) == sorted(i for i, _, _ in todo) and peak[0] >= 2        # each once, and really overlapping
    assert PL.pending(pid) == [] and all(len(generated(pid, i)) == 1 for i, _, _ in todo)


def test_a_failure_among_parallel_pictures_is_isolated_and_stop_prevents_new_ones(proj, monkeypatch):
    pid, sid, iid = proj; PL.set_rules(pid, [dict(source=sid, pipeline='web')]); real = PL.generate

    def flaky(p, i, *a, **k):
        if i.endswith('S_0001'): raise RuntimeError('boom')
        return real(p, i, *a, **k)
    monkeypatch.setattr(PL, 'generate', flaky); r = PL.apply_rules(pid, workers=2); assert r['done'] == 1 and r['failed'] == [(iid, 'boom')]
    PL.set_rules(pid, [dict(source=sid, pipeline='archive')]); r = PL.apply_rules(pid, stop=lambda: True, workers=2); assert r['done'] == 0 and r['failed'] == []


def test_the_device_slot_limits_how_many_pictures_are_on_the_card(monkeypatch):
    pytest.importorskip('torch')
    import threading, time
    from stratastills import gpu
    monkeypatch.setenv('STRATASTILLS_GPU_SLOTS', '2'); monkeypatch.setattr(gpu, '_slots', None); assert gpu.slots() == 2
    inside = [0]; peak = [0]; lock = threading.Lock()

    def work():
        with gpu.slot():
            with lock: inside[0] += 1; peak[0] = max(peak[0], inside[0])
            time.sleep(0.05)
            with lock: inside[0] -= 1
    ts = [threading.Thread(target=work) for _ in range(6)]; [t.start() for t in ts]; [t.join() for t in ts]
    assert peak[0] == 2 and inside[0] == 0
    monkeypatch.setattr(gpu, '_slots', None)


def test_worker_count_follows_the_machine(monkeypatch):
    from stratastills import resources
    assert resources.auto_workers(True) == 2
    monkeypatch.setattr(resources, 'total_ram', lambda: 16e9); monkeypatch.setattr(resources.os, 'cpu_count', lambda: 16); assert resources.auto_workers(False) == 1                          # 16 GB holds one 7 GB picture at 80 %
    monkeypatch.setattr(resources, 'total_ram', lambda: 64e9); assert resources.auto_workers(False) == 4                                                                                          # 16 cores / 4
    monkeypatch.setattr(resources, 'total_ram', lambda: 0); assert resources.auto_workers(False) == 1 and resources.total_ram.__name__ == '<lambda>'


def test_output_dims_are_multiples_of_16_when_the_aspect_allows_and_never_exceed_native():
    from stratastills import sizeplan as S
    assert S.output_dims(8064, 6048, 6016) == (6016, 4512) and S.output_dims(8064, 6048, 6000) == (6016, 4512) or S.output_dims(8064, 6048, 6000)[0] % 16 == 0
    for w, h, ls in ((8064, 6048, 5000), (6048, 8064, 5000), (4032, 3024, 2048), (3000, 2000, 1700), (4000, 3001, 1999)):
        ow, oh = S.output_dims(w, h, ls); assert max(ow, oh) <= max(w, h) and (ow < oh) == (w < h) and abs(ow / oh - w / h) < 0.01 * w / h
    assert S.output_dims(8064, 6048, 8064) == (8064, 6048) and S.output_dims(8064, 6048, 9000) == (8064, 6048) and S.output_dims(8064, 6048, 2048) == (2048, 1536)
    ow, oh = S.output_dims(8064, 6048, 5000); assert ow % 16 == 0 and oh % 16 == 0


def test_auto_size_is_the_folders_median_so_every_picture_of_a_size_gets_the_same(proj):
    from stratastills import sizeplan as S
    pid, sid, iid = proj; item = P.get_item(pid, iid); native = tuple(images_dims(item)); calls = []
    def measure(path): calls.append(path); return dict(ok=True, scale30=0.70)
    a = S.folder_size(pid, item, native, measure, PL.source_variant); n = len(calls); b = S.folder_size(pid, item, native, lambda p: 1 / 0, PL.source_variant)                  # the second answer is from the cache
    assert a == b and a % 64 == 0 and a >= 0.7 * max(native) and a < 0.7 * max(native) + 64 and n >= 1
    assert S.folder_size(pid, item, (7, 7), measure, PL.source_variant) is None


def images_dims(item):
    from stratastills import images
    return images.dims(next(v for v in item['variants'] if v['kind'] != 'generated')['path'])


def test_auto_size_never_goes_below_half_the_native_size(proj):
    from stratastills import sizeplan as S
    pid, sid, iid = proj; item = P.get_item(pid, iid); native = tuple(images_dims(item))
    assert S.folder_size(pid, item, native, lambda path: dict(ok=True, scale30=0.2), PL.source_variant) >= 0.5 * max(native)


def test_merged_panoramas_have_a_higher_floor_for_the_auto_size():
    from stratastills import sizeplan as S
    assert S.min_share('linear-dng') == 0.75 and S.min_share('raw') == 0.5 and S.min_share(None) == 0.5
