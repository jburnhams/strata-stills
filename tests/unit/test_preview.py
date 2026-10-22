import cv2, numpy as np, pytest
from PIL import Image
from stratastills import fullres, images, preview, projects as P


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects')); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'home'))
    src = tmp_path / 'src'; src.mkdir()
    img = np.full((300, 400, 3), 200, np.uint8)
    for x in range(0, 400, 40): cv2.line(img, (x, 0), (x, 300), (30, 30, 30), 2)
    for y in range(0, 300, 40): cv2.line(img, (0, y), (400, y), (30, 30, 30), 2)
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((200, 150), 5, 1.0), (400, 300), borderValue=(200, 200, 200))
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(src / 'A_0001.JPG'); Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(src / 'A_0001-Enhanced.jpg')
    (src / 'A_0002.MP4').write_bytes(b'x')
    d = P.create('T', [str(src)]); sid = P.source_summaries(d['id'])[0]['id']; return d['id'], sid


def test_source_summary_and_items(proj):
    pid, sid = proj
    (s,) = P.source_summaries(pid); assert (s['stills'], s['panos'], s['videos'], s['reachable']) == (1, 0, 1, True)
    items = {i['stem']: i for i in P.source_items(pid, sid)}
    assert items['A_0001']['primary'] == 'jpeg' and [v['id'] for v in items['A_0001']['variants']] == ['jpeg', 'enhanced.jpg'] and items['A_0002']['kind'] == 'video'


def test_set_primary_is_remembered_and_validated(proj):
    pid, sid = proj; iid = P.item_id(sid, 'A_0001')
    assert P.set_primary(pid, iid, 'enhanced.jpg')['primary'] == 'enhanced.jpg' and not P.get_item(pid, P.item_id(sid, 'A_0002'))['chosen']
    assert P.get_item(pid, iid)['chosen'] and P.get_item(pid, iid)['primary'] == 'enhanced.jpg'
    with pytest.raises(P.ProjectError): P.set_primary(pid, iid, 'raw')
    with pytest.raises(P.ProjectError): P.get_item(pid, P.item_id(sid, 'nope'))


def test_render_original_and_level_are_cached(proj):
    pid, sid = proj; iid = P.item_id(sid, 'A_0001')
    p, info = preview.render(pid, iid, None, 'original', 200); im = cv2.imread(p); assert max(im.shape[:2]) == 200 and not info['applied']
    p2, info2 = preview.render(pid, iid, 'jpeg', 'level', 400)
    assert info2['reason'] is None or info2['applied'] is False
    p3, info3 = preview.render(pid, iid, 'jpeg', 'level', 400); assert p3 == p2 and info3 == info2
    with pytest.raises(P.ProjectError): preview.render(pid, iid, 'raw', 'original', 100)
    with pytest.raises(P.ProjectError): preview.render(pid, iid, None, 'nope', 100)


def test_cache_key_changes_with_the_file(tmp_path):
    f = tmp_path / 'a.jpg'; Image.new('RGB', (8, 8)).save(f); k1 = images.cache_key(str(f), 'x'); f.write_bytes(f.read_bytes() + b'0'); assert images.cache_key(str(f), 'x') != k1


def test_unreadable_file_is_an_image_error(tmp_path):
    f = tmp_path / 'bad.jpg'; f.write_bytes(b'not a jpeg')
    with pytest.raises(images.ImageError): images.load(str(f))
    r = tmp_path / 'bad.dng'; r.write_bytes(b'not a raw')
    with pytest.raises(images.ImageError): images.load(str(r))


def wait_ready(pid, iid, variant, tries=100):
    import time
    for _ in range(tries):
        if preview.fullres_status(pid, iid, variant)['state'] == 'ready': return
        time.sleep(0.05)
    raise AssertionError('the full-size copy was not made')


def jpg(buf): return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def test_crop_waits_for_the_full_size_copy_then_cuts_it(proj):
    pid, sid = proj; iid = P.item_id(sid, 'A_0001')
    with pytest.raises(preview.NotReady) as e: preview.crop(pid, iid, 'jpeg', 'original', 0.5, 0.5, 100, 2)
    assert e.value.status['state'] in ('running', 'ready'); wait_ready(pid, iid, 'jpeg')
    c = jpg(preview.crop(pid, iid, 'jpeg', 'original', 0.5, 0.5, 100, 2)); assert c.shape[:2] == (200, 200)
    assert jpg(preview.crop(pid, iid, 'jpeg', 'original', -5, 9, 5000, 99)).shape == (preview.MAX_OUT, preview.MAX_OUT, 3)             # absurd requests are clamped
    with pytest.raises(P.ProjectError): preview.crop(pid, iid, 'raw', 'original')
    with pytest.raises(P.ProjectError): preview.crop(pid, iid, 'jpeg', 'nope')
    assert fullres.get(pid, P.get_item(pid, iid)['variants'][0]['path']).shape == (300, 400, 3) and fullres.status(pid, P.get_item(pid, iid)['variants'][0]['path'])['state'] == 'ready'


def test_variants_of_different_sizes_show_the_same_scene_area(proj):
    from stratastills import pipeline as PL
    pid, sid = proj; iid = P.item_id(sid, 'A_0001'); r = PL.generate(pid, iid, options=dict(size='200', sharpen=False, level=False)); gid = 'gen-' + r['key']; assert r['output'][0] == 200
    wait_ready(pid, iid, 'jpeg'); wait_ready(pid, iid, gid)
    a = jpg(preview.crop(pid, iid, 'jpeg', 'original', 0.5, 0.5, 100, 2)); b = jpg(preview.crop(pid, iid, gid, 'original', 0.5, 0.5, 100, 2))                   # 100 px of the 400 px original, 50 px of the 200 px export: the same area
    assert a.shape == b.shape == (200, 200, 3) and np.abs(a.astype(float) - b.astype(float)).mean() < 25 and np.abs(a.astype(float) - jpg(preview.crop(pid, iid, 'jpeg', 'original', 0.9, 0.2, 100, 2)).astype(float)).mean() > 5


def test_levelled_crop_turns_only_the_crop(proj):
    pid, sid = proj; iid = P.item_id(sid, 'A_0001'); wait_ready(pid, iid, 'jpeg')
    info = preview.render(pid, iid, 'jpeg', 'level', preview.WORK_PX)[1]
    a = jpg(preview.crop(pid, iid, 'jpeg', 'original', 0.5, 0.5, 120, 1)); b = jpg(preview.crop(pid, iid, 'jpeg', 'level', 0.5, 0.5, 120, 1))
    assert a.shape == b.shape == (120, 120, 3) and (not info.get('applied') or np.abs(a.astype(float) - b.astype(float)).mean() > 1)


def test_a_16_bit_tiff_is_readable(tmp_path):
    import tifffile
    p = tmp_path / 'a.tif'; a = np.zeros((40, 60, 3), np.uint16); a[..., 0] = 65535; tifffile.imwrite(p, a, photometric='rgb')
    im = images.load(str(p)); assert im.shape == (40, 60, 3) and im[0, 0].tolist() == [0, 0, 255]                          # BGR: red
    g = tmp_path / 'g.tif'; tifffile.imwrite(g, np.full((20, 30), 40000, np.uint16)); assert images.load(str(g)).shape == (20, 30, 3)
