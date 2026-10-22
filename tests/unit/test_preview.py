import cv2, numpy as np, pytest
from PIL import Image
from stratastills import images, preview, projects as P


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


def test_crop_takes_the_same_point_of_each_variant_and_transform(proj):
    pid, sid = proj; iid = P.item_id(sid, 'A_0001')
    c = cv2.imdecode(np.frombuffer(preview.crop(pid, iid, 'jpeg', 'original', 0.5, 0.5, 100, 2), np.uint8), cv2.IMREAD_COLOR); assert c.shape[:2] == (200, 200)
    assert cv2.imdecode(np.frombuffer(preview.crop(pid, iid, 'enhanced.jpg', 'level', 0.1, 0.9, 60, 1), np.uint8), cv2.IMREAD_COLOR).shape[:2] == (60, 60)
    assert cv2.imdecode(np.frombuffer(preview.crop(pid, iid, None, 'original', -5, 9, 5000, 99), np.uint8), cv2.IMREAD_COLOR).shape[0] == 300 * 6          # clamped to the picture (300 px high) and 6x
    with pytest.raises(P.ProjectError): preview.crop(pid, iid, 'raw', 'original')
