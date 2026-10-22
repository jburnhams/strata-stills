import json, os

import cv2, numpy as np, pytest
from PIL import Image
from stratastills import export as X, projects as P
from test_detail import texture


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects')); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'home'))
    src = tmp_path / 'src'; src.mkdir()
    y = texture(1600, 1200)
    for x in range(0, 1600, 100): cv2.line(y, (x, 0), (x, 1200), 0.05, 4)
    for r in range(0, 1200, 100): cv2.line(y, (0, r), (1600, r), 0.05, 4)
    y = cv2.GaussianBlur(y, (0, 0), 1.6); img = (np.dstack([y, y * 0.9, y * 0.8]) * 255).astype(np.uint8)
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((800, 600), 4, 1.05), (1600, 1200), borderMode=cv2.BORDER_REFLECT)
    Image.fromarray(img).save(src / 'S_0001.JPG', exif=Image.Exif().tobytes())
    d = P.create('E', [str(src)]); sid = P.source_summaries(d['id'])[0]['id']; return d['id'], P.item_id(sid, 'S_0001')


def test_choose_size():
    res = dict(recommended=4096)
    assert X.choose_size('auto', 8064, res)[0] == 4096 and X.choose_size('native', 8064, res)[0] == 8064 and X.choose_size('3000', 8064, res)[0] == 3000
    n, why = X.choose_size('9000', 8064, res); assert n == 8064 and 'limited' in why
    assert X.choose_size('auto', 8064, {})[0] == 8064 and 'could not be measured' in X.choose_size('auto', 8064, {})[1]
    with pytest.raises(P.ProjectError): X.choose_size('10', 8064, res)


def test_export_jpeg_at_the_measured_size_with_a_record(proj):
    pid, iid = proj; steps = []
    r = X.export_item(pid, iid, None, 'auto', 'jpeg', progress=steps.append)
    assert r['detail']['ok'] and r['output'][0] < 1600 and r['output'][0] % 64 == 0 and os.path.exists(r['file']) and r['file'].endswith('.jpg')
    assert steps[0].startswith('developing') and 'writing' in steps
    assert r['level']['applied'] and abs(r['level']['angle'] + 4) < 1.0                       # the picture was turned 4 degrees anticlockwise: the fix is the other way
    im = Image.open(r['file']); assert im.size == tuple(r['output']) and im.info.get('icc_profile') and im.getexif().get(0x0112) in (None, 1)
    side = json.load(open(os.path.splitext(r['file'])[0] + '.json')); assert side['size']['why'] and side['steps']
    assert [e['name'] for e in X.list_exports(pid, 'S_0001')] == [os.path.basename(r['file'])] and X.list_exports(pid, 'other') == []


def test_export_native_tiff_without_level_or_sharpen(proj):
    import tifffile
    pid, iid = proj; r = X.export_item(pid, iid, 'jpeg', 'native', 'tiff', level=False, sharpen=False)
    assert r['output'] == [1600, 1200] and 'level' not in r and not any('sharpened' in s for s in r['steps'])
    a = tifffile.imread(r['file']); assert a.dtype == np.uint16 and a.shape == (1200, 1600, 3)


def test_sharpening_changes_the_picture(proj):
    pid, iid = proj
    a = X.export_item(pid, iid, None, 'native', 'jpeg', level=False, sharpen=False); ia = np.asarray(Image.open(a['file'])).astype(float)
    b = X.export_item(pid, iid, None, 'native', 'jpeg', level=False, sharpen=True); ib = np.asarray(Image.open(b['file'])).astype(float)
    assert os.path.basename(a['file']) == os.path.basename(b['file']) and np.abs(ia - ib).mean() > 0.2                        # same name: the later export replaces the earlier one


def test_bad_requests(proj):
    pid, iid = proj
    with pytest.raises(P.ProjectError): X.export_item(pid, iid, 'raw')
    with pytest.raises(P.ProjectError): X.export_item(pid, iid, None, 'auto', 'bmp')
    with pytest.raises(P.ProjectError): X.export_item(pid, 'zz~nope')


def test_flat_picture_exports_at_native_size(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'p')); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'h')); s = tmp_path / 's'; s.mkdir(); Image.new('RGB', (900, 700), 'gray').save(s / 'F_1.JPG')
    d = P.create('F', [str(s)]); iid = P.item_id(P.source_summaries(d['id'])[0]['id'], 'F_1'); r = X.export_item(d['id'], iid)
    assert r['output'] == [900, 700] and not r['detail']['ok'] and 'could not be measured' in r['size']['why']
