import pytest
from fastapi.testclient import TestClient
from stratastills.server.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects')); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'home')); return TestClient(create_app())


def test_landing_page_and_empty_list(client):
    assert 'strata-stills' in client.get('/').text and client.get('/api/projects').json() == []


def test_project_lifecycle(client, tmp_path):
    src = tmp_path / 's'; src.mkdir(); (src / 'DJI_1_0001_D.JPG').write_bytes(b'x')
    p = client.post('/api/projects', json=dict(name='Demo', sources=[str(src)])).json(); assert p['id'] == 'demo'
    assert client.post('/api/projects', json=dict(name='Demo')).status_code == 400
    assert [s['stem'] for s in client.get('/api/project/shots', params=dict(id='demo')).json()] == ['DJI_1_0001_D']
    assert client.post('/api/project/sources', params=dict(id='demo'), json=dict(path=str(tmp_path / 'missing'))).status_code == 400
    assert client.delete('/api/project/sources', params=dict(id='demo', path=str(src))).json()['sources'] == []
    assert client.get('/api/project', params=dict(id='nope')).status_code == 400


def test_sources_items_primary_and_image(client, tmp_path):
    from PIL import Image
    src = tmp_path / 's'; src.mkdir(); Image.new('RGB', (80, 60), 'gray').save(src / 'X_1.JPG'); Image.new('RGB', (80, 60)).save(src / 'X_1-Enhanced.jpg')
    client.post('/api/projects', json=dict(name='Demo', sources=[str(src)]))
    (s,) = client.get('/api/project/sources', params=dict(id='demo')).json(); assert s['stills'] == 1
    (it,) = client.get('/api/project/items', params=dict(id='demo', source=s['id'])).json(); assert it['primary'] == 'jpeg'
    assert client.put('/api/project/primary', params=dict(id='demo', item=it['id']), json=dict(variant='enhanced.jpg')).json()['primary'] == 'enhanced.jpg'
    assert client.put('/api/project/primary', params=dict(id='demo', item=it['id']), json=dict(variant='zzz')).status_code == 400
    r = client.get('/api/project/image', params=dict(id='demo', item=it['id'], transform='level', px=64)); assert r.status_code == 200 and r.headers['content-type'] == 'image/jpeg'
    assert client.get('/api/project/transform', params=dict(id='demo', item=it['id'], transform='level')).json()['applied'] is False
    c = client.get('/api/project/crop', params=dict(id='demo', item=it['id'], x=0.3, y=0.6, size=32, zoom=3)); assert c.status_code == 200 and c.headers['content-type'] == 'image/jpeg'
    assert client.get('/api/project/crop', params=dict(id='demo', item='nope~x')).status_code == 400
    assert client.get('/api/project/image', params=dict(id='demo', item='nope~x')).status_code == 400
    assert client.get('/api/project/item', params=dict(id='demo', item=it['id'])).json()['stem'] == 'X_1'


def test_settings_roundtrip(client, tmp_path):
    assert client.put('/api/settings', json=dict(projects_dir='relative')).status_code == 400
    assert client.get('/api/settings').json()['projects_dir'] == str(tmp_path / 'projects')


def test_detail_export_job_and_download(client, tmp_path):
    import time
    from PIL import Image
    import numpy as np
    src = tmp_path / 's'; src.mkdir(); rng = np.random.default_rng(0); Image.fromarray((rng.random((600, 800, 3)) * 255).astype('uint8')).save(src / 'Y_1.JPG')
    client.post('/api/projects', json=dict(name='Demo', sources=[str(src)])); (s,) = client.get('/api/project/sources', params=dict(id='demo')).json()
    iid = client.get('/api/project/items', params=dict(id='demo', source=s['id'])).json()[0]['id']
    assert client.get('/api/project/export', params=dict(id='demo', item=iid)).json()['state'] == 'idle'
    assert 'ok' in client.get('/api/project/detail', params=dict(id='demo', item=iid)).json()
    assert client.get('/api/project/detail', params=dict(id='demo', item=iid, variant='raw')).status_code == 400
    assert client.post('/api/project/export', params=dict(id='demo', item=iid), json=dict(format='bmp')).status_code == 400
    assert client.post('/api/project/export', params=dict(id='demo', item=iid), json=dict(size='native', format='jpeg', level=False, sharpen=False)).json()['state'] in ('running', 'done')
    for _ in range(100):
        j = client.get('/api/project/export', params=dict(id='demo', item=iid)).json()
        if j['state'] != 'running': break
        time.sleep(0.1)
    assert j['state'] == 'done', j
    (e,) = client.get('/api/project/exports', params=dict(id='demo', stem='Y_1')).json(); assert e['output'] == [800, 600]
    f = client.get('/api/project/export/file', params=dict(id='demo', name=e['name'])); assert f.status_code == 200 and f.content[:2] == b'\xff\xd8'
    assert client.get('/api/project/export/file', params=dict(id='demo', name='../x')).status_code == 404
