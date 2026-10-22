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
    import time
    cq = dict(id='demo', item=it['id'], x=0.3, y=0.6, size=32, zoom=3); c = client.get('/api/project/crop', params=cq)
    assert c.status_code in (200, 202) and (c.status_code == 200 or c.json()['state'] in ('running', 'ready'))                 # 202 while the full-size copy is being made
    for _ in range(100):
        if client.get('/api/project/fullres', params=dict(id='demo', item=it['id'])).json()['state'] == 'ready': break
        time.sleep(0.05)
    c = client.get('/api/project/crop', params=cq); assert c.status_code == 200 and c.headers['content-type'] == 'image/jpeg'
    assert client.get('/api/project/crop', params=dict(id='demo', item='nope~x')).status_code == 400
    assert client.get('/api/project/image', params=dict(id='demo', item='nope~x')).status_code == 400
    assert client.get('/api/project/item', params=dict(id='demo', item=it['id'])).json()['stem'] == 'X_1'


def test_settings_roundtrip(client, tmp_path):
    assert client.put('/api/settings', json=dict(projects_dir='relative')).status_code == 400
    assert client.get('/api/settings').json()['projects_dir'] == str(tmp_path / 'projects')
