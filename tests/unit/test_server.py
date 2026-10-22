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


def test_settings_roundtrip(client, tmp_path):
    assert client.put('/api/settings', json=dict(projects_dir='relative')).status_code == 400
    assert client.get('/api/settings').json()['projects_dir'] == str(tmp_path / 'projects')
