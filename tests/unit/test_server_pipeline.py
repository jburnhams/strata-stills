import time

import numpy as np, pytest
from fastapi.testclient import TestClient
from PIL import Image
from stratastills.server.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects')); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'home')); return TestClient(create_app())


def wait(client, url, **params):
    for _ in range(300):
        j = client.get(url, params=params).json()
        if j['state'] != 'running': return j
        time.sleep(0.1)
    raise AssertionError('still running')


def test_pipelines_generate_check_rules_and_download(client, tmp_path):
    src = tmp_path / 's'; src.mkdir(); rng = np.random.default_rng(0); Image.fromarray((rng.random((600, 800, 3)) * 255).astype('uint8')).save(src / 'Y_1.JPG')
    client.post('/api/projects', json=dict(name='Demo', sources=[str(src)])); (s_,) = client.get('/api/project/sources', params=dict(id='demo')).json()
    iid = client.get('/api/project/items', params=dict(id='demo', source=s_['id'])).json()[0]['id']; q = dict(id='demo', item=iid)

    names = [p['name'] for p in client.get('/api/pipelines').json()]; assert names[:3] == ['best', 'archive', 'web']
    assert client.put('/api/pipeline', params=dict(name='mine'), json=dict(label='Mine', stages=[dict(stage='develop'), dict(stage='encode', quality=92)])).json()['label'] == 'Mine'
    assert client.put('/api/pipeline', params=dict(name='best'), json=dict(stages=[dict(stage='develop'), dict(stage='encode')])).status_code == 400
    assert [p['name'] for p in client.delete('/api/pipeline', params=dict(name='mine')).json()] == names

    assert client.get('/api/project/generate', params=q).json()['state'] == 'idle'
    assert 'ok' in client.get('/api/project/detail', params=q).json() and client.get('/api/project/detail', params=dict(q, variant='raw')).status_code == 400
    body = dict(options=dict(size='native', level=False, sharpen=False, quality=90)); c = client.post('/api/project/generate/check', params=q, json=body).json()
    assert c['exists'] is False and c['summary'].startswith('JPEG q90') and c['stages'][0]['stage'] == 'develop'
    assert client.post('/api/project/generate/check', params=q, json=dict(options=dict(format='bmp'))).status_code == 400
    assert client.post('/api/project/generate', params=q, json=body).json()['state'] in ('running', 'done')
    j = wait(client, '/api/project/generate', **q); assert j['state'] == 'done', j

    c2 = client.post('/api/project/generate/check', params=q, json=body).json()
    assert c2['exists'] and c2['variant'] == 'gen-' + c['key'] and client.post('/api/project/generate', params=q, json=body).json() == dict(state='exists', variant='gen-' + c['key'])
    item = client.get('/api/project/item', params=q).json(); gen = [v for v in item['variants'] if v['kind'] == 'generated']; assert len(gen) == 1 and gen[0]['output'] == [800, 600]
    f = client.get('/api/project/generated/file', params=dict(q, variant=gen[0]['id'])); assert f.status_code == 200 and f.content[:2] == b'\xff\xd8'
    assert client.get('/api/project/generated/file', params=dict(q, variant='jpeg')).status_code == 404

    assert client.get('/api/project/rules', params=dict(id='demo')).json() == dict(rules=[], pending=0)
    assert client.put('/api/project/rules', params=dict(id='demo'), json=dict(rules=[dict(source=s_['id'], pipeline='web')])).json()['pending'] == 1
    assert client.put('/api/project/rules', params=dict(id='demo'), json=dict(rules=[dict(source=s_['id'], pipeline='nope')])).status_code == 400
    assert client.get('/api/project/apply', params=dict(id='demo')).json()['state'] == 'idle' and client.post('/api/project/apply', params=dict(id='demo')).json()['state'] in ('running', 'done')
    j = wait(client, '/api/project/apply', id='demo')
    assert j['state'] == 'done' and j['done'] == 1 and j['failed'] == [] and client.get('/api/project/rules', params=dict(id='demo')).json()['pending'] == 0
    assert client.delete('/api/project/apply', params=dict(id='demo')).json()['state'] == 'done'
