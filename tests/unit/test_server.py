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


def test_the_visual_report_is_served_from_its_folder(client, tmp_path, monkeypatch):
    rep = tmp_path / 'rep'; (rep / 'assets').mkdir(parents=True); (rep / 'index.html').write_text('<h1>report</h1>'); (rep / 'assets' / 'a.jpg').write_bytes(b'jpg'); (tmp_path / 'secret.txt').write_text('no')
    monkeypatch.setenv('STRATASTILLS_REPORT', str(tmp_path / 'none')); assert client.get('/api/report').json() == dict(available=False, built=None, neutral=False, auto=False) and client.get('/report/').status_code == 404
    (rep / 'auto').mkdir(); (rep / 'auto' / 'index.html').write_text('a')
    monkeypatch.setenv('STRATASTILLS_REPORT', str(rep)); j = client.get('/api/report').json(); assert j['available'] is True and j['auto'] is True and j['neutral'] is False
    assert client.get('/report/').text == '<h1>report</h1>' and client.get('/report/assets/a.jpg').content == b'jpg' and client.get('/report', follow_redirects=False).status_code in (302, 307)
    assert client.get('/report/../secret.txt').status_code == 404 and client.get('/report/%2e%2e/secret.txt').status_code == 404


def test_report_flags_are_kept_per_report_and_picture(client, tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_REPORT_FLAGS', str(tmp_path / 'flags'))
    assert client.get('/api/report-flags/auto').json() == {} and client.get('/api/report-flags/other').status_code == 404
    r = client.put('/api/report-flags/auto/2023-12-16_DSC05595', json=dict(rating='bad', reasons=['too red / warm'], note='skin ok, room red', status='deferred')); assert r.status_code == 200 and r.json()['status'] == 'deferred' and r.json()['rating'] == 'bad' and r.json()['first'] and r.json()['at']
    assert client.get('/api/report-flags/auto').json()['2023-12-16_DSC05595']['reasons'] == ['too red / warm'] and client.get('/api/report-flags/neutral').json() == {}
    assert client.put('/api/report-flags/auto/..%2Fx', json=dict(flag=True)).status_code in (400, 404, 405)
    assert client.put('/api/report-flags/auto/2023-12-16_DSC05595', json=dict(rating='nonsense')).json() == {}; assert client.get('/api/report-flags/auto').json() == {}


def test_the_arena_offers_blind_pairs_records_votes_and_rates(client, tmp_path, monkeypatch):
    arena = tmp_path / 'arena'; (arena / 'img').mkdir(parents=True); monkeypatch.setenv('STRATASTILLS_ARENA', str(arena))
    assert client.get('/api/arena/next').status_code == 404 and client.get('/arena/').status_code == 200
    for n in ('lr', 'auto1'): (arena / 'img' / f'{n}.jpg').write_bytes(b'jpg-' + n.encode())
    (arena / 'pool.json').write_text('{"p1": {"lr": {"file": "img/lr.jpg", "hash": "a"}, "auto1": {"file": "img/auto1.jpg", "hash": "b"}}}')
    p = client.get('/api/arena/next').json(); assert 'lr' not in str(p) and 'auto1' not in str(p)
    got = {client.get('/arena/' + p['left']).content, client.get('/arena/' + p['right']).content}; assert got == {b'jpg-lr', b'jpg-auto1'}
    assert client.post('/api/arena/vote', json=dict(token=p['token'], choice='left', ms=900)).json()['votes'] == 1 and client.post('/api/arena/vote', json=dict(token=p['token'], choice='left')).status_code == 404
    q = client.get('/api/arena/next').json(); assert client.post('/api/arena/vote', json=dict(token=q['token'], choice='sideways')).status_code == 400
    r = client.get('/api/arena/ratings').json(); assert r['votes'] == 1 and {v['name'] for v in r['variants']} == {'lr', 'auto1'} and r['head_to_head'][0]['ties'] == 0
    assert client.post('/api/arena/undo').json()['votes'] == 0 and client.post('/api/arena/undo').status_code == 404
