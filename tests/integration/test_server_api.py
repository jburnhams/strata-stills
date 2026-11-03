"""The HTTP API over real project files: browse a project, make a variant, download it, apply a rule."""
import time


def wait(client, url, **params):
    import time
    deadline = time.time() + 10.0
    while time.time() < deadline:
        j = client.get(url, params=params).json()
        if j['state'] != 'running': return j
        time.sleep(0.05)
    raise AssertionError('still running')


def test_browse_a_project_from_the_projects_down_to_a_picture(client, project):
    pid, sid = project
    assert [p['id'] for p in client.get('/api/projects').json()] == [pid] and client.get('/api/project', params=dict(id=pid)).json()['name'] == 'Demo'
    (src,) = client.get('/api/project/sources', params=dict(id=pid)).json(); assert src['id'] == sid and src['reachable'] is True
    items = {i['stem']: i for i in client.get('/api/project/items', params=dict(id=pid, source=sid)).json()}
    assert sorted(items) == ['DJI_20250402190111_0001_D', 'DJI_20250402190111_0001_D-Pano', 'DJI_20250402190211_0002_D', 'DJI_20250402190511_0009_D'] and items['DJI_20250402190111_0001_D-Pano']['kind'] == 'pano'
    item = client.get('/api/project/item', params=dict(id=pid, item=items['DJI_20250402190111_0001_D']['id'])).json(); assert item['primary'] == 'jpeg' and [v['id'] for v in item['variants']] == ['jpeg']
    pano = client.get('/api/project/item', params=dict(id=pid, item=items['DJI_20250402190111_0001_D-Pano']['id'])).json(); assert pano['primary'] == 'raw'                      # a panorama is made from its linear DNG
    img = client.get('/api/project/image', params=dict(id=pid, item=item['id'], variant='jpeg')); assert img.status_code == 200 and img.content[:2] == b'\xff\xd8'


def test_a_variant_is_generated_downloaded_and_not_made_twice(client, project):
    pid, sid = project; (iid,) = [i['id'] for i in client.get('/api/project/items', params=dict(id=pid, source=sid)).json() if i['stem'].endswith('0002_D')]; q = dict(id=pid, item=iid); body = dict(options=dict(size='native', sharpen=False, quality=88))
    check = client.post('/api/project/generate/check', params=q, json=body).json(); assert check['exists'] is False and check['summary'].startswith('JPEG q88')
    assert client.post('/api/project/generate', params=q, json=body).json()['state'] in ('running', 'done'); j = wait(client, '/api/project/generate', **q); assert j['state'] == 'done', j
    assert client.post('/api/project/generate', params=q, json=body).json() == dict(state='exists', variant='gen-' + check['key'])
    (gen,) = [v for v in client.get('/api/project/item', params=q).json()['variants'] if v['kind'] == 'generated']; assert gen['output'] == [800, 600]
    f = client.get('/api/project/generated/file', params=dict(q, variant=gen['id'])); assert f.status_code == 200 and f.content[:2] == b'\xff\xd8' and len(f.content) == gen['bytes']
    crop = client.get('/api/project/crop', params=dict(q, variant=gen['id'], transform='original', x=0.5, y=0.5, size=100, scale=2)); assert crop.status_code in (200, 202)


def test_a_rule_makes_every_picture_of_the_source(client, project):
    pid, sid = project
    assert client.put('/api/project/rules', params=dict(id=pid), json=dict(rules=[dict(source=sid, pipeline='web')])).json()['pending'] == 3
    assert client.post('/api/project/apply', params=dict(id=pid)).json()['state'] in ('running', 'done'); j = wait(client, '/api/project/apply', id=pid)
    assert j['state'] == 'done' and j['done'] == 3 and j['failed'] == [] and client.get('/api/project/rules', params=dict(id=pid)).json()['pending'] == 0


def test_server_api_errors_and_settings(client, project):
    pid, sid = project
    # Test PUT /api/settings with relative path
    assert client.put('/api/settings', json={'projects_dir': 'relative'}).status_code == 400

    # Test GET /api/project/detail with non-existent variant
    items = client.get('/api/project/items', params=dict(id=pid, source=sid)).json()
    item_id = items[0]['id']
    assert client.get('/api/project/detail', params=dict(id=pid, item=item_id, variant='fake')).status_code == 400

    # Test GET /api/project/generated/file with missing variant
    assert client.get('/api/project/generated/file', params=dict(id=pid, item=item_id, variant='gen-fake')).status_code == 404

    # Test PUT /api/project/primary with missing variant
    assert client.put('/api/project/primary', params=dict(id=pid, item=item_id), json={'variant': 'fake'}).status_code == 400

def test_server_api_job_conflicts_and_cancel(client, project):
    pid, sid = project
    items = client.get('/api/project/items', params=dict(id=pid, source=sid)).json()
    iid = items[0]['id']
    q = dict(id=pid, item=iid)
    body = dict(options=dict(size='native', sharpen=False, quality=88))

    # Start generation
    j = client.post('/api/project/generate', params=q, json=body).json()
    assert j['state'] in ('running', 'done')

    # Try again while running, if not finished yet, will get 409
    try:
        if j['state'] == 'running':
            res = client.post('/api/project/generate', params=q, json=body)
            assert res.status_code == 409
    finally:
        wait(client, '/api/project/generate', **q)

    # Start apply rules
    client.put('/api/project/rules', params=dict(id=pid), json=dict(rules=[dict(source=sid, pipeline='web')]))
    j2 = client.post('/api/project/apply', params=dict(id=pid)).json()

    if j2['state'] == 'running':
        # Post apply rules while running -> 409
        res = client.post('/api/project/apply', params=dict(id=pid))
        assert res.status_code == 409

    # Stop apply
    stop_res = client.delete('/api/project/apply', params=dict(id=pid)).json()
    assert stop_res['stop'] is True or stop_res['state'] == 'idle'
    wait(client, '/api/project/apply', id=pid)

def test_server_api_pipelines(client):
    presets = client.get('/api/pipelines').json()
    initial_count = len(presets)

    # Create custom pipeline
    res = client.put('/api/pipeline', params=dict(name='my_custom'), json={'label': 'My Custom', 'stages': [{'stage': 'develop'}, {'stage': 'encode'}]})
    assert len(res.json()) == initial_count + 1

    # Delete custom pipeline
    res = client.delete('/api/pipeline', params=dict(name='my_custom'))
    assert len(res.json()) == initial_count

def test_server_api_index_and_reports(client, monkeypatch, tmp_path):
    # Test index route
    res = client.get('/')
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]

    # Redirect
    assert client.get('/report', follow_redirects=False).status_code == 307

    # No report exists
    report_dir = tmp_path / 'report'
    monkeypatch.setenv('STRATASTILLS_REPORT', str(report_dir))
    assert client.get('/api/report').json() == {'available': False, 'built': None}
    assert client.get('/report/').status_code == 404

    # Report exists
    report_dir.mkdir()
    index_file = report_dir / 'index.html'
    index_file.write_text('<html>report</html>')

    report_info = client.get('/api/report').json()
    assert report_info['available'] is True
    assert report_info['built'] is not None

    res = client.get('/report/')
    assert res.status_code == 200
    assert res.text == '<html>report</html>'

    # Security boundary: path traversal
    assert client.get('/report/%2E%2E/index.html').status_code == 404

def test_server_api_shots_and_sources(client, project, tmp_path, monkeypatch):
    pid, sid = project

    # Get shots
    shots = client.get('/api/project/shots', params=dict(id=pid)).json()
    assert len(shots) == 4
    assert all('source' in s for s in shots)

    # Test unreachable source handling in shots
    from stratastills import ingest
    monkeypatch.setattr(ingest, 'scan', lambda *args, **kw: (_ for _ in ()).throw(OSError("Offline")))
    offline_shots = client.get('/api/project/shots', params=dict(id=pid)).json()
    assert len(offline_shots) == 1
    assert offline_shots[0]['unreachable'] is True

    # Add source
    new_src = tmp_path / 'new_src'
    new_src.mkdir()
    res = client.post('/api/project/sources', params=dict(id=pid), json=dict(path=str(new_src)))
    assert res.status_code == 200
    assert len(res.json()['sources']) == 2

    # Remove source
    res = client.delete('/api/project/sources', params=dict(id=pid, path=str(new_src)))
    assert res.status_code == 200
    assert len(res.json()['sources']) == 1

def test_server_api_fullres_and_transform(client, project):
    pid, sid = project
    items = client.get('/api/project/items', params=dict(id=pid, source=sid)).json()
    iid = items[0]['id']

    # Fullres status
    res = client.get('/api/project/fullres', params=dict(id=pid, item=iid))
    assert res.status_code == 200
    assert res.json()['state'] in ('idle', 'ready', 'running')

    # Transform
    res = client.get('/api/project/transform', params=dict(id=pid, item=iid, transform='original'))
    assert res.status_code == 200
    info = res.json()
    assert type(info) == dict

    # Detail
    res = client.get('/api/project/detail', params=dict(id=pid, item=iid))
    assert res.status_code == 200
    assert 'native' in res.json()



def test_server_api_job_failures(client, project, monkeypatch):
    pid, sid = project
    items = client.get('/api/project/items', params=dict(id=pid, source=sid)).json()
    iid = items[0]['id']
    q = dict(id=pid, item=iid)
    body = dict(options=dict(size='native', sharpen=False, quality=88))

    from stratastills import pipeline as PL
    def failing_generate(*args, **kw): raise RuntimeError("Simulated generate failure")
    monkeypatch.setattr(PL, 'generate', failing_generate)

    def dummy_find(*args, **kw): return "dummy_key", None
    monkeypatch.setattr(PL, 'find_existing', dummy_find)

    res = client.post('/api/project/generate', params=q, json=body).json()
    assert res['state'] in ('running', 'done', 'failed')

    j = wait(client, '/api/project/generate', **q)
    assert j['state'] == 'failed'
    assert "Simulated generate failure" in j['error']

    def failing_apply(*args, **kw): raise RuntimeError("Simulated apply failure")
    monkeypatch.setattr(PL, 'apply_rules', failing_apply)
    client.put('/api/project/rules', params=dict(id=pid), json=dict(rules=[dict(source=sid, pipeline='web')]))

    res = client.post('/api/project/apply', params=dict(id=pid)).json()
    assert res['state'] in ('running', 'done', 'failed')

    j = wait(client, '/api/project/apply', id=pid)
    assert j['state'] == 'failed'
    assert "Simulated apply failure" in j['text']


def test_server_main_entrypoint(monkeypatch):
    import os
    from stratastills.server import app

    # Prevent uvicorn from blocking
    import uvicorn
    # Mock uvicorn run to just record that it was called
    call_args = {}
    def mock_run(*args, **kw):
        call_args['args'] = args
        call_args['kw'] = kw
    monkeypatch.setattr(uvicorn, 'run', mock_run)

    app.main(['--host', '127.0.0.1', '--port', '9999'])
    assert call_args['kw']['host'] == '127.0.0.1'
    assert call_args['kw']['port'] == 9999

def test_server_api_get_settings_and_detail_error(client, project, monkeypatch):
    pid, sid = project
    # Test get settings
    res = client.get('/api/settings')
    assert res.status_code == 200
    assert 'projects_dir' in res.json()

    items = client.get('/api/project/items', params=dict(id=pid, source=sid)).json()
    iid = items[0]['id']

    # Test detail error
    from stratastills import export as X
    def mock_measure_detail(*args, **kw): raise RuntimeError("Simulated detail error")
    monkeypatch.setattr(X, 'measure_detail', mock_measure_detail)

    res = client.get('/api/project/detail', params=dict(id=pid, item=iid))
    assert res.status_code == 400
    assert "Simulated detail error" in res.json()['detail']

def test_server_api_new_project_error(client, project, monkeypatch, tmp_path):
    pid, sid = project
    # Trigger a ProjectError by trying to create a project that already exists
    res = client.post('/api/projects', json={'name': 'Demo', 'sources': []})
    assert res.status_code == 400

    # Also test put_settings error
    res = client.put('/api/settings', json={'projects_dir': 'relative'})
    assert res.status_code == 400

    # Test settings success
    res = client.put('/api/settings', json={'projects_dir': str(tmp_path)})
    assert res.status_code == 200
