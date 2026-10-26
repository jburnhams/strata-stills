"""The HTTP API over real project files: browse a project, make a variant, download it, apply a rule."""
import time


def wait(client, url, **params):
    for _ in range(600):
        j = client.get(url, params=params).json()
        if j['state'] != 'running': return j
        time.sleep(0.1)
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
