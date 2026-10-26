"""Test projects.py edge cases: folders, missing projects, UNC paths, and handling unreachable sources."""
import os
import pytest
from stratastills import ingest, projects, settings


def test_project_folder_errors(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects'))
    with pytest.raises(projects.ProjectError, match='bad project id'):
        projects.folder('../outside')

    with pytest.raises(projects.ProjectError, match='no project'):
        projects.load('does_not_exist')

    projects.create('test', group='group')
    with pytest.raises(projects.ProjectError, match='already exists'):
        projects.create('test', group='group')


def test_list_projects_max_depth(tmp_path, monkeypatch):
    root = tmp_path / 'projects'
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(root))

    # Create projects at depth 1, 2, 3, 4
    projects.create('d1', group=None)
    projects.create('d2', group='group1')
    projects.create('d3', group='group1/group2')
    # Depth 4 project creation
    d4_pid = 'group1/group2/group3/d4'
    d4_folder = root / d4_pid
    d4_folder.mkdir(parents=True)
    with open(d4_folder / 'project.json', 'w') as f:
        f.write('{"schema": 1, "id": "%s", "name": "d4", "sources": []}' % d4_pid)

    listed = projects.list_projects()
    found = {p['name'] for p in listed}
    assert 'd1' in found
    assert 'd2' in found
    assert 'd3' in found
    assert 'd4' not in found, "Project at depth 4 should be ignored due to MAX_DEPTH=3"


def test_add_and_remove_sources_errors(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects'))
    pid = projects.create('test')['id']

    with pytest.raises(projects.ProjectError, match='not a folder'):
        projects.add_source(pid, str(tmp_path / 'does_not_exist'))

    src = tmp_path / 'src'
    src.mkdir()
    projects.add_source(pid, str(src))

    with pytest.raises(projects.ProjectError, match='already a source'):
        projects.add_source(pid, str(src))

    with pytest.raises(projects.ProjectError, match='not a source'):
        projects.remove_source(pid, str(tmp_path / 'other'))


def test_resolve_unc_paths(monkeypatch):
    # Mock settings.source_roots to return a specific root
    monkeypatch.setattr(settings, 'source_roots', lambda: [os.path.normcase(os.path.normpath('/home/foo/raw'))])

    # Simulate a path that was stored relative to the root on Linux
    assert projects._rel_of(os.path.join('/home/foo/raw', 'shoot')) == 'shoot'

    # Simulate a UNC path that represents the root itself
    assert projects._rel_of('\\\\Server\\raw') is None

    # Simulate a UNC path that is inside the share
    assert projects._rel_of('\\\\Server\\raw\\shoot') == 'shoot'

    # Ensure resolve works with both
    orig_isdir = os.path.isdir
    monkeypatch.setattr(os.path, 'isdir', lambda p: True if p == os.path.join('/home/foo/raw', 'shoot') else orig_isdir(p))
    assert projects.resolve({'path': '\\\\Server\\raw\\shoot', 'rel': 'shoot'}) == os.path.join('/home/foo/raw', 'shoot')
    assert projects.resolve({'path': '/unknown/path'}) == '/unknown/path'

def test_shots_ignores_offline_drives(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects'))
    pid = projects.create('test')['id']
    src = tmp_path / 'src'
    src.mkdir()
    projects.add_source(pid, str(src))

    # Monkeypatch ingest.scan to always raise OSError for this test
    monkeypatch.setattr(ingest, 'scan', lambda *args, **kw: (_ for _ in ()).throw(OSError("Share unreachable")))

    # Verify that shots() does not crash, but returns None for the unreachable source
    res = projects.shots(pid)
    assert len(res) == 1
    assert res[0][1] is None

    # Verify that source_summaries flags it as unreachable
    sums = projects.source_summaries(pid)
    assert len(sums) == 1
    assert sums[0]['reachable'] is False

def test_generated_variants_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects'))
    pid = projects.create('test')['id']

    gen_dir = tmp_path / 'projects' / pid / 'generated' / 'item_id'
    gen_dir.mkdir(parents=True)

    # Write a valid json file but the referenced file is missing
    with open(gen_dir / '123.json', 'w') as f:
        f.write('{"key": "123", "output": [800, 600], "file": "does_not_exist.jpg"}')

    # Write an invalid json file
    with open(gen_dir / 'bad.json', 'w') as f:
        f.write('{')

    variants = projects.generated_variants(pid, 'item/id')
    assert variants == []
