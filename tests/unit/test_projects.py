import os, pytest
from stratastills import projects as P, settings


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / 'projects'; monkeypatch.setenv('STRATASTILLS_PROJECTS', str(r)); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'home')); return str(r)


def test_default_projects_dir_is_in_the_repo(monkeypatch, tmp_path):
    monkeypatch.delenv('STRATASTILLS_PROJECTS', raising=False); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path))
    assert settings.DEFAULT_PROJECTS_DIR == os.path.join(settings.REPO, 'projects') and os.path.isfile(os.path.join(settings.REPO, 'pyproject.toml'))
    monkeypatch.setattr(settings, 'DEFAULT_PROJECTS_DIR', str(tmp_path / 'p')); assert settings.projects_dir() == str(tmp_path / 'p')


def test_settings_file_and_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv('STRATASTILLS_PROJECTS', raising=False); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'h')); monkeypatch.setattr(settings, 'DEFAULT_PROJECTS_DIR', str(tmp_path / 'default'))
    settings.save(dict(projects_dir=str(tmp_path / 'mine'))); assert settings.projects_dir() == str(tmp_path / 'mine')
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'env')); assert settings.projects_dir() == str(tmp_path / 'env')
    with pytest.raises(KeyError): settings.save(dict(nope=1))
    monkeypatch.delenv('STRATASTILLS_PROJECTS'); settings.save(dict(projects_dir=None)); assert settings.projects_dir() == str(tmp_path / 'default')


def test_create_list_and_sources(root, tmp_path):
    a = tmp_path / 'a'; b = tmp_path / 'b'; a.mkdir(); b.mkdir(); (a / 'DJI_1_0001_D.JPG').write_bytes(b'x'); (b / 'DJI_2_0001_D.DNG').write_bytes(b'x')
    d = P.create('Hazelmere 2025', [str(a)])
    assert d['id'] == 'hazelmere-2025' and os.path.isfile(os.path.join(root, 'hazelmere-2025', 'project.json'))
    P.add_source(d['id'], str(b))
    with pytest.raises(P.ProjectError): P.add_source(d['id'], str(b))
    with pytest.raises(P.ProjectError): P.add_source(d['id'], str(tmp_path / 'nope'))
    assert [s.stem for _, s in P.shots(d['id'])] == ['DJI_1_0001_D', 'DJI_2_0001_D']
    P.remove_source(d['id'], str(a)); assert len(P.load(d['id'])['sources']) == 1
    with pytest.raises(P.ProjectError): P.remove_source(d['id'], str(a))


def test_nested_projects_are_listed_and_duplicates_refused(root):
    P.create('One'); P.create('Two', group='2025/drone')
    assert sorted(p['id'] for p in P.list_projects()) == ['2025/drone/two', 'one']
    with pytest.raises(P.ProjectError): P.create('One')
    with pytest.raises(P.ProjectError): P.create('  ')


def test_ids_cannot_leave_the_projects_folder(root):
    for bad in ('../x', '..', ''):
        with pytest.raises(P.ProjectError): P.load(bad)


def test_unreachable_source_does_not_fail_the_rest(root, tmp_path):
    a = tmp_path / 'a'; a.mkdir(); d = P.create('P', [str(a)]); a.rmdir()
    assert P.shots(d['id']) == [(str(a), None)]
