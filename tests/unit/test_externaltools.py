"""externaltools.py: finding darktable and RawTherapee (on the PATH, then where installers put them) and RawTherapee's profile folder."""
import os, pytest

from stratastills import externaltools as E


def test_a_program_on_the_path_is_found_there(monkeypatch):
    monkeypatch.setattr(E.shutil, 'which', lambda name: '/usr/bin/' + name)
    assert E.find('darktable') == '/usr/bin/darktable-cli'


def test_otherwise_the_last_match_of_the_installer_locations_wins(monkeypatch, tmp_path):
    for v in ('1.0', '2.0'): d = tmp_path / v; d.mkdir(); (d / 'rawtherapee-cli').write_text('')
    monkeypatch.setattr(E.shutil, 'which', lambda name: None); monkeypatch.setitem(E.EXTRA, 'rawtherapee', [str(tmp_path / '*' / 'rawtherapee-cli')])
    assert E.find('rawtherapee') == str(tmp_path / '2.0' / 'rawtherapee-cli')                            # the newest version of several installs
    monkeypatch.setitem(E.EXTRA, 'rawtherapee', [str(tmp_path / 'none' / '*')]); assert E.find('rawtherapee') is None


def test_the_profile_folder_is_found_next_to_the_program_and_none_without_it(monkeypatch, tmp_path):
    exe = tmp_path / 'bin' / 'rawtherapee-cli'; prof = tmp_path / 'bin' / 'profiles'; prof.mkdir(parents=True); exe.write_text('')
    monkeypatch.setattr(E, 'find', lambda p: str(exe))
    assert E.rt_profiles() is None                                                                       # the folder exists but has no Auto-Matched Curve profile (a real install may exist on this machine: checked below)
    (prof / 'Auto-Matched Curve - ISO Low.pp3').write_text(''); assert E.rt_profiles() == str(prof)
    monkeypatch.setattr(E, 'find', lambda p: None); monkeypatch.setattr(E.os.path, 'exists', lambda p: False); assert E.rt_profiles() is None


def test_missing_lists_what_is_not_installed_and_a_missing_profile_folder(monkeypatch):
    monkeypatch.setattr(E, 'find', lambda p: None if p == 'darktable' else '/x/rt'); monkeypatch.setattr(E, 'rt_profiles', lambda: None)
    assert E.missing() == ['darktable', "RawTherapee's profiles folder"]
    monkeypatch.setattr(E, 'find', lambda p: '/x'); monkeypatch.setattr(E, 'rt_profiles', lambda: '/p'); assert E.missing() == []


def test_require_stops_with_what_to_install(monkeypatch):
    monkeypatch.setattr(E, 'missing', lambda: ['darktable'])
    with pytest.raises(SystemExit) as e: E.require()
    assert 'missing: darktable' in str(e.value) and 'pacman' in str(e.value)
    monkeypatch.setattr(E, 'missing', lambda: []); E.require()
