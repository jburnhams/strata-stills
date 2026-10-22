import json

from stratastills import settings


def test_doctor_reports_the_environment(cli):
    r = cli('doctor', check=False); assert 'stratastills' in r.stdout and 'numpy' in r.stdout and 'rawpy' in r.stdout and r.code == 0 and ('GPU' in r.stdout or 'CPU' in r.stdout or 'torch' in r.stdout)


def test_scan_groups_the_shots_of_a_folder(cli, shoot):
    shots = json.loads(cli('scan', shoot, '--json').stdout); by = {s['stem']: s for s in shots}
    assert sorted(by) == ['DJI_20250402190111_0001_D', 'DJI_20250402190111_0001_D-Pano', 'DJI_20250402190211_0002_D', 'DJI_20250402190511_0009_D']
    assert by['DJI_20250402190111_0001_D']['kind'] == 'still' and by['DJI_20250402190111_0001_D-Pano']['kind'] == 'pano' and by['DJI_20250402190511_0009_D']['kind'] == 'video'
    assert by['DJI_20250402190111_0001_D-Pano']['parent'] == 'DJI_20250402190111_0001_D' and by['DJI_20250402190111_0001_D-Pano']['raw'].endswith('.dng')


def test_scan_info_reads_the_drone_metadata(cli, shoot):
    shots = {s['stem']: s for s in json.loads(cli('scan', shoot, '--json', '--info').stdout)}
    nadir, oblique = shots['DJI_20250402190111_0001_D']['info'], shots['DJI_20250402190211_0002_D']['info']
    assert nadir['nadir'] is True and nadir['dji']['GimbalPitchDegree'] == -90.0 and oblique['nadir'] is False and nadir['width'] == 800 and nadir['taken'].startswith('2025-04-02T19:01:11')
    text = cli('scan', shoot, '--info').stdout; assert 'nadir' in text and '4 shots (2 stills, 1 videos)' in text                                                      # the panorama is counted on its own


def test_projects_are_created_listed_and_changed_from_the_command_line(cli, shoot, tmp_path, monkeypatch):
    assert 'none yet' in cli('projects').stdout
    assert 'created demo' in cli('new', 'Demo', '--source', shoot).stdout and "'Demo'" in cli('projects').stdout and '1 source folder' in cli('projects').stdout
    other = tmp_path / 'more'; other.mkdir(); assert '2 source folder(s)' in cli('source', 'demo', str(other)).stdout and '1 source folder(s)' in cli('source', 'demo', str(other), '--remove').stdout
    assert cli('new', 'Demo', check=False).code != 0                                                                                  # a name already taken
    monkeypatch.delenv('STRATASTILLS_PROJECTS')                                                                                   # the environment variable beats the setting
    elsewhere = tmp_path / 'elsewhere'; assert str(elsewhere) in cli('set', 'projects_dir', elsewhere).stdout and settings.projects_dir() == str(elsewhere.resolve())
    assert 'projects_dir' in cli('set', 'projects_dir', 'default').stdout and cli('set', 'nope', 'x', check=False).code != 0


def test_the_command_line_runs_as_a_real_process(cli_subprocess, shoot):
    assert 'stratastills' in cli_subprocess('doctor', check=False).stdout and '4 shots' in cli_subprocess('scan', shoot).stdout
