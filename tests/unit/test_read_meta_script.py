"""scripts/read_meta.py: an exiftool error record is not a success, and a rerun retries it."""
import importlib.util, json, os

spec = importlib.util.spec_from_file_location('read_meta', os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'read_meta.py'))
rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)


def test_is_error():
    assert rm.is_error({'ExifTool:Error': 'File format error'}) and rm.is_error({'Error': 'x'}) and not rm.is_error({'IFD0:Make': 'SONY', 'ExifTool:Warning': 'minor'})


def test_resume_retries_error_records_and_keeps_good_ones(tmp_path, monkeypatch, capsys):
    h = tmp_path / 'h.jsonl'; out = tmp_path / 'm.jsonl'
    h.write_text(''.join(json.dumps(dict(path=f'/r/{n}.ARW', sha256=n * 64, size=1, mtime_ns=1)) + '\n' for n in 'abc'), encoding='utf8')
    out.write_text(json.dumps(dict(path='/r/a.ARW', sha256='a' * 64, size=1, mtime_ns=1, tags={'IFD0:Make': 'X'})) + '\n' + json.dumps(dict(path='/r/b.ARW', sha256='b' * 64, size=1, mtime_ns=1, tags={'ExifTool:Error': 'bad'})) + '\n', encoding='utf8')
    seen = []
    def fake(paths): seen.extend(paths); return {p: ({'ExifTool:Error': 'still bad'} if p.endswith('c.ARW') else {'IFD0:Make': 'Y'}) for p in paths}
    monkeypatch.setattr(rm, 'run_exiftool', fake); monkeypatch.setattr(rm.shutil, 'which', lambda n: '/bin/' + n); monkeypatch.setattr(rm, 'lower_priority', lambda: None)
    monkeypatch.setattr('sys.argv', ['read_meta.py', '--hashes', str(h), '--out', str(out)])
    assert rm.main() == 0 and sorted(seen) == ['/r/b.ARW', '/r/c.ARW']                                         # a is done, b (an error before) is retried, c is new
    lines = [json.loads(x) for x in out.read_text(encoding='utf8').splitlines()]
    assert [x['path'] for x in lines] == ['/r/a.ARW', '/r/b.ARW', '/r/b.ARW'] and 'read 1, unreadable 1' in capsys.readouterr().out    # c's error is not written
