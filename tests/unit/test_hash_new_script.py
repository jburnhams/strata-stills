"""scripts/hash_new.py helpers that need no database: the hash and the candidate walk."""
import hashlib, importlib.util, os

spec = importlib.util.spec_from_file_location('hash_new', os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'hash_new.py'))
hn = importlib.util.module_from_spec(spec); spec.loader.exec_module(hn)


def test_sha256_and_candidates(tmp_path):
    (tmp_path / 'a').mkdir(); f = tmp_path / 'a' / 'x.ARW'; f.write_bytes(b'abc' * 3_000_000); (tmp_path / 'a' / 'y.txt').write_text('no')
    assert hn.sha256_of(f) == hashlib.sha256(b'abc' * 3_000_000).hexdigest()
    assert list(hn.candidates([str(tmp_path)], {'.arw'})) == [str(f)]
