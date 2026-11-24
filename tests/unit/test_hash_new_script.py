"""scripts/hash_new.py helpers that need no database: the candidate walk (the hash and the low priority are strata-python's files.sha256_hex and oslib.lower_priority)."""

import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "hash_new", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "hash_new.py")
)
hn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hn)


def test_candidates(tmp_path):
    (tmp_path / "a").mkdir()
    f = tmp_path / "a" / "x.ARW"
    f.write_bytes(b"abc")
    (tmp_path / "a" / "y.txt").write_text("no")
    assert list(hn.candidates([str(tmp_path)], {".arw"})) == [str(f)]
