"""Raw SHA-256 into schema photos: planning needs no database, writing is checked on SQLite (the `db_session` fixture; `pytest --pg` runs the same on Postgres)."""
import json

import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from sqlalchemy import func, select

from stratastills.db import hashes
from stratastills.db.models import File, Photo

A, B, C = 'a' * 64, 'b' * 64, 'c' * 64


def rec(path, sha, size=10, mtime=1): return dict(path=str(path), sha256=sha, size=size, mtime_ns=mtime)


@pytest.fixture
def roots(tmp_path):
    r, d = tmp_path / 'raw', tmp_path / 'drone'; r.mkdir(); d.mkdir(); return r, d


def test_plan_labels_roots_and_reports_duplicates_and_problems(roots, tmp_path):
    r, d = roots
    p = hashes.plan([rec(r / '2012' / 'DSC00650.ARW', A), rec(d / 'x' / 'DJI_0001.DNG', B), rec(r / 'copy' / 'DSC00650.ARW', A), rec(tmp_path / 'elsewhere' / 'z.ARW', C), rec(r / 'bad.ARW', 'xyz'), rec(r / '2012' / 'DSC00650.ARW', A)], [r, d])
    assert p['counts'] == dict(files=3, photos=2, duplicate_hashes=1, duplicate_files=2, roots=['drone', 'raw'])
    assert {f['relpath'] for f in p['files']} == {'2012/DSC00650.ARW', 'x/DJI_0001.DNG', 'copy/DSC00650.ARW'}
    assert p['photos'][A] == 'DSC00650' and len(p['problems']) == 2 and any('none of the source roots' in x for x in p['problems'])


def test_read_jsonl_skips_a_half_written_line_and_a_missing_file_is_fatal(tmp_path):
    f = tmp_path / 'h.jsonl'; f.write_text(json.dumps(rec('/a/b.ARW', A)) + '\n{"path": "/a/c', encoding='utf8'); problems = []
    assert len(hashes.read_jsonl(f, problems)) == 1 and 'skipped' in problems[0]
    gone = []; hashes.read_jsonl(tmp_path / 'nope.jsonl', gone)
    assert gone[0].startswith('FATAL')


def test_write_then_again_adds_nothing_and_a_duplicate_is_one_photo(roots, db_session):
    r, d = roots; p = hashes.plan([rec(r / 'a.ARW', A), rec(r / 'b.ARW', B), rec(d / 'copy.ARW', A)], [r, d])
    assert hashes.write(db_session, p) == dict(photo=2, file=3, superseded=0)
    assert hashes.write(db_session, p) == dict(photo=0, file=0, superseded=0)
    assert db_session.scalar(select(func.count()).select_from(Photo)) == 2 and db_session.scalar(select(func.count()).select_from(File)) == 3
    ids = {f.root: f.photo_id for f in db_session.scalars(select(File).where(File.sha256 == A))}
    assert ids['raw'] == ids['drone'] and all(f.role == 'raw' for f in db_session.scalars(select(File)))


def test_a_changed_file_is_a_new_row_and_the_old_one_stays(roots, db_session):
    r, d = roots; hashes.write(db_session, hashes.plan([rec(r / 'a.ARW', A, 10, 1)], [r]))
    assert hashes.write(db_session, hashes.plan([rec(r / 'a.ARW', B, 12, 2)], [r])) == dict(photo=1, file=1, superseded=1)
    rows = list(db_session.scalars(select(File).order_by(File.mtime_ns)))
    assert [(f.sha256, f.current) for f in rows] == [(A, False), (B, True)]
    assert db_session.scalar(select(func.count()).select_from(Photo)) == 2


def test_an_unreadable_source_is_refused(roots, db_session):
    p = hashes.plan([], list(roots), ['FATAL unreadable x'])
    with pytest.raises(hashes.HashMismatch): hashes.write(db_session, p)


def test_an_older_record_loaded_after_a_newer_one_leaves_one_current_row_the_newer(roots, db_session):
    r, _ = roots; hashes.write(db_session, hashes.plan([rec(r / 'a.ARW', B, 12, 2)], [r]))              # hash_new recorded the newer version first
    assert hashes.write(db_session, hashes.plan([rec(r / 'a.ARW', A, 10, 1)], [r])) == dict(photo=1, file=1, superseded=1)
    rows = list(db_session.scalars(select(File).order_by(File.mtime_ns)))
    assert [(f.sha256, f.current) for f in rows] == [(A, False), (B, True)]
    assert hashes.write(db_session, hashes.plan([rec(r / 'a.ARW', A, 10, 1)], [r])) == dict(photo=0, file=0, superseded=0)


def test_a_load_bigger_than_a_chunk_stays_fast_and_correct(roots, db_session):
    """The verification used to rescan every planned file for every path (quadratic: 7 minutes for 21k files); 3 chunks of paths must write, verify and rerun in a blink, and a changed file among them still supersedes only its own path."""
    import time
    r, d = roots; n = 3 * hashes.CHUNK + 7
    recs = [rec(r / f'y{i // 100}' / f'DSC{i:05d}.ARW', f'{i:064x}', size=i + 1, mtime=i + 1) for i in range(n)]
    t = time.perf_counter(); assert hashes.write(db_session, hashes.plan(recs, [r, d])) == dict(photo=n, file=n, superseded=0); took = time.perf_counter() - t
    assert took < 60
    assert hashes.write(db_session, hashes.plan(recs, [r, d])) == dict(photo=0, file=0, superseded=0)
    changed = [dict(recs[5], sha256='f' * 64, mtime_ns=999)]
    assert hashes.write(db_session, hashes.plan(changed, [r, d])) == dict(photo=1, file=1, superseded=1)
    assert db_session.scalar(select(func.count()).select_from(File).where(File.current.is_(True))) == n
