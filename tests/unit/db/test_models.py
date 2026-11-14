"""The photos schema on SQLite (fast; the fixtures come from the strata_python.db plugin). Tests that need real Postgres features (pgvector similarity, the extension) are marked `pg` and run with `pytest --pg`."""

from datetime import UTC

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from stratastills.db.models import Analysis, ArenaFlag, ArenaFlagEvent, ArenaPic, ArenaVote, Embedding, Photo


def test_photo_identity_is_the_sha256(db_session):
    db_session.add(Photo(sha256="a" * 64, stem="DSC00010"))
    db_session.commit()
    db_session.add(Photo(sha256="a" * 64))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_vote_undo_marks_and_never_deletes(db_session):
    from datetime import datetime

    db_session.add(ArenaPic(name="2023-01-11_DSC00010"))
    db_session.flush()
    db_session.add(
        ArenaVote(
            pic="2023-01-11_DSC00010",
            a="lightroom",
            b="auto1",
            winner="a",
            voted_at=1793841361.0,
            ms=4780,
            legacy_key="k1",
            raw={"a": "lightroom"},
        )
    )
    db_session.commit()
    v = db_session.scalars(select(ArenaVote)).one()
    v.undone_at = datetime.now(UTC).replace(tzinfo=None)
    db_session.commit()
    assert (
        db_session.scalars(select(ArenaVote).where(ArenaVote.undone_at.is_(None))).all() == []
        and len(db_session.scalars(select(ArenaVote)).all()) == 1
    )


def test_a_vote_import_is_repeatable(db_session):
    db_session.add(ArenaPic(name="p"))
    db_session.flush()
    db_session.add_all(
        [
            ArenaVote(pic="p", a="x", b="y", winner="tie", voted_at=1.0, legacy_key="same"),
            ArenaVote(pic="p", a="x", b="y", winner="tie", voted_at=1.0, legacy_key="same"),
        ]
    )
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_flag_keeps_its_history(db_session):
    db_session.add(ArenaPic(name="p"))
    db_session.flush()
    db_session.add(ArenaFlag(id="907f1ef0ea", pic="p", variant="auto1"))
    db_session.commit()
    db_session.add_all(
        [
            ArenaFlagEvent(flag_id="907f1ef0ea", at=1.0, status="investigating"),
            ArenaFlagEvent(flag_id="907f1ef0ea", at=2.0, status="fixed"),
        ]
    )
    db_session.commit()
    assert [e.status for e in db_session.scalars(select(ArenaFlagEvent).order_by(ArenaFlagEvent.at))] == [
        "investigating",
        "fixed",
    ]


def test_analysis_is_one_row_per_input(db_session):
    p = Photo(sha256="b" * 64)
    db_session.add(p)
    db_session.flush()
    db_session.add(
        Analysis(photo_id=p.id, kind="scene", producer="clip", producer_version="1", input_key="x", label="beach")
    )
    db_session.commit()
    db_session.add(Analysis(photo_id=p.id, kind="scene", producer="clip", producer_version="1", input_key="x"))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
    db_session.add(Analysis(photo_id=p.id, kind="scene", producer="clip", producer_version="1", input_key="y"))
    db_session.commit()  # the input changed: a new row, the old one stays
    assert len(db_session.scalars(select(Analysis)).all()) == 2


def test_embedding_round_trips_as_a_list(db_session):
    p = Photo(sha256="c" * 64)
    db_session.add(p)
    db_session.flush()
    db_session.add(Embedding(photo_id=p.id, model="clip-pca32", vector=[0.5, -1, 2]))
    db_session.commit()
    assert db_session.scalars(select(Embedding.vector)).one() == [0.5, -1.0, 2.0]


@pytest.mark.pg
def test_embedding_is_a_pgvector_column(db_engine, db_backend):
    if db_backend != "postgres":
        pytest.skip("needs --pg")
    with db_engine.connect() as c:
        assert (
            c.exec_driver_sql(
                "select data_type, udt_name from information_schema.columns where table_name='embedding' and column_name='vector'"
            ).one()[1]
            == "vector"
        )
