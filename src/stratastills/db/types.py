"""Column types that are not in SQLAlchemy's generic set. The app runs on real Postgres only, but the unit tests run on SQLite for speed, so each type says what it is on Postgres and what stands in for it elsewhere."""

import sqlalchemy as sa
from sqlalchemy.types import TypeDecorator


class Vector(TypeDecorator):
    """An embedding: pgvector's `vector` on Postgres (`dim` None = any length per row; an index needs a fixed length, so make a partial index per model when search needs one), a JSON list of floats on SQLite so the tables still migrate and round-trip in the fast tests (similarity search is Postgres only and its tests need a real server)."""

    impl = sa.JSON
    cache_ok = True

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import VECTOR

            return dialect.type_descriptor(VECTOR(self.dim))
        return dialect.type_descriptor(sa.JSON())

    def process_bind_param(self, value, dialect):
        if value is None or dialect.name == "postgresql":
            return value
        return [float(x) for x in value]
