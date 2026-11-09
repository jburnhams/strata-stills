"""The standard database checks from omarchy-db: the migrations run from empty, have one head, downgrade step by step and match the models.
SQLite by default, a throwaway Postgres with `pytest --pg`; never the real database."""
import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from omarchy_db.testing import test_migrations, test_schema_matches_models  # noqa: E402,F401
