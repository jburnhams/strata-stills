"""arena_entry.revision

Revision ID: 3b7d2e91c4a0
Revises: c5f892fbf963
Create Date: 2026-11-11 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3b7d2e91c4a0'
down_revision: Union[str, Sequence[str], None] = 'c5f892fbf963'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('arena_entry', schema=None) as batch_op:
        batch_op.add_column(sa.Column('revision', sa.JSON(none_as_null=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('arena_entry', schema=None) as batch_op:
        batch_op.drop_column('revision')
