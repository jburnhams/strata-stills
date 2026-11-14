"""project_source.stems

Revision ID: 5e1a0c7d9b23
Revises: 3b7d2e91c4a0
Create Date: 2026-11-14 08:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5e1a0c7d9b23'
down_revision: Union[str, Sequence[str], None] = '3b7d2e91c4a0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('project_source', schema=None) as batch_op:
        batch_op.add_column(sa.Column('stems', sa.JSON(none_as_null=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('project_source', schema=None) as batch_op:
        batch_op.drop_column('stems')
