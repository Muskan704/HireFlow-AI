"""
Database engine + table creation.

This is the piece that turns db_models.py (table DEFINITIONS) into actual
tables inside your Supabase/Neon Postgres instance.

Usage
-----
One-time setup (run this once, or any time you add/change a model in
db_models.py):

    python -m app.core.database

This connects using settings.database_url and creates any tables that don't
already exist. It does NOT drop or modify existing tables — SQLModel's
create_all() is additive only, so it's safe to re-run.

Using it inside the app (later, in pipeline.py / session_repo.py):

    from app.core.database import get_session

    with get_session() as session:
        session.add(some_record)
        session.commit()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlmodel import SQLModel, Session, create_engine
from loguru import logger

from app.core.config import get_settings

# Import db_models so SQLModel's metadata actually knows about every table.
# This import has no visible effect in the code below, but without it,
# create_all() would create zero tables — SQLModel only registers a model
# once its class body has actually executed.
from app.models import db_models  # noqa: F401


settings = get_settings()

if not settings.database_url:
    logger.warning(
        "DATABASE_URL is not set — persistence layer will fail on first use. "
        "Add DATABASE_URL to your .env (see Supabase/Neon setup)."
    )

# echo=False in normal use; flip to True temporarily if you need to see the
# raw SQL being executed, e.g. while debugging a query.
engine = create_engine(settings.database_url, echo=False, pool_pre_ping=True)


def init_db() -> None:
    """
    Create every table defined in db_models.py that doesn't already exist.
    Safe to call multiple times — existing tables are left untouched.
    """
    logger.info("Creating tables (if not already present)...")
    SQLModel.metadata.create_all(engine)
    logger.info("Done. Tables are ready in your Postgres instance.")


@contextmanager
def get_session() -> Iterator[Session]:
    """
    Context-managed DB session. Use as:

        with get_session() as session:
            session.add(record)
            session.commit()

    Rolls back automatically if an exception is raised inside the block, so
    a failed pipeline run doesn't leave a half-written session in the DB.
    """
    session = Session(engine)
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    # Entry point for: python -m app.core.database
    init_db()