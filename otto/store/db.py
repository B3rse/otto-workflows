"""
Otto store/db.py
----------------
Engine creation, session management, and schema initialization.

Usage:
    from otto.store.db import get_engine, get_session, init_db

    engine = get_engine("/path/to/project/.otto/otto.db")
    init_db(engine)          # creates tables + runs migrations

    with get_session(engine) as session:
        session.add(some_model)
        session.commit()
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event as sa_event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

# Import tables so SQLModel metadata is populated before init_db is called
from otto.store.tables import (  # noqa: F401
    Artifact,
    Edge,
    Event,
    ExternalJob,
    Lease,
    Run,
    TaskRun,
)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def get_engine(db_path: str | Path) -> Engine:
    """
    Create a SQLite engine for the given path.
    Enables WAL mode and foreign key enforcement on every connection.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"sqlite:///{db_path}"
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        echo=False, # -> set true for debugging
    )

    @sa_event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        # WAL for concurrent reads during long scheduler loops
        cursor.execute("PRAGMA journal_mode=WAL")
        # Enforce FK constraints (SQLite doesn't by default)
        cursor.execute("PRAGMA foreign_keys=ON")
        # Reasonable cache size (10 MB)
        cursor.execute("PRAGMA cache_size=-10000")
        cursor.close()

    return engine


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db(engine: Engine) -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    SQLModel.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

@contextmanager
def get_session(engine: Engine) -> Generator[Session, None, None]:
    """Context manager yielding a SQLModel Session."""
    with Session(engine) as session:
        yield session
