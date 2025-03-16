"""Database management for WesTube."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def get_engine(db_path: str = "westube.db"):
    """Create a SQLAlchemy engine.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        SQLAlchemy engine instance.
    """
    return create_engine(f"sqlite:///{db_path}")


def get_session(engine=None) -> Session:
    """Get a new database session.

    Args:
        engine: Optional SQLAlchemy engine instance.

    Returns:
        New SQLAlchemy session.
    """
    if engine is None:
        engine = get_engine()
    return Session(engine)
