"""Pytest configuration for WesTube tests."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from wes_tube.models.base import Base


@pytest.fixture
def engine():
    """Create a test database engine."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(engine):
    """Create a test database session."""
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
