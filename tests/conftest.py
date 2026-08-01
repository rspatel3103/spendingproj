"""Shared pytest fixtures.

`test_sessionmaker` points the app's DB access at an isolated temp-file
SQLite database for the duration of a test, instead of the real
cashflow.db. It patches two things:
  - app.db.session's own `engine`/`async_session_factory` -- functions
    defined *inside* app/db/session.py (like `get_session()`, used by
    `Depends(get_session)` routers) resolve these as module globals at
    call time, so patching the module attribute is enough for them.
  - app.agents.forecaster's `async_session_factory` -- that module did
    `from app.db.session import async_session_factory` at import time,
    which copies the reference into its own namespace; patching
    app.db.session afterwards would NOT be seen there, so it needs its
    own patch (the classic "patch where it's used" mocking rule).

`high_confidence_result`/`low_confidence_result` are fixed
CategorizationResult fixtures -- the categorizer/apply-layer tests never
call the real LLM, so there's nothing to mock beyond providing these.
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.categorizer import CategorizationResult
from app.db.models import Base


@pytest_asyncio.fixture
async def test_sessionmaker(monkeypatch, tmp_path):
    db_path = tmp_path / f"test_{uuid.uuid4().hex}.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.agents.forecaster as forecaster_module
    import app.agents.dispatcher as dispatcher_module
    import app.db.session as db_session_module

    monkeypatch.setattr(db_session_module, "engine", engine)
    monkeypatch.setattr(db_session_module, "async_session_factory", sessionmaker)
    monkeypatch.setattr(forecaster_module, "async_session_factory", sessionmaker)
    monkeypatch.setattr(dispatcher_module, "async_session_factory", sessionmaker)

    yield sessionmaker

    await engine.dispose()


@pytest.fixture
def high_confidence_result() -> CategorizationResult:
    return CategorizationResult(
        category="Dining",
        subcategory="coffee shop",
        confidence=0.9,
        reasoning="Fixed test fixture: strong match to a known coffee vendor.",
    )


@pytest.fixture
def low_confidence_result() -> CategorizationResult:
    return CategorizationResult(
        category="Shopping",
        subcategory="online retail",
        confidence=0.6,
        reasoning="Fixed test fixture: weak/ambiguous retrieval context.",
    )
