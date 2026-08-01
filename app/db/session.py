"""Async database engine and session factory.

Creates the SQLAlchemy async engine from `settings.DATABASE_URL` and
exposes an `async_sessionmaker` plus a `get_session` dependency for use
in FastAPI routes via `Depends`. Supports both `sqlite+aiosqlite://`
(local/dev default) and `postgresql+asyncpg://` (production) connection
strings.
"""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a scoped AsyncSession."""
    async with async_session_factory() as session:
        yield session
