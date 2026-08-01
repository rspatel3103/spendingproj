"""FastAPI application entrypoint.

Creates the FastAPI app, wires up the transactions/categorize/forecast/
audit/jobs/review/metrics routers, and creates any missing DB tables on
startup (previously only scripts/generate_synthetic_data.py ever did
this -- the API itself couldn't stand up its own schema). Run locally
with:

    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.models import Base
from app.db.session import engine
from app.rag.vector_store import verify_vendor_index
from app.routers import audit, categorize, forecast, jobs, metrics, review, transactions


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Refuse to serve on an empty vendor index: retrieval failure is otherwise
    # completely silent (see EmptyIndexError in app/rag/vector_store.py).
    verify_vendor_index()
    yield


app = FastAPI(title="cashflow-agent", lifespan=lifespan)

app.include_router(transactions.router)
app.include_router(categorize.router)
app.include_router(forecast.router)
app.include_router(audit.router)
app.include_router(jobs.router)
app.include_router(review.router)
app.include_router(metrics.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
