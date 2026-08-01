"""Categorize router.

POST /categorize/run triggers categorization of every uncategorized
transaction, via the dispatcher (`app.agents.dispatcher.route`), as a
background job -- same async job pattern as
POST /transactions/ingest. Unlike ingest, this is intentionally unscoped
(the dispatcher's categorize_new action always means "everything still
uncategorized"), so it's a real, potentially expensive operation against
a large table -- see app.agents.dispatcher.categorize_and_apply for the
scoping mechanism ingest uses instead.
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.dispatcher import route
from app.db.models import Job
from app.db.session import get_session
from app.jobs import run_categorization_job
from app.schemas import JobCreatedResponse

router = APIRouter(prefix="/categorize", tags=["categorize"])


@router.post("/run", response_model=JobCreatedResponse)
async def run_categorization(
    background_tasks: BackgroundTasks, session: AsyncSession = Depends(get_session)
) -> JobCreatedResponse:
    job_id = str(uuid.uuid4())
    session.add(Job(id=job_id, job_type="categorize_run", status="pending"))
    await session.commit()

    async def _work() -> dict:
        outcome = await route({"action": "categorize_new"}, session_id=job_id)
        return outcome["result"]

    background_tasks.add_task(run_categorization_job, job_id, _work)

    return JobCreatedResponse(job_id=job_id, status="pending")
