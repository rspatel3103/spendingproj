"""Jobs router.

Exposes the status/result of background jobs started by
POST /transactions/ingest and POST /categorize/run.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Job
from app.db.session import get_session
from app.schemas import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, session: AsyncSession = Depends(get_session)) -> JobStatusResponse:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        n_categorized=job.n_categorized,
        n_auto_applied=job.n_auto_applied,
        n_queued_for_review=job.n_queued_for_review,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )
