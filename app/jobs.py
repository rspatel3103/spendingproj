"""Background job execution helper.

Runs categorization work (kicked off from a router via FastAPI
BackgroundTasks) and tracks its progress/result in the `Job` table, so
`GET /jobs/{job_id}` has something to report. A plain DB-backed table
(no Celery/Redis) is enough at this scope, and keeps job state
consistent with everything else in this app being persisted for
auditability.
"""

from datetime import datetime
from typing import Awaitable, Callable

from app.db.models import Job
from app.db.session import async_session_factory


async def run_categorization_job(job_id: str, work: Callable[[], Awaitable[dict]]) -> None:
    """Run `work()` -- expected to return {n_categorized, n_auto_applied,
    n_queued_for_review} -- updating the Job row's status/summary as it
    goes. Any exception marks the job "failed" with the error recorded,
    rather than leaving it stuck at "running" or crashing silently.
    """
    async with async_session_factory() as session:
        job = await session.get(Job, job_id)
        job.status = "running"
        await session.commit()

    try:
        summary = await work()
    except Exception as exc:
        async with async_session_factory() as session:
            job = await session.get(Job, job_id)
            job.status = "failed"
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()
            await session.commit()
        return

    async with async_session_factory() as session:
        job = await session.get(Job, job_id)
        job.status = "complete"
        job.n_categorized = summary.get("n_categorized")
        job.n_auto_applied = summary.get("n_auto_applied")
        job.n_queued_for_review = summary.get("n_queued_for_review")
        job.completed_at = datetime.utcnow()
        await session.commit()
