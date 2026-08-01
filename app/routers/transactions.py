"""Transactions router.

POST /transactions/ingest accepts either a JSON batch or a CSV file
upload of new transactions, writes them (uncategorized), and kicks off
categorization for just that batch as a background job -- scoped to the
ids it just created, never the whole table (see
app.agents.dispatcher.categorize_and_apply).
"""

import csv
import io
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.dispatcher import categorize_and_apply
from app.db.models import Job, Transaction
from app.db.session import get_session
from app.jobs import run_categorization_job
from app.schemas import IngestResponse, TransactionIngestBatch, TransactionIngestItem

router = APIRouter(prefix="/transactions", tags=["transactions"])


def _parse_csv(raw_bytes: bytes) -> list[TransactionIngestItem]:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode CSV as UTF-8: {exc}")

    reader = csv.DictReader(io.StringIO(text))
    items = []
    for i, row in enumerate(reader, start=1):
        try:
            items.append(
                TransactionIngestItem(
                    account_id=row["account_id"],
                    posted_date=row["posted_date"],
                    amount=row["amount"],
                    raw_description=row["raw_description"],
                    is_recurring=str(row.get("is_recurring", "")).strip().lower() in ("1", "true", "yes"),
                )
            )
        except (KeyError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=f"Row {i}: invalid transaction data ({exc})")

    if not items:
        raise HTTPException(status_code=400, detail="CSV upload contained no transaction rows.")
    return items


async def _parse_json(request: Request) -> list[TransactionIngestItem]:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")

    try:
        batch = TransactionIngestBatch.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    if not batch.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")
    return batch.transactions


@router.post("/ingest", response_model=IngestResponse)
async def ingest_transactions(
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="No file provided (expected form field 'file').")
        items = _parse_csv(await upload.read())
    else:
        items = await _parse_json(request)

    new_rows = [
        Transaction(
            account_id=item.account_id,
            posted_date=item.posted_date,
            amount=item.amount,
            raw_description=item.raw_description,
            is_recurring=item.is_recurring,
        )
        for item in items
    ]
    session.add_all(new_rows)
    await session.commit()
    new_ids = [row.id for row in new_rows]

    job_id = str(uuid.uuid4())
    session.add(Job(id=job_id, job_type="ingest_categorize", status="pending"))
    await session.commit()

    background_tasks.add_task(run_categorization_job, job_id, lambda: categorize_and_apply(transaction_ids=new_ids))

    return IngestResponse(job_id=job_id, status="pending", n_transactions_ingested=len(new_rows))
