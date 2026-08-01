"""Audit router.

Exposes the decision history for a transaction -- every DecisionLog row
written about it (currently just the categorizer; any future
per-transaction agent decisions land in the same table and show up here
automatically). This includes benchmark rows written by
scripts/run_eval.py, which carry agent_name="categorizer_eval" so they
stay out of the live review queue while remaining visible here -- the
audit trail should show everything that ever formed an opinion about a
transaction, clearly labelled by who formed it.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DecisionLog, Transaction
from app.db.session import get_session
from app.schemas import DecisionLogEntry, TransactionAuditResponse

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/{transaction_id}", response_model=TransactionAuditResponse)
async def get_transaction_audit(
    transaction_id: int, session: AsyncSession = Depends(get_session)
) -> TransactionAuditResponse:
    transaction = await session.get(Transaction, transaction_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")

    decisions = (
        await session.execute(
            select(DecisionLog).where(DecisionLog.transaction_id == transaction_id).order_by(DecisionLog.created_at)
        )
    ).scalars().all()

    return TransactionAuditResponse(
        transaction_id=transaction.id,
        raw_description=transaction.raw_description,
        posted_date=transaction.posted_date,
        amount=transaction.amount,
        category=transaction.category,
        subcategory=transaction.subcategory,
        decisions=[DecisionLogEntry.model_validate(d) for d in decisions],
    )
