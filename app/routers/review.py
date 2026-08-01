"""Review router.

GET /review-queue lists transactions the apply layer
(`app.agents.categorizer.apply_categorization_result`) queued for human
review instead of auto-applying, because confidence was below
AUTO_APPLY_CONFIDENCE_THRESHOLD. POST /review/{id}/approve commits one
of those suggestions.

`decision_log` is append-only and grows by at least one row per
categorization forever, so "the latest decision per transaction" is
resolved in SQL with a window function rather than by loading the table
into Python. The cost of listing the queue is then proportional to the
queue (tens of rows), not to the entire decision history (unbounded).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DecisionLog, Transaction
from app.db.session import get_session
from app.rag.vector_store import index_transactions
from app.schemas import ReviewApproveResponse, ReviewQueueItem

router = APIRouter(tags=["review"])


def _latest_categorizer_decisions():
    """Subquery: one row per transaction, its most recent categorizer decision.

    `ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY id DESC)`
    numbers each transaction's decisions newest-first; callers filter to
    `rn == 1`. Supported by SQLite 3.25+ and Postgres alike, so this works
    on both backends the app targets.
    """
    return (
        select(
            DecisionLog.id.label("decision_log_id"),
            DecisionLog.transaction_id,
            DecisionLog.reasoning,
            DecisionLog.confidence_score,
            DecisionLog.suggested_category,
            DecisionLog.suggested_subcategory,
            DecisionLog.action_taken,
            func.row_number()
            .over(partition_by=DecisionLog.transaction_id, order_by=DecisionLog.id.desc())
            .label("rn"),
        )
        .where(DecisionLog.agent_name == "categorizer")
        .subquery()
    )


async def _latest_decision(session: AsyncSession, transaction_id: int) -> DecisionLog:
    """Most recent categorizer DecisionLog row for one transaction, or None."""
    return (
        await session.execute(
            select(DecisionLog)
            .where(DecisionLog.transaction_id == transaction_id, DecisionLog.agent_name == "categorizer")
            .order_by(DecisionLog.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


@router.get("/review-queue", response_model=list[ReviewQueueItem])
async def get_review_queue(session: AsyncSession = Depends(get_session)) -> list[ReviewQueueItem]:
    latest = _latest_categorizer_decisions()

    rows = (
        await session.execute(
            select(
                latest.c.decision_log_id,
                latest.c.suggested_category,
                latest.c.suggested_subcategory,
                latest.c.confidence_score,
                latest.c.reasoning,
                Transaction,
            )
            .join(Transaction, Transaction.id == latest.c.transaction_id)
            .where(latest.c.rn == 1, latest.c.action_taken == "queued_for_review")
            # Oldest first: the queue is a worklist, so the transaction that
            # has been waiting longest should be handled first.
            .order_by(Transaction.posted_date, Transaction.id)
        )
    ).all()

    return [
        ReviewQueueItem(
            transaction_id=transaction.id,
            raw_description=transaction.raw_description,
            posted_date=transaction.posted_date,
            amount=transaction.amount,
            suggested_category=suggested_category,
            suggested_subcategory=suggested_subcategory,
            confidence=confidence_score,
            reasoning=reasoning,
            decision_log_id=decision_log_id,
        )
        for (
            decision_log_id,
            suggested_category,
            suggested_subcategory,
            confidence_score,
            reasoning,
            transaction,
        ) in rows
    ]


@router.post("/review/{transaction_id}/approve", response_model=ReviewApproveResponse)
async def approve_review(transaction_id: int, session: AsyncSession = Depends(get_session)) -> ReviewApproveResponse:
    transaction = await session.get(Transaction, transaction_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")

    decision = await _latest_decision(session, transaction_id)
    if decision is None:
        raise HTTPException(status_code=404, detail=f"No categorization decision found for transaction {transaction_id}")
    if decision.action_taken != "queued_for_review":
        raise HTTPException(
            status_code=409,
            detail=f"Transaction {transaction_id} is not queued for review (latest action: {decision.action_taken})",
        )

    transaction.category = decision.suggested_category
    transaction.subcategory = decision.suggested_subcategory
    transaction.confidence_score = decision.confidence_score

    session.add(
        DecisionLog(
            transaction_id=transaction_id,
            agent_name="categorizer",
            decision_type="categorization",
            reasoning=decision.reasoning,
            confidence_score=decision.confidence_score,
            suggested_category=decision.suggested_category,
            suggested_subcategory=decision.suggested_subcategory,
            action_taken="human_approved",
        )
    )
    await session.commit()

    # A human-approved label is the highest-confidence signal there is --
    # index it so future retrievals can ground on it (same incremental path
    # the bulk categorizer uses).
    await index_transactions([transaction_id])

    return ReviewApproveResponse(
        transaction_id=transaction_id,
        status="approved",
        category=transaction.category,
        subcategory=transaction.subcategory,
    )
