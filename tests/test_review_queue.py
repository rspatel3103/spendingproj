"""Tests for GET /review-queue.

The endpoint answers "which transactions are currently waiting for a
human?", which means resolving the *latest* categorizer decision per
transaction out of an append-only log. Two things about that are easy to
get wrong and silent when they are:

  - Only the newest decision counts. An approved transaction must leave
    the queue even though its original `queued_for_review` row is still
    in the table forever.
  - Only PRODUCTION decisions count. scripts/run_eval.py writes decision
    rows against real transaction ids; when those shared an agent_name
    with live decisions they became the "latest" one and silently pulled
    transactions out of the operator's worklist. That is what
    EVAL_AGENT_NAME exists to prevent.

Neither failure raises. The queue just quietly returns the wrong set.
"""

from datetime import date
from decimal import Decimal

import httpx
import pytest
from httpx import ASGITransport

from app.agents.categorizer import (
    CATEGORIZER_AGENT_NAME,
    EVAL_AGENT_NAME,
    CategorizationResult,
    build_decision_log,
)
from app.db.models import DecisionLog, Transaction
from app.main import app


@pytest.fixture
def client():
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _add_transaction(sessionmaker, description: str) -> int:
    async with sessionmaker() as session:
        txn = Transaction(
            account_id="acct_test",
            posted_date=date(2026, 7, 1),
            amount=Decimal("-20.00"),
            raw_description=description,
        )
        session.add(txn)
        await session.commit()
        await session.refresh(txn)
        return txn.id


def _result(category="Shopping", confidence=0.3):
    return CategorizationResult(
        category=category,
        subcategory="unknown",
        confidence=confidence,
        reasoning="fixed test fixture",
    )


async def test_queued_transaction_appears(test_sessionmaker, client):
    txn_id = await _add_transaction(test_sessionmaker, "SQ *MERCHANT 88213")
    async with test_sessionmaker() as session:
        session.add(build_decision_log(txn_id, _result(), action_taken="queued_for_review"))
        await session.commit()

    async with client as c:
        items = (await c.get("/review-queue")).json()

    assert [i["transaction_id"] for i in items] == [txn_id]
    assert items[0]["raw_description"] == "SQ *MERCHANT 88213"
    assert items[0]["suggested_category"] == "Shopping"
    assert items[0]["reasoning"] == "fixed test fixture"


async def test_auto_applied_transaction_does_not_appear(test_sessionmaker, client):
    txn_id = await _add_transaction(test_sessionmaker, "WM SUPERCENTER #4521")
    async with test_sessionmaker() as session:
        session.add(
            build_decision_log(txn_id, _result(confidence=0.9), action_taken="auto_applied")
        )
        await session.commit()

    async with client as c:
        assert (await c.get("/review-queue")).json() == []


async def test_only_the_latest_decision_counts(test_sessionmaker, client):
    """An approved transaction leaves the queue, despite its older queued row."""
    txn_id = await _add_transaction(test_sessionmaker, "PAYPAL *MISCSVC")
    async with test_sessionmaker() as session:
        session.add(build_decision_log(txn_id, _result(), action_taken="queued_for_review"))
        await session.commit()
    async with test_sessionmaker() as session:
        session.add(build_decision_log(txn_id, _result(), action_taken="human_approved"))
        await session.commit()

    async with client as c:
        assert (await c.get("/review-queue")).json() == []


async def test_eval_rows_do_not_hide_queued_transactions(test_sessionmaker, client):
    """Regression: an eval run must not empty the operator's review queue.

    run_eval.py writes decision rows against real transaction ids. When
    they shared agent_name with production, an eval row written *after* a
    queued one became the latest decision and the transaction vanished
    from the queue.
    """
    txn_id = await _add_transaction(test_sessionmaker, "POS DEBIT 4471")
    async with test_sessionmaker() as session:
        session.add(build_decision_log(txn_id, _result(), action_taken="queued_for_review"))
        await session.commit()

    # eval runs afterwards, writing a newer row for the same transaction
    async with test_sessionmaker() as session:
        session.add(
            build_decision_log(
                txn_id, _result(), action_taken="pending", agent_name=EVAL_AGENT_NAME
            )
        )
        await session.commit()

    async with client as c:
        items = (await c.get("/review-queue")).json()

    assert [i["transaction_id"] for i in items] == [txn_id], (
        "an eval decision row hid a genuinely-queued transaction from the review queue"
    )


async def test_eval_rows_use_a_distinct_agent_name():
    """Guards the constant itself -- the whole separation rests on it."""
    assert EVAL_AGENT_NAME != CATEGORIZER_AGENT_NAME


async def test_queue_is_ordered_oldest_first(test_sessionmaker, client):
    """It's a worklist: whatever has waited longest should be handled first."""
    async with test_sessionmaker() as session:
        for day in (14, 3, 28):
            session.add(
                Transaction(
                    account_id="acct_test",
                    posted_date=date(2026, 7, day),
                    amount=Decimal("-9.00"),
                    raw_description=f"SQ *VENDOR {day}",
                )
            )
        await session.commit()

    async with test_sessionmaker() as session:
        from sqlalchemy import select

        rows = (await session.execute(select(Transaction))).scalars().all()
        for row in rows:
            session.add(build_decision_log(row.id, _result(), action_taken="queued_for_review"))
        await session.commit()

    async with client as c:
        items = (await c.get("/review-queue")).json()

    assert [i["posted_date"] for i in items] == ["2026-07-03", "2026-07-14", "2026-07-28"]


async def test_empty_queue_returns_empty_list(test_sessionmaker, client):
    async with client as c:
        response = await c.get("/review-queue")
    assert response.status_code == 200
    assert response.json() == []
