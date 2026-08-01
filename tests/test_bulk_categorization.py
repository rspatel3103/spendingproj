"""Tests for the concurrent bulk categorizer
(app.agents.dispatcher.categorize_and_apply).

`categorize_and_apply` fans categorization out across worker threads and
then applies the results serially. The dangerous failure mode is silent:
if the parallel results ever stopped lining up with the rows they came
from, every transaction would be labeled with some *other* transaction's
category, and nothing would raise -- the counts would still be right, the
job would still report success, and the data would be quietly wrong.

`asyncio.gather` documents that it preserves input order, so this is a
regression guard rather than a suspicion. It matters because the zip()
that relies on that guarantee is easy to "optimize" into something
unordered (as_completed, a dict comprehension, a thread pool map) by
someone who doesn't know the ordering is load-bearing.

No OpenAI calls: categorize_transaction is monkeypatched with a
deterministic stand-in that echoes each row's own description back, which
is what makes a mix-up detectable.
"""

import asyncio
import time
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import select

import app.agents.dispatcher as dispatcher
from app.agents.categorizer import CategorizationResult
from app.db.models import DecisionLog, Transaction

N_ROWS = 20


@pytest.fixture
def tracer_categorizer(monkeypatch):
    """Replace the real categorizer with one that tags each result with its input.

    Also stubs out the post-batch reindex, which would otherwise make a
    real embedding API call.
    """

    def fake_categorize(row):
        time.sleep(0.02)  # long enough that workers genuinely overlap
        return CategorizationResult(
            category="Dining",
            subcategory=row.raw_description,  # the tracer
            confidence=0.95,
            reasoning=f"fixed test fixture for {row.raw_description}",
        )

    async def noop_index(ids):
        return 0

    monkeypatch.setattr(dispatcher, "categorize_transaction", fake_categorize)
    monkeypatch.setattr(dispatcher, "index_transactions", noop_index)
    return fake_categorize


async def _seed(sessionmaker, n=N_ROWS):
    async with sessionmaker() as session:
        for i in range(n):
            session.add(
                Transaction(
                    account_id="acct_test",
                    posted_date=date(2026, 7, 1),
                    amount=Decimal("-5.00"),
                    raw_description=f"ROW-{i:03d}",
                    is_recurring=False,
                )
            )
        await session.commit()


async def test_every_row_keeps_its_own_result(test_sessionmaker, tracer_categorizer):
    await _seed(test_sessionmaker)

    summary = await dispatcher.categorize_and_apply()
    assert summary["n_categorized"] == N_ROWS

    async with test_sessionmaker() as session:
        logs = (await session.execute(select(DecisionLog))).scalars().all()
        transactions = {
            t.id: t for t in (await session.execute(select(Transaction))).scalars().all()
        }

    assert len(logs) == N_ROWS
    for log in logs:
        expected = transactions[log.transaction_id].raw_description
        assert log.suggested_subcategory == expected, (
            f"transaction {log.transaction_id} ({expected!r}) was labeled with "
            f"{log.suggested_subcategory!r} -- parallel results are misaligned"
        )


async def test_transactions_receive_their_own_category(test_sessionmaker, tracer_categorizer):
    """The same alignment guarantee, checked on the rows themselves."""
    await _seed(test_sessionmaker)
    await dispatcher.categorize_and_apply()

    async with test_sessionmaker() as session:
        rows = (await session.execute(select(Transaction))).scalars().all()

    for row in rows:
        assert row.subcategory == row.raw_description


async def test_runs_concurrently(test_sessionmaker, tracer_categorizer, monkeypatch):
    """Serialized execution would still be correct, just slow -- pin the speedup."""
    monkeypatch.setattr(dispatcher, "CATEGORIZATION_CONCURRENCY", 5)
    await _seed(test_sessionmaker)

    start = time.perf_counter()
    await dispatcher.categorize_and_apply()
    elapsed = time.perf_counter() - start

    serial_estimate = N_ROWS * 0.02
    assert elapsed < serial_estimate * 0.75, (
        f"took {elapsed:.3f}s; serial would be ~{serial_estimate:.3f}s -- "
        f"work does not appear to be running in parallel"
    )


async def test_scoping_limits_work_to_named_ids(test_sessionmaker, tracer_categorizer):
    """The guardrail against a batch job sweeping up the whole table."""
    await _seed(test_sessionmaker)

    async with test_sessionmaker() as session:
        ids = [
            t.id
            for t in (await session.execute(select(Transaction).limit(3))).scalars().all()
        ]

    summary = await dispatcher.categorize_and_apply(transaction_ids=ids)

    assert summary["n_categorized"] == 3
    async with test_sessionmaker() as session:
        categorized = (
            await session.execute(select(Transaction).where(Transaction.category.is_not(None)))
        ).scalars().all()
    assert {t.id for t in categorized} == set(ids)


async def test_empty_batch_is_a_noop(test_sessionmaker, tracer_categorizer):
    summary = await dispatcher.categorize_and_apply(transaction_ids=[])
    assert summary == {"n_categorized": 0, "n_auto_applied": 0, "n_queued_for_review": 0}
