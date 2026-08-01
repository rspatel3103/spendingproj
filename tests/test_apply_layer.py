"""Tests for the guardrail apply layer
(app.agents.categorizer.apply_categorization_result).

apply_categorization_result() takes an already-computed
CategorizationResult -- it never calls the categorizer or the LLM
itself, so there's no OpenAI call to mock here at all; the fixed
high_confidence_result/low_confidence_result fixtures (tests/conftest.py)
stand in for whatever categorize_transaction() would have returned.
"""

from datetime import date
from decimal import Decimal

from sqlalchemy import select

from app.agents.categorizer import AUTO_APPLY_CONFIDENCE_THRESHOLD, apply_categorization_result
from app.db.models import DecisionLog, Transaction


def test_threshold_is_080():
    assert AUTO_APPLY_CONFIDENCE_THRESHOLD == 0.80


async def test_high_confidence_auto_applies(test_sessionmaker, high_confidence_result):
    assert high_confidence_result.confidence >= AUTO_APPLY_CONFIDENCE_THRESHOLD

    async with test_sessionmaker() as session:
        txn = Transaction(
            account_id="acct_test",
            posted_date=date(2026, 7, 1),
            amount=Decimal("-6.25"),
            raw_description="SQ *JOES COFFEE",
        )
        session.add(txn)
        await session.commit()
        await session.refresh(txn)

        action_taken = apply_categorization_result(session, txn, high_confidence_result)
        await session.commit()

        assert action_taken == "auto_applied"
        assert txn.category == high_confidence_result.category
        assert txn.subcategory == high_confidence_result.subcategory
        assert txn.confidence_score == high_confidence_result.confidence

        decision = (
            await session.execute(select(DecisionLog).where(DecisionLog.transaction_id == txn.id))
        ).scalar_one()
        assert decision.action_taken == "auto_applied"
        assert decision.reasoning == high_confidence_result.reasoning
        assert decision.suggested_category == high_confidence_result.category
        assert decision.suggested_subcategory == high_confidence_result.subcategory
        assert decision.confidence_score == high_confidence_result.confidence


async def test_low_confidence_queues_for_review_without_writing_category(test_sessionmaker, low_confidence_result):
    assert low_confidence_result.confidence < AUTO_APPLY_CONFIDENCE_THRESHOLD

    async with test_sessionmaker() as session:
        txn = Transaction(
            account_id="acct_test",
            posted_date=date(2026, 7, 1),
            amount=Decimal("-42.10"),
            raw_description="PAYPAL *MISCSVC",
        )
        session.add(txn)
        await session.commit()
        await session.refresh(txn)

        action_taken = apply_categorization_result(session, txn, low_confidence_result)
        await session.commit()

        assert action_taken == "queued_for_review"
        # the whole point of the guardrail: nothing gets written to the
        # transaction's category fields when confidence is low
        assert txn.category is None
        assert txn.subcategory is None
        assert txn.confidence_score is None

        decision = (
            await session.execute(select(DecisionLog).where(DecisionLog.transaction_id == txn.id))
        ).scalar_one()
        assert decision.action_taken == "queued_for_review"
        assert decision.reasoning == low_confidence_result.reasoning
        # the suggestion is still recorded for the review queue, even
        # though it wasn't applied
        assert decision.suggested_category == low_confidence_result.category
        assert decision.suggested_subcategory == low_confidence_result.subcategory


async def test_confidence_exactly_at_threshold_auto_applies(test_sessionmaker):
    from app.agents.categorizer import CategorizationResult

    boundary_result = CategorizationResult(
        category="Utilities",
        subcategory="internet",
        confidence=AUTO_APPLY_CONFIDENCE_THRESHOLD,
        reasoning="Fixed test fixture: confidence exactly at the threshold.",
    )

    async with test_sessionmaker() as session:
        txn = Transaction(
            account_id="acct_test",
            posted_date=date(2026, 7, 1),
            amount=Decimal("-85.00"),
            raw_description="COMCAST XFINITY",
        )
        session.add(txn)
        await session.commit()
        await session.refresh(txn)

        action_taken = apply_categorization_result(session, txn, boundary_result)

        assert action_taken == "auto_applied"
        assert txn.category == "Utilities"
