"""Tests for recurring-pattern handling in the forecaster.

There's no separate "is this transaction recurring" classifier anywhere
in this app -- `Transaction.is_recurring` is set at data-generation time
(scripts/generate_synthetic_data.py), not detected algorithmically. What
IS algorithmic, and what these tests actually cover:

  - `_cluster_recurring_legs()`: given raw (day_of_month, amount)
    occurrences of one recurring vendor, does it correctly split a
    twice-monthly vendor (e.g. payroll on the 1st and 15th) into two
    separate legs instead of blending them into one meaningless
    mid-month average, while keeping a single-monthly vendor (e.g. rent)
    as one leg?
  - `forecast_cashflow()`: given a fixed, seeded set of transactions
    mixing recurring and one-off, does it correctly route recurring
    amounts to their scheduled occurrence dates (a visible spike in the
    balance curve) while one-off amounts smooth into a flat daily rate?
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agents.forecaster import _cluster_recurring_legs, forecast_cashflow
from app.db.models import Transaction


def test_cluster_recurring_legs_splits_twice_monthly_vendor():
    # payroll-like: two paydays a month, a few months of history
    occurrences = [
        (1, Decimal("2600.00")),
        (15, Decimal("2650.00")),
        (1, Decimal("2610.00")),
        (15, Decimal("2590.00")),
    ]

    legs = _cluster_recurring_legs(occurrences)

    assert len(legs) == 2
    days = sorted(round(leg[0]) for leg in legs)
    assert days == [1, 15]


def test_cluster_recurring_legs_keeps_single_monthly_vendor_as_one_leg():
    # rent-like: always around the 1st, +/- a day of jitter
    occurrences = [(1, Decimal("-1800.00")), (2, Decimal("-1800.00")), (1, Decimal("-1800.00"))]

    legs = _cluster_recurring_legs(occurrences)

    assert len(legs) == 1
    avg_day, avg_amount = legs[0]
    assert 1 <= avg_day <= 2
    assert avg_amount == Decimal("-1800.00")


async def test_forecast_distinguishes_recurring_spike_from_oneoff_smoothing(test_sessionmaker):
    today = date.today()

    async with test_sessionmaker() as session:
        session.add_all(
            [
                # recurring rent, posted on the 1st of last month
                Transaction(
                    account_id="acct_test",
                    posted_date=today.replace(day=1) - timedelta(days=28),
                    amount=Decimal("-1800.00"),
                    raw_description="RENT",
                    is_recurring=True,
                ),
                # a handful of one-off transactions spread across 10 days,
                # small amounts -> should smooth into a flat daily rate
                Transaction(
                    account_id="acct_test",
                    posted_date=today - timedelta(days=9),
                    amount=Decimal("-30.00"),
                    raw_description="GROCERIES",
                    is_recurring=False,
                ),
                Transaction(
                    account_id="acct_test",
                    posted_date=today,
                    amount=Decimal("-30.00"),
                    raw_description="GROCERIES",
                    is_recurring=False,
                ),
            ]
        )
        await session.commit()

    result = await forecast_cashflow(horizon=35)

    balances = [result["starting_balance"]] + [point["balance"] for point in result["balance_curve"]]
    deltas = [balances[i] - balances[i - 1] for i in range(1, len(balances))]

    # the rent leg firing should show up as a clearly larger single-day
    # drop than the flat one-off rate (~-6/day: -60 total over a 10-day span)
    biggest_single_day_drop = min(deltas)
    assert biggest_single_day_drop < Decimal("-1000")

    # almost every day should be a small, flat one-off-only change -- a
    # 35-day horizon can span two occurrences of the same day-of-month
    # depending on where "today" falls, so allow for that boundary case
    # rather than requiring exactly one big day
    small_deltas = [d for d in deltas if d > Decimal("-100")]
    big_deltas = [d for d in deltas if d <= Decimal("-100")]
    assert len(small_deltas) >= len(deltas) - 2
    assert 1 <= len(big_deltas) <= 2
