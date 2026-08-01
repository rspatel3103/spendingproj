"""Forecaster skill.

Projects a daily cash-flow balance curve from historical transactions,
using simple time-series arithmetic (no LLM call, so it stays fast/free
enough to run synchronously from a request handler): recurring
transactions (`Transaction.is_recurring`, already flagged at data-gen
time -- "recurring detection" is just a column read, nothing to cache)
get projected forward at their historical day-of-month and average
amount; everything else contributes a flat average daily rate.

Self-contained and versioned (`SKILL_VERSION` below) -- importable and
reusable by any other endpoint or agent without modification. No hidden
global state: every function is either a pure computation over its
arguments (`_add_months`, `_cluster_recurring_legs`,
`_occurrence_dates_within`) or explicit about the one side effect it has
(`forecast_cashflow`'s DB reads/writes, documented below) -- nothing is
cached or memoized across calls.

Public interface
-----------------
`forecast_cashflow(horizon: int) -> dict`
    Inputs: `horizon`, a number of days to project forward.
    Outputs: a dict with `horizon`, `starting_balance`, `lowest_point`
    (+ `lowest_point_date`), `ending_balance`, a plain-language
    `explanation` string, and the full daily `balance_curve` (list of
    `{"date", "balance"}`).
    Side effects: reads every `Transaction` row from the database (no
    filtering by caller -- it always projects from the full history);
    emits one structured JSON log line (`app/observability.py`); writes
    one `AgentMetric` row recording its own latency. Does NOT call any
    LLM, does NOT write to `Transaction` or any other business table,
    and does NOT cache or memoize anything between calls -- every call
    re-reads and re-computes from scratch.

Does NOT do
-----------
- Does not detect which transactions are recurring -- that's decided
  once, upstream, at data-generation time (`Transaction.is_recurring`);
  this module only clusters and schedules already-flagged occurrences.
- Does not reason about anomalies or one-off spikes with an LLM (a
  possible future extension, not implemented here).
- Does not accept a starting balance or account scope -- it always
  sums every transaction in the table.

See SKILLS.md at the project root for a narrative summary of this skill.
"""

import calendar
import time
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select

from app.db.models import AgentMetric, Transaction
from app.db.session import async_session_factory
from app.observability import get_logger, log_event

logger = get_logger("cashflow_agent.forecaster")

SKILL_VERSION = "1.0.0"

# Occurrences of a recurring description more than this many days apart
# (by day-of-month) are treated as separate "legs" -- e.g. payroll hits
# on the 1st and the 15th, which should project as two monthly events,
# not one blended, meaningless mid-month average.
_LEG_CLUSTER_GAP_DAYS = 10

# Average days in a month, for turning a daily discretionary rate into a
# human-readable monthly figure in the explanation text.
_AVG_DAYS_PER_MONTH = 30.44


def _add_months(d: date, n: int) -> date:
    month_index = d.month - 1 + n
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _cluster_recurring_legs(occurrences: list[tuple[int, Decimal]]) -> list[tuple[float, Decimal]]:
    """Cluster (day_of_month, amount) occurrences of one recurring vendor into legs.

    Returns (avg_day, avg_amount) per leg, e.g. one leg for rent (day~1),
    two legs for a twice-monthly payroll (day~1 and day~15).
    """
    sorted_occurrences = sorted(occurrences, key=lambda o: o[0])
    clusters: list[list[tuple[int, Decimal]]] = []
    for occurrence in sorted_occurrences:
        if clusters and occurrence[0] - clusters[-1][-1][0] <= _LEG_CLUSTER_GAP_DAYS:
            clusters[-1].append(occurrence)
        else:
            clusters.append([occurrence])

    legs = []
    for cluster in clusters:
        avg_day = sum(o[0] for o in cluster) / len(cluster)
        avg_amount = sum(o[1] for o in cluster) / len(cluster)
        legs.append((avg_day, avg_amount))
    return legs


def _occurrence_dates_within(after: date, avg_day: float, through: date) -> list[date]:
    """Every monthly occurrence date for a leg strictly after `after`, up to and including `through`."""
    day = min(round(avg_day), 28)
    candidate = date(after.year, after.month, min(day, calendar.monthrange(after.year, after.month)[1]))
    if candidate <= after:
        candidate = _add_months(candidate, 1)

    dates = []
    while candidate <= through:
        dates.append(candidate)
        candidate = _add_months(candidate, 1)
    return dates


async def forecast_cashflow(horizon: int) -> dict:
    """Project a daily balance curve `horizon` days into the future.

    Returns a dict with starting_balance, lowest_point (+ its date),
    ending_balance, a plain-language explanation, and the full daily
    balance_curve.
    """
    start = time.perf_counter()

    async with async_session_factory() as session:
        rows = (await session.execute(select(Transaction))).scalars().all()

    today = date.today()
    starting_balance = sum((row.amount for row in rows), start=Decimal("0"))

    recurring_by_description: dict = defaultdict(list)
    oneoff_total = Decimal("0")
    oneoff_dates: list[date] = []

    for row in rows:
        if row.is_recurring:
            recurring_by_description[row.raw_description].append((row.posted_date.day, row.amount))
        else:
            oneoff_total += row.amount
            oneoff_dates.append(row.posted_date)

    recurring_legs: list[tuple[float, Decimal]] = []
    for occurrences in recurring_by_description.values():
        recurring_legs.extend(_cluster_recurring_legs(occurrences))

    span_days = (max(oneoff_dates) - min(oneoff_dates)).days + 1 if oneoff_dates else 1
    avg_daily_oneoff = oneoff_total / span_days

    end_date = today + timedelta(days=horizon)
    recurring_amount_by_date: dict = defaultdict(lambda: Decimal("0"))
    for avg_day, avg_amount in recurring_legs:
        for occurrence_date in _occurrence_dates_within(today, avg_day, end_date):
            recurring_amount_by_date[occurrence_date] += avg_amount

    balance = starting_balance
    curve = []
    for offset in range(1, horizon + 1):
        current_date = today + timedelta(days=offset)
        balance += avg_daily_oneoff + recurring_amount_by_date.get(current_date, Decimal("0"))
        curve.append({"date": current_date, "balance": balance.quantize(Decimal("0.01"))})

    lowest = min(curve, key=lambda point: point["balance"])
    ending_balance = curve[-1]["balance"]

    monthly_recurring_income = sum((amt for _, amt in recurring_legs if amt > 0), start=Decimal("0"))
    monthly_recurring_expense = sum((amt for _, amt in recurring_legs if amt < 0), start=Decimal("0"))
    monthly_discretionary = avg_daily_oneoff * Decimal(str(_AVG_DAYS_PER_MONTH))

    explanation = (
        f"Based on ~${monthly_recurring_income:,.0f}/mo in recurring income, "
        f"~${abs(monthly_recurring_expense):,.0f}/mo in recurring bills, and "
        f"~${abs(monthly_discretionary):,.0f}/mo in average discretionary spending, your balance is "
        f"projected to dip to its lowest point of ${lowest['balance']:,.2f} around {lowest['date'].isoformat()}, "
        f"ending near ${ending_balance:,.2f} by {end_date.isoformat()} ({horizon} days out)."
    )

    latency_ms = (time.perf_counter() - start) * 1000

    log_event(logger, "forecast", agent="forecaster", horizon=horizon, latency_ms=round(latency_ms, 1))

    async with async_session_factory() as metrics_session:
        metrics_session.add(AgentMetric(agent_name="forecaster", event="forecast", latency_ms=latency_ms))
        await metrics_session.commit()

    return {
        "horizon": horizon,
        "starting_balance": starting_balance.quantize(Decimal("0.01")),
        "lowest_point": lowest["balance"],
        "lowest_point_date": lowest["date"],
        "ending_balance": ending_balance,
        "explanation": explanation,
        "balance_curve": curve,
    }
