"""Metrics router.

GET /metrics aggregates the AgentMetric table (app/db/models.py) --
written by app.agents.dispatcher.categorize_and_apply() and
app.agents.forecaster.forecast_cashflow(), the two real production
entry points -- into the handful of numbers worth pointing to: average
categorization latency/confidence, auto-applied vs. queued-for-review
split, and average forecast latency. Not Prometheus-grade, just real
numbers from real calls.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentMetric
from app.db.session import get_session
from app.schemas import MetricsResponse

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(session: AsyncSession = Depends(get_session)) -> MetricsResponse:
    categorizer_stats = (
        await session.execute(
            select(func.count(AgentMetric.id), func.avg(AgentMetric.latency_ms), func.avg(AgentMetric.confidence))
            .where(AgentMetric.agent_name == "categorizer")
        )
    ).one()
    n_categorization_calls, avg_categorization_latency_ms, avg_confidence = categorizer_stats

    action_counts = dict(
        (
            await session.execute(
                select(AgentMetric.action_taken, func.count(AgentMetric.id))
                .where(AgentMetric.agent_name == "categorizer")
                .group_by(AgentMetric.action_taken)
            )
        ).all()
    )
    n_auto_applied = action_counts.get("auto_applied", 0)
    n_queued_for_review = action_counts.get("queued_for_review", 0)
    pct_auto_applied = (100.0 * n_auto_applied / n_categorization_calls) if n_categorization_calls else None
    pct_queued_for_review = (100.0 * n_queued_for_review / n_categorization_calls) if n_categorization_calls else None

    forecaster_stats = (
        await session.execute(
            select(func.count(AgentMetric.id), func.avg(AgentMetric.latency_ms))
            .where(AgentMetric.agent_name == "forecaster")
        )
    ).one()
    n_forecast_calls, avg_forecast_latency_ms = forecaster_stats

    return MetricsResponse(
        n_categorization_calls=n_categorization_calls,
        avg_categorization_latency_ms=round(avg_categorization_latency_ms, 1) if avg_categorization_latency_ms is not None else None,
        avg_confidence=round(avg_confidence, 4) if avg_confidence is not None else None,
        pct_auto_applied=round(pct_auto_applied, 1) if pct_auto_applied is not None else None,
        pct_queued_for_review=round(pct_queued_for_review, 1) if pct_queued_for_review is not None else None,
        n_forecast_calls=n_forecast_calls,
        avg_forecast_latency_ms=round(avg_forecast_latency_ms, 1) if avg_forecast_latency_ms is not None else None,
    )
