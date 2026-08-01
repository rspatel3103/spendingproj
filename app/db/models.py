"""SQLAlchemy ORM models.

Defines the persisted tables for this project: transactions (raw imported
spending records), the vendor knowledge base used for RAG-assisted
categorization, the agents' running memory, a log of agent decisions and
their justification/confidence, and eval results for measuring
categorization accuracy. Models are declared against the shared `Base` and
consumed by `db/session.py` for table creation and by the routers/agents
for queries.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import ForeignKey, Index, Numeric, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[str]
    posted_date: Mapped[date]
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    raw_description: Mapped[str]
    vendor_name: Mapped[Optional[str]]
    category: Mapped[Optional[str]]
    subcategory: Mapped[Optional[str]]
    confidence_score: Mapped[Optional[float]]
    is_recurring: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class VendorKB(Base):
    __tablename__ = "vendor_kb"

    id: Mapped[int] = mapped_column(primary_key=True)
    vendor_name: Mapped[str]
    canonical_category: Mapped[str]
    canonical_subcategory: Mapped[Optional[str]]
    description: Mapped[str]
    source: Mapped[str]


# Removed: AgentMemory ("session_id / agent_name / input_summary /
# output_summary"). It was written only by the dispatcher's route() and
# read by nothing. Because both production entry points pass structured
# intents rather than free text, every row it wrote recorded the same
# constant reasoning string, so it was not an underused table -- it was a
# table logging a fixed value from a path almost nothing takes. The
# routing decision now goes to the structured log in
# app/agents/dispatcher.py, where a log aggregator can reach it.
# Per-transaction audit lives in DecisionLog below; per-call performance
# lives in AgentMetric.


class DecisionLog(Base):
    __tablename__ = "decision_log"

    # Append-only: at least one row per categorization, forever, and this is
    # the fastest-growing table in the schema. The index supports the
    # "latest decision per transaction" window function behind
    # GET /review-queue (app/routers/review.py) -- its column order matches
    # that query's filter, partition, and sort respectively.
    __table_args__ = (
        Index("ix_decision_log_agent_txn_id", "agent_name", "transaction_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    transaction_id: Mapped[Optional[int]] = mapped_column(ForeignKey("transactions.id"))
    agent_name: Mapped[str]
    decision_type: Mapped[str]
    reasoning: Mapped[str]
    confidence_score: Mapped[Optional[float]]
    suggested_category: Mapped[Optional[str]]
    suggested_subcategory: Mapped[Optional[str]]
    action_taken: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(primary_key=True)
    job_type: Mapped[str]
    status: Mapped[str]
    n_categorized: Mapped[Optional[int]]
    n_auto_applied: Mapped[Optional[int]]
    n_queued_for_review: Mapped[Optional[int]]
    error_message: Mapped[Optional[str]]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[Optional[datetime]]


class AgentMetric(Base):
    __tablename__ = "agent_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_name: Mapped[str]
    event: Mapped[str]
    transaction_id: Mapped[Optional[int]] = mapped_column(ForeignKey("transactions.id"))
    session_id: Mapped[Optional[str]]
    latency_ms: Mapped[float]
    confidence: Mapped[Optional[float]]
    action_taken: Mapped[Optional[str]]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str]
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    predicted_category: Mapped[str]
    true_category: Mapped[str]
    correct: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
