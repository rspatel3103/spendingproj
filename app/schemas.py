"""Pydantic request/response models.

Holds the API-facing schemas shared across routers. Kept separate from
`db/models.py` (the SQLAlchemy ORM models) so the API contract can evolve
independently of the storage layer.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel


class DecisionLogEntry(BaseModel):
    id: int
    agent_name: str
    decision_type: str
    reasoning: str
    confidence_score: Optional[float]
    suggested_category: Optional[str]
    suggested_subcategory: Optional[str]
    action_taken: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TransactionAuditResponse(BaseModel):
    transaction_id: int
    raw_description: str
    posted_date: date
    amount: Decimal
    category: Optional[str]
    subcategory: Optional[str]
    decisions: list[DecisionLogEntry]


class TransactionIngestItem(BaseModel):
    account_id: str
    posted_date: date
    amount: Decimal
    raw_description: str
    is_recurring: bool = False


class TransactionIngestBatch(BaseModel):
    transactions: list[TransactionIngestItem]


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str


class IngestResponse(JobCreatedResponse):
    n_transactions_ingested: int


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    n_categorized: Optional[int]
    n_auto_applied: Optional[int]
    n_queued_for_review: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}


class BalancePoint(BaseModel):
    date: date
    balance: Decimal


class ForecastResponse(BaseModel):
    horizon: int
    starting_balance: Decimal
    lowest_point: Decimal
    lowest_point_date: date
    ending_balance: Decimal
    explanation: str
    balance_curve: list[BalancePoint]


class ReviewQueueItem(BaseModel):
    transaction_id: int
    raw_description: str
    posted_date: date
    amount: Decimal
    suggested_category: Optional[str]
    suggested_subcategory: Optional[str]
    confidence: Optional[float]
    reasoning: str
    decision_log_id: int


class ReviewApproveResponse(BaseModel):
    transaction_id: int
    status: str
    category: Optional[str]
    subcategory: Optional[str]


class MetricsResponse(BaseModel):
    n_categorization_calls: int
    avg_categorization_latency_ms: Optional[float]
    avg_confidence: Optional[float]
    pct_auto_applied: Optional[float]
    pct_queued_for_review: Optional[float]
    n_forecast_calls: int
    avg_forecast_latency_ms: Optional[float]
