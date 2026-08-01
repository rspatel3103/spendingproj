"""Intent dispatcher.

Routes a request to one of three handlers -- `categorize_new`,
`forecast`, `ingest` -- and runs it. A structured intent
(`{"action": "categorize_new"}`) selects the branch directly; free text
("how much will I have left at the end of next month?") goes through a
small GPT-4o structured-output call first to classify it.

Deliberately named a dispatcher rather than a planner or an agent,
because that is what it is: a three-branch switch with an optional LLM
classifier in front. It does not use tools, does not loop, does not
maintain state across calls, and cannot compose actions (there is no way
to express "ingest, then forecast"). Calling it a planner would invite a
comparison to agent frameworks that it would lose, and the honest
description is not a weaker one -- for three mutually exclusive actions,
a switch is the right amount of machinery.

Also home to `categorize_and_apply()`, the bulk categorization entry
point behind both POST /transactions/ingest and POST /categorize/run.
That function, not the routing, is where the real work happens.
"""

import asyncio
import time
from functools import lru_cache
from typing import Literal, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.agents.categorizer import (
    CategorizationResult,
    apply_categorization_result,
    categorize_transaction,
)
from app.agents.forecaster import forecast_cashflow
from app.config import settings
from app.db.models import AgentMetric, Transaction
from app.db.session import async_session_factory
from app.observability import get_logger, log_event
from app.rag.vector_store import build_transaction_index, build_vendor_index, index_transactions

logger = get_logger("cashflow_agent.dispatcher")

DEFAULT_FORECAST_HORIZON = 30

# How many transactions to categorize in parallel. Each one is I/O-bound
# (two embedding calls + one GPT-4o call), so fanning out is a real win --
# but the binding constraint is the OpenAI org's tokens-per-minute limit,
# not local resources. Tuned via settings so it can be raised on a higher
# tier without a code change; see Settings.categorization_concurrency.
CATEGORIZATION_CONCURRENCY = settings.categorization_concurrency


class IntentDecision(BaseModel):
    action: Literal["categorize_new", "forecast", "ingest"]
    horizon: Optional[int] = Field(
        default=None, description="Forecast horizon in days (30 or 90). Only meaningful when action is 'forecast'."
    )
    reasoning: str = Field(description="One sentence explaining why this action was chosen.")


@lru_cache
def get_llm() -> ChatOpenAI:
    """Memoized ChatOpenAI client -- one instance per process, no `global` mutation."""
    return ChatOpenAI(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key)


def classify_intent(text: str) -> IntentDecision:
    """Classify free-text into one of the three dispatchable actions."""
    start = time.perf_counter()

    system_prompt = (
        "You are a routing assistant for a personal finance app. Given a user's request, "
        "decide which single action to take:\n"
        "- 'categorize_new': the user wants uncategorized transactions processed/categorized.\n"
        "- 'forecast': the user is asking about future spending, balance, or cash flow. "
        "Extract a horizon in days -- default to 30 for near-term questions ('next month'), "
        "use 90 only if they clearly mean a longer-term question ('next quarter', 'next few months').\n"
        "- 'ingest': the user mentions new data landing, refreshing, or re-indexing the "
        "vendor/transaction knowledge base.\n"
        "Give one sentence of reasoning for your choice."
    )
    structured_llm = get_llm().with_structured_output(IntentDecision, include_raw=True)
    raw_result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=text)])
    decision = raw_result["parsed"]
    if decision is None:
        raise raw_result["parsing_error"] or RuntimeError("classify_intent: LLM output failed to parse")

    usage = raw_result["raw"].usage_metadata or {}
    latency_ms = (time.perf_counter() - start) * 1000

    log_event(
        logger,
        "classify_intent",
        agent="dispatcher",
        latency_ms=round(latency_ms, 1),
        action=decision.action,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )

    return decision


async def _categorize_with_latency(
    row: Transaction, semaphore: asyncio.Semaphore
) -> tuple[CategorizationResult, float]:
    """Categorize one row on a worker thread, returning its own latency.

    Timing lives inside the semaphore so `latency_ms` measures the actual
    call, not how long the row waited its turn -- otherwise every metric
    would inflate with batch size and stop meaning anything.
    """
    async with semaphore:
        start = time.perf_counter()
        result = await asyncio.to_thread(categorize_transaction, row)
        return result, (time.perf_counter() - start) * 1000


async def categorize_and_apply(transaction_ids: Optional[list[int]] = None) -> dict:
    """Categorize and apply-or-queue every uncategorized transaction.

    Scoped to `transaction_ids` when given -- callers that just ingested
    a specific batch should always pass this, so a query that would
    otherwise be "every uncategorized transaction in the whole table"
    can't accidentally sweep up unrelated historical rows.

    Runs in two phases. Categorization is I/O-bound and touches no
    database state, so it fans out across CATEGORIZATION_CONCURRENCY
    workers. Applying results is then strictly serial, because
    AsyncSession is not safe for concurrent use -- writing to one session
    from parallel tasks corrupts its internal state.
    """
    async with async_session_factory() as session:
        stmt = select(Transaction).where(Transaction.category.is_(None))
        if transaction_ids is not None:
            stmt = stmt.where(Transaction.id.in_(transaction_ids))
        rows = (await session.execute(stmt)).scalars().all()

        # Phase 1 -- parallel, no DB access.
        semaphore = asyncio.Semaphore(CATEGORIZATION_CONCURRENCY)
        outcomes = await asyncio.gather(
            *(_categorize_with_latency(row, semaphore) for row in rows)
        )

        # Phase 2 -- serial, single session.
        n_auto_applied = 0
        n_queued_for_review = 0
        auto_applied_ids: list[int] = []
        for row, (result, latency_ms) in zip(rows, outcomes):
            action_taken = apply_categorization_result(session, row, result)

            session.add(
                AgentMetric(
                    agent_name="categorizer",
                    event="categorization",
                    transaction_id=row.id,
                    latency_ms=latency_ms,
                    confidence=result.confidence,
                    action_taken=action_taken,
                )
            )

            if action_taken == "auto_applied":
                n_auto_applied += 1
                auto_applied_ids.append(row.id)
            else:
                n_queued_for_review += 1

        await session.commit()

    # Feed the newly-labeled rows back into the retrieval index, so the
    # "similar past transactions" leg actually reflects what's been
    # categorized. Incremental (cost scales with the batch, not the table) and
    # after the commit, so nothing is indexed that didn't persist. Only
    # auto-applied ids: queued rows have no confirmed category yet, and get
    # indexed when a human approves them instead.
    await index_transactions(auto_applied_ids)

    return {
        "n_categorized": len(rows),
        "n_auto_applied": n_auto_applied,
        "n_queued_for_review": n_queued_for_review,
    }


async def _run_categorize_new() -> dict:
    return await categorize_and_apply()


async def _run_forecast(horizon: Optional[int], session_id: str) -> dict:
    resolved_horizon = horizon or DEFAULT_FORECAST_HORIZON
    return await forecast_cashflow(resolved_horizon)


async def _run_ingest() -> dict:
    vendor_count = await build_vendor_index()
    transaction_count = await build_transaction_index()
    return {"vendor_vectors_indexed": vendor_count, "transaction_vectors_indexed": transaction_count}


async def route(intent: Union[str, dict], session_id: str = "default") -> dict:
    """Route a structured or free-text intent to the matching handler.

    The routing decision -- including the LLM's reasoning when free text
    was classified -- goes to the structured log rather than to a
    database table. An earlier version persisted it to `AgentMemory`,
    which nothing ever read; and because both production entry points
    pass structured intents, every row it wrote recorded the same
    constant string. A log line is reachable by any aggregator and costs
    nothing to keep.
    """
    start = time.perf_counter()

    if isinstance(intent, dict):
        action = intent["action"]
        horizon = intent.get("horizon")
        reasoning = "Structured intent provided directly."
    else:
        decision = classify_intent(intent)
        action = decision.action
        horizon = decision.horizon
        reasoning = decision.reasoning

    if action == "categorize_new":
        result = await _run_categorize_new()
    elif action == "forecast":
        result = await _run_forecast(horizon, session_id)
    else:
        result = await _run_ingest()

    latency_ms = (time.perf_counter() - start) * 1000
    log_event(
        logger,
        "route",
        agent="dispatcher",
        session_id=session_id,
        action=action,
        horizon=horizon,
        reasoning=reasoning,
        latency_ms=round(latency_ms, 1),
    )

    return {"action": action, "horizon": horizon, "reasoning": reasoning, "result": result}
