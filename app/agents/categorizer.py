"""Categorizer skill.

Classifies a single transaction into a spending category using GPT-4o,
grounded in retrieval-augmented context from the vendor knowledge base and
already-categorized past transactions (`app/rag/vector_store.py`).

Self-contained and versioned (`SKILL_VERSION` below) -- importable and
reusable by any other endpoint or agent without modification. No hidden
global state: the one cached resource (the `ChatOpenAI` client) is
memoized via `functools.lru_cache`, not a mutable module-level variable
reassigned through a `global` statement.

Public interface
-----------------
`categorize_transaction(transaction, allowed_categories=DEFAULT_CATEGORIES) -> CategorizationResult`
    Inputs: a `Transaction` (needs `.raw_description`, `.amount`; `.id`
    is used only for logging and may be `None` for transient/unpersisted
    objects), and an optional closed list of allowed top-level
    categories.
    Outputs: a `CategorizationResult` (category, subcategory, confidence
    0-1, one-sentence reasoning).
    Side effects: emits one structured JSON log line
    (`app/observability.py`); makes network calls (RAG retrieval via
    `app/rag/vector_store.py`, then one GPT-4o structured-output call).
    Does NOT write to the database, does NOT mutate `transaction`, and
    does NOT decide auto-apply vs. review -- that's
    `apply_categorization_result()`'s job, not this function's.
    Deliberately synchronous and BLOCKING (real network calls, no
    `await` inside) -- callers on the FastAPI event loop must invoke it
    via `asyncio.to_thread(categorize_transaction, ...)`, not directly,
    or it will stall the event loop for the call's full duration. See
    `app.agents.dispatcher.categorize_and_apply` for the pattern.

`build_decision_log(transaction_id, result, action_taken="pending") -> DecisionLog`
    Inputs: a transaction id, a `CategorizationResult`, an
    `action_taken` string. Outputs: an unpersisted `DecisionLog` row --
    the caller must `session.add()`/commit it. Side effects: none (pure
    builder, no I/O).

`apply_categorization_result(session, transaction, result) -> str`
    Inputs: an open `AsyncSession`, a `Transaction`, a
    `CategorizationResult`. Outputs: the action taken (`"auto_applied"`
    or `"queued_for_review"`). Side effects: WRITES onto `transaction`
    (`category`/`subcategory`/`confidence_score`) when confidence is
    `>= AUTO_APPLY_CONFIDENCE_THRESHOLD`; otherwise leaves `transaction`
    untouched. Either way, `session.add()`s a new `DecisionLog` row --
    the caller must commit. This is the only function in this module
    that touches the database or mutates its input.

Does NOT do
-----------
- Does not fetch transactions from the database -- callers supply them.
- Does not decide which transactions need categorizing (that's the
  caller's job, e.g. `app.agents.dispatcher.categorize_and_apply`).
- Does not retry on LLM failure -- a parse failure raises immediately.
- Does not call the forecaster or the dispatcher.

See SKILLS.md at the project root for current eval accuracy and a
narrative summary of this skill.
"""

import time
from functools import lru_cache
from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import DecisionLog, Transaction
from app.observability import get_logger, log_event
from app.rag.vector_store import retrieve_similar, retrieve_similar_transactions

logger = get_logger("cashflow_agent.categorizer")

SKILL_VERSION = "1.0.0"

# Confidence at/above this gets applied straight to the transaction;
# below it gets queued for a human to review.
#
# Observed confidence is discretized to a handful of values
# (0.95/0.90/0.85/0.80/0.70), so this is a choice between buckets, not a
# continuous dial. 0.80 keeps everything the model is reasonably sure
# about flowing through automatically, while still routing genuine
# unknowns -- statement strings that identify no merchant at all, see
# UNRESOLVABLE_DESCRIPTIONS in scripts/generate_synthetic_data.py -- to a
# human. Note this guards against MODEL uncertainty only: it cannot
# detect retrieval being broken, because a model given no context still
# answers confidently (see EmptyIndexError in app/rag/vector_store.py).
AUTO_APPLY_CONFIDENCE_THRESHOLD = 0.80

DEFAULT_CATEGORIES = (
    "Housing",
    "Utilities",
    "Groceries",
    "Dining",
    "Transport",
    "Shopping",
    "Entertainment",
    "Healthcare",
    "Personal Care",
    "Subscriptions",
    "Income",
    "Transfer",
    "Travel",
    "Other",
)


class CategorizationResult(BaseModel):
    category: str
    subcategory: str
    confidence: float = Field(ge=0, le=1, description="Confidence in this categorization, from 0 to 1.")
    reasoning: str = Field(description="One sentence explaining why this category/subcategory was chosen.")


@lru_cache
def get_llm() -> ChatOpenAI:
    """Memoized ChatOpenAI client -- one instance per process, no `global` mutation.

    `max_retries` matters for bulk runs: a 2,500-row batch against a
    30,000 TPM account will hit 429s, and without retry/backoff the first
    one fails the entire batch.
    """
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
        max_retries=settings.openai_max_retries,
    )


def _format_vendor_hits(hits: list[dict]) -> str:
    if not hits:
        return "(no similar known vendors found)"
    return "\n".join(
        f"- {hit['vendor_name']} -> {hit['canonical_category']}/{hit['canonical_subcategory']} "
        f"(similarity {hit['similarity']:.2f})"
        for hit in hits
    )


def _format_transaction_hits(hits: list[dict]) -> str:
    if not hits:
        return "(no similar past categorized transactions found)"
    return "\n".join(
        f"- \"{hit['document']}\" -> {hit['vendor_name']} / {hit['category']} "
        f"(similarity {hit['similarity']:.2f})"
        for hit in hits
    )


def _build_messages(
    transaction: Transaction,
    vendor_hits: list[dict],
    transaction_hits: list[dict],
    allowed_categories: Sequence[str],
) -> list:
    direction = "debit (money out)" if transaction.amount < 0 else "credit (money in)"

    system_prompt = (
        "You are a financial transaction categorization assistant. "
        f"You must choose `category` from exactly this closed list, with no exceptions: "
        f"{', '.join(allowed_categories)}. "
        "Choose a specific, human-readable `subcategory` that narrows down the category "
        "(e.g. category 'Dining', subcategory 'coffee shop'). "
        "Ground your answer in the retrieved vendor knowledge base and similar past "
        "transactions provided below where relevant -- prefer their categories when they "
        "clearly apply. If the retrieved context is weak, sparse, or conflicting, or the "
        "transaction description is genuinely ambiguous, reflect that by giving a lower "
        "`confidence` score rather than guessing with false certainty."
    )

    human_prompt = (
        "Transaction to categorize:\n"
        f"- Description: {transaction.raw_description}\n"
        f"- Amount: {transaction.amount} ({direction})\n\n"
        "Similar known vendors (from vendor knowledge base):\n"
        f"{_format_vendor_hits(vendor_hits)}\n\n"
        "Similar past transactions already categorized:\n"
        f"{_format_transaction_hits(transaction_hits)}\n"
    )

    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


def categorize_transaction(
    transaction: Transaction,
    allowed_categories: Sequence[str] = DEFAULT_CATEGORIES,
) -> CategorizationResult:
    """Recommend a category/subcategory for `transaction`. Pure function -- no DB writes.

    Synchronous and blocking (network calls, no `await` inside). From an
    async caller, invoke via `asyncio.to_thread(categorize_transaction, ...)`.
    """
    start = time.perf_counter()

    vendor_hits = retrieve_similar(transaction.raw_description, k=5)
    transaction_hits = retrieve_similar_transactions(transaction.raw_description, k=3)

    messages = _build_messages(transaction, vendor_hits, transaction_hits, allowed_categories)

    structured_llm = get_llm().with_structured_output(CategorizationResult, include_raw=True)
    raw_result = structured_llm.invoke(messages)
    result = raw_result["parsed"]
    if result is None:
        raise raw_result["parsing_error"] or RuntimeError("categorize_transaction: LLM output failed to parse")

    usage = raw_result["raw"].usage_metadata or {}
    all_hits = vendor_hits + transaction_hits
    top_similarity = max((hit["similarity"] for hit in all_hits), default=None)
    latency_ms = (time.perf_counter() - start) * 1000

    log_event(
        logger,
        "categorization",
        agent="categorizer",
        transaction_id=transaction.id,
        latency_ms=round(latency_ms, 1),
        n_vendor_hits=len(vendor_hits),
        n_transaction_hits=len(transaction_hits),
        top_similarity=round(top_similarity, 4) if top_similarity is not None else None,
        confidence=result.confidence,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )

    return result


CATEGORIZER_AGENT_NAME = "categorizer"
# Offline benchmark runs write decision rows against REAL transaction ids,
# so they must not share an agent_name with production decisions. Anything
# that asks "what is the latest categorizer decision for this transaction?"
# -- GET /review-queue most importantly -- filters on agent_name, and an
# eval row landing on top of a live one silently drops that transaction out
# of the human review queue.
EVAL_AGENT_NAME = "categorizer_eval"


def build_decision_log(
    transaction_id: int,
    result: CategorizationResult,
    action_taken: str = "pending",
    agent_name: str = CATEGORIZER_AGENT_NAME,
) -> DecisionLog:
    """Build (but don't persist) a DecisionLog row for a categorization decision.

    Takes a plain id rather than a Transaction object -- callers like
    run_eval.py work with transient, unpersisted Transaction instances
    whose `.id` is None; the real id has to come from elsewhere.
    `action_taken` defaults to "pending" for callers (like run_eval.py)
    that aren't running the real apply layer; the apply layer itself
    passes "auto_applied" or "queued_for_review".

    Benchmark callers must pass `agent_name=EVAL_AGENT_NAME` so their rows
    stay out of the queries that drive live behaviour -- see the comment
    on that constant.
    """
    return DecisionLog(
        transaction_id=transaction_id,
        agent_name=agent_name,
        decision_type="categorization",
        reasoning=result.reasoning,
        confidence_score=result.confidence,
        suggested_category=result.category,
        suggested_subcategory=result.subcategory,
        action_taken=action_taken,
    )


def apply_categorization_result(session: AsyncSession, transaction: Transaction, result: CategorizationResult) -> str:
    """The Phase 8 apply layer: decide auto-apply vs. queue-for-review and act on it.

    High-confidence results get written straight onto the transaction;
    low-confidence ones are left untouched for a human to review via
    GET /review-queue + POST /review/{id}/approve. Either way, a
    DecisionLog row records what happened, including the full reasoning
    string (never just the category). Returns the action taken.
    """
    if result.confidence >= AUTO_APPLY_CONFIDENCE_THRESHOLD:
        transaction.category = result.category
        transaction.subcategory = result.subcategory
        transaction.confidence_score = result.confidence
        action_taken = "auto_applied"
    else:
        action_taken = "queued_for_review"

    session.add(build_decision_log(transaction.id, result, action_taken=action_taken))
    return action_taken
