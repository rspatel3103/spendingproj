"""Runs the categorizer against the fixed eval set and reports accuracy.

For every row in `eval_set.json` (see `scripts/build_eval_set.py`), calls
`categorize_transaction()` and compares its predicted category against the
row's `true_category`, writing one `EvalResult` row per transaction. Also
computes a naive baseline (exact substring match against VendorKB vendor
names, no LLM, no RAG) so the real number has something to be compared
against.

This makes ~150 sequential OpenAI calls (one embedding lookup + one GPT-4o
call per row) and will take a few minutes.

Run with: python -m scripts.run_eval
"""

import asyncio
import json
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import select

from app.agents.categorizer import (
    AUTO_APPLY_CONFIDENCE_THRESHOLD,
    EVAL_AGENT_NAME,
    build_decision_log,
    categorize_transaction,
)
from app.db.models import EvalResult, Transaction, VendorKB
from app.db.session import async_session_factory
from scripts.generate_synthetic_data import CATEGORY_NAME_MAP

EVAL_SET_PATH = "eval_set.json"

# Buckets are pinned to AUTO_APPLY_CONFIDENCE_THRESHOLD rather than to
# round numbers, because the question the eval has to answer is "how
# accurate are the predictions we apply without a human looking at them?"
# A bucket boundary that doesn't match the threshold can't answer it.
_T = AUTO_APPLY_CONFIDENCE_THRESHOLD
AUTO_APPLIED_BUCKET = f">={_T:.2f} (auto-applied)"
QUEUED_BUCKET = f"<{_T:.2f} (queued for review)"
CONFIDENCE_BUCKETS = (AUTO_APPLIED_BUCKET, QUEUED_BUCKET)


def _confidence_bucket(confidence: float) -> str:
    return AUTO_APPLIED_BUCKET if confidence >= _T else QUEUED_BUCKET


async def _naive_baseline(eval_rows: list[dict]) -> tuple[float, dict]:
    async with async_session_factory() as session:
        vendors = (await session.execute(select(VendorKB))).scalars().all()

    predictions = {}
    for row in eval_rows:
        description_upper = row["raw_description"].upper()
        best_match = None
        for vendor in vendors:
            name_upper = vendor.vendor_name.upper()
            if name_upper in description_upper:
                if best_match is None or len(name_upper) > len(best_match.vendor_name):
                    best_match = vendor

        predicted = CATEGORY_NAME_MAP.get(best_match.canonical_category, "Other") if best_match else "Other"
        predictions[row["transaction_id"]] = predicted

    correct = sum(
        1 for row in eval_rows if predictions[row["transaction_id"]] == row["true_category"]
    )
    accuracy = correct / len(eval_rows)
    return accuracy, predictions


async def main() -> None:
    with open(EVAL_SET_PATH) as f:
        eval_rows = json.load(f)

    print(f"Loaded {len(eval_rows)} eval rows from {EVAL_SET_PATH}.\n")

    print("Computing naive baseline (exact substring match against VendorKB, no LLM/RAG)...")
    baseline_accuracy, _ = await _naive_baseline(eval_rows)
    print(f"Naive baseline accuracy: {baseline_accuracy:.1%}\n")

    run_id = f"eval_{datetime.utcnow():%Y%m%dT%H%M%S}"
    print(f"Running categorize_transaction() over {len(eval_rows)} rows (run_id={run_id})...")

    misses = []
    bucket_totals = {bucket: 0 for bucket in CONFIDENCE_BUCKETS}
    bucket_correct = {bucket: 0 for bucket in CONFIDENCE_BUCKETS}
    total_correct = 0

    eval_result_rows = []
    decision_log_rows = []

    for i, row in enumerate(eval_rows, start=1):
        transaction = Transaction(
            account_id=row["account_id"],
            posted_date=date.fromisoformat(row["posted_date"]),
            amount=Decimal(row["amount"]),
            raw_description=row["raw_description"],
        )
        result = categorize_transaction(transaction)

        is_correct = result.category == row["true_category"]
        total_correct += int(is_correct)

        bucket = _confidence_bucket(result.confidence)
        bucket_totals[bucket] += 1
        bucket_correct[bucket] += int(is_correct)

        mark = "OK" if is_correct else "MISS"
        print(
            f"[{i}/{len(eval_rows)}] {mark:>4}  {row['raw_description']!r:35s} "
            f"pred={result.category:<14} true={row['true_category']:<14} conf={result.confidence:.2f}"
        )

        if not is_correct:
            misses.append(
                {
                    "raw_description": row["raw_description"],
                    "predicted": result.category,
                    "true": row["true_category"],
                    "confidence": result.confidence,
                }
            )

        eval_result_rows.append(
            EvalResult(
                run_id=run_id,
                transaction_id=row["transaction_id"],
                predicted_category=result.category,
                true_category=row["true_category"],
                correct=is_correct,
            )
        )
        # agent_name=EVAL_AGENT_NAME keeps these benchmark rows out of the
        # "latest categorizer decision" queries that drive GET /review-queue.
        # They reference real transaction ids, so without the distinct name
        # an eval run silently empties rows out of the human review queue.
        decision_log_rows.append(
            build_decision_log(row["transaction_id"], result, agent_name=EVAL_AGENT_NAME)
        )

    async with async_session_factory() as session:
        session.add_all(eval_result_rows + decision_log_rows)
        await session.commit()

    overall_accuracy = total_correct / len(eval_rows)

    print("\n" + "=" * 70)
    print("EVAL REPORT")
    print("=" * 70)
    print(f"Overall accuracy:      {overall_accuracy:.1%}  ({total_correct}/{len(eval_rows)})")
    print(f"Naive baseline:        {baseline_accuracy:.1%}  (exact substring match, no LLM/RAG)")
    print()
    print("Accuracy by confidence bucket (split at the auto-apply threshold):")
    for bucket in CONFIDENCE_BUCKETS:
        total = bucket_totals[bucket]
        if total == 0:
            print(f"  {bucket:>26}: no predictions in this bucket")
            continue
        acc = bucket_correct[bucket] / total
        print(f"  {bucket:>26}: {acc:.1%}  ({bucket_correct[bucket]}/{total})")
    print()
    print(
        "  The top bucket is the number that matters: those are applied to the\n"
        "  database with no human in the loop, so its accuracy is the real\n"
        "  error rate a user would experience."
    )
    print()
    print(f"Misses ({len(misses)}):")
    for miss in sorted(misses, key=lambda m: m["confidence"], reverse=True):
        print(
            f"  {miss['raw_description']!r:35s} predicted={miss['predicted']:<14} "
            f"true={miss['true']:<14} confidence={miss['confidence']:.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
