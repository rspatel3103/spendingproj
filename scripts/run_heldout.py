"""Scores the categorizer against the held-out generalization set.

Unlike scripts/run_eval.py, nothing here shares a source with the vendor
knowledge base or the synthetic generator -- see scripts/heldout_set.py
for why that matters. This is the measurement that speaks to whether
retrieval generalizes to merchants the system has never seen.

Reports accuracy overall, by difficulty, and split at the auto-apply
threshold -- the last of these is the one that matters, since it is the
error rate a user would actually experience on unfamiliar merchants.

Makes one GPT-4o call per row (~60). Writes nothing to the database:
these are not real transactions and must not pollute production tables
or the review queue.

Run with: python -m scripts.run_heldout
"""

import asyncio
import time
from collections import defaultdict
from datetime import date
from decimal import Decimal

from app.agents.categorizer import AUTO_APPLY_CONFIDENCE_THRESHOLD, categorize_transaction
from app.config import settings
from app.db.models import Transaction
from scripts.heldout_set import HELDOUT_TRANSACTIONS


async def _categorize(description: str, amount: float, semaphore: asyncio.Semaphore):
    async with semaphore:
        transaction = Transaction(
            account_id="heldout",
            posted_date=date(2026, 8, 1),
            amount=Decimal(str(amount)),
            raw_description=description,
        )
        return await asyncio.to_thread(categorize_transaction, transaction)


async def main() -> None:
    print(f"Scoring {len(HELDOUT_TRANSACTIONS)} held-out transactions.")
    print("None of these merchants appear in the vendor knowledge base.\n")

    semaphore = asyncio.Semaphore(settings.categorization_concurrency)
    start = time.perf_counter()
    results = await asyncio.gather(
        *(_categorize(d, a, semaphore) for d, a, _, _ in HELDOUT_TRANSACTIONS)
    )
    elapsed = time.perf_counter() - start

    correct = 0
    by_difficulty = defaultdict(lambda: [0, 0])
    by_bucket = defaultdict(lambda: [0, 0])
    misses = []

    for (description, amount, truth, difficulty), result in zip(HELDOUT_TRANSACTIONS, results):
        is_correct = result.category == truth
        correct += is_correct

        by_difficulty[difficulty][0] += 1
        by_difficulty[difficulty][1] += is_correct

        bucket = (
            "auto-applied" if result.confidence >= AUTO_APPLY_CONFIDENCE_THRESHOLD else "queued"
        )
        by_bucket[bucket][0] += 1
        by_bucket[bucket][1] += is_correct

        mark = "ok  " if is_correct else "MISS"
        print(
            f"  {mark} {description:<28} pred={result.category:<14} "
            f"true={truth:<14} conf={result.confidence:.2f}"
        )
        if not is_correct:
            misses.append((description, result.category, truth, result.confidence, result.reasoning))

    n = len(HELDOUT_TRANSACTIONS)
    print("\n" + "=" * 74)
    print("HELD-OUT GENERALIZATION REPORT")
    print("=" * 74)
    print(f"Overall accuracy: {correct}/{n} = {correct/n:.1%}   ({elapsed:.0f}s)\n")

    print("By difficulty:")
    for difficulty in ("easy", "medium", "hard"):
        total, right = by_difficulty[difficulty]
        if total:
            print(f"  {difficulty:<8} {right}/{total} = {right/total:.1%}")

    print(f"\nSplit at the auto-apply threshold ({AUTO_APPLY_CONFIDENCE_THRESHOLD}):")
    for bucket in ("auto-applied", "queued"):
        total, right = by_bucket[bucket]
        if total:
            print(f"  {bucket:<14} n={total:<4} accuracy={right/total:.1%}")
        else:
            print(f"  {bucket:<14} n=0")
    print(
        "\n  The auto-applied row is the one that matters: those would be written\n"
        "  to the database unreviewed. If the guardrail works, errors concentrate\n"
        "  in the queued bucket."
    )

    print(f"\nMisses ({len(misses)}):")
    for description, predicted, truth, confidence, reasoning in sorted(misses, key=lambda m: -m[3]):
        print(f"  {description!r} conf={confidence:.2f}")
        print(f"      predicted={predicted!r}  true={truth!r}")
        print(f"      reasoning: {reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
