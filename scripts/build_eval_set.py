"""Builds a fixed, reproducible evaluation set for the categorizer.

Samples 150 transactions from the live database -- a mix of "obvious"
descriptions and the deliberately-ambiguous POS strings -- and labels each
with its ground-truth category, derived from the synthetic data
generator's own category tables (`true_category_for_description()` in
`scripts/generate_synthetic_data.py`), not from any RAG/LLM output.

Writes the result to `eval_set.json` at the project root. That file (not
this script's random sampling) is what makes eval runs reproducible -- run
`scripts/run_eval.py` against it as many times as you like.

Prerequisite: the database should have the full synthetic dataset loaded
(`python -m scripts.generate_synthetic_data`), not just a small sample --
there needs to be enough data to draw 150 rows with good ambiguous
coverage from.

Run with: python -m scripts.build_eval_set
"""

import asyncio
import json
import random
from collections import defaultdict

from sqlalchemy import select

from app.db.models import Transaction
from app.db.session import async_session_factory
from scripts.generate_synthetic_data import (
    AMBIGUOUS_DESCRIPTIONS,
    UNRESOLVABLE_DESCRIPTIONS,
    true_category_for_description,
)

EVAL_SET_PATH = "eval_set.json"
TOTAL_SIZE = 150
AMBIGUOUS_TARGET = 40

random.seed(7)


def _row_to_entry(row: Transaction) -> dict:
    return {
        "transaction_id": row.id,
        "account_id": row.account_id,
        "posted_date": row.posted_date.isoformat(),
        "amount": str(row.amount),
        "raw_description": row.raw_description,
        "true_category": true_category_for_description(row.raw_description),
        "is_ambiguous": row.raw_description in AMBIGUOUS_DESCRIPTIONS,
    }


def _sample_ambiguous(rows: list[Transaction], target: int) -> list[Transaction]:
    """At least one of each distinct ambiguous description, then top up randomly."""
    by_description = defaultdict(list)
    for row in rows:
        by_description[row.raw_description].append(row)

    picked = [random.choice(group) for group in by_description.values()]
    picked_ids = {row.id for row in picked}

    remaining = [row for row in rows if row.id not in picked_ids]
    random.shuffle(remaining)
    extra_needed = max(target - len(picked), 0)
    picked.extend(remaining[:extra_needed])

    return picked[:target]


async def main() -> None:
    async with async_session_factory() as session:
        all_rows = (await session.execute(select(Transaction))).scalars().all()

    # Drop the deliberately-unresolvable descriptors before sampling. They
    # identify no merchant, so there is no defensible ground truth for
    # them -- including them would measure whether the model guesses the
    # same way the label-writer did, not whether it categorizes correctly.
    # They exist to populate the review queue, and are exercised there.
    n_before = len(all_rows)
    all_rows = [row for row in all_rows if row.raw_description not in UNRESOLVABLE_DESCRIPTIONS]
    n_excluded = n_before - len(all_rows)
    if n_excluded:
        print(f"Excluded {n_excluded} unresolvable rows (no ground truth) from eval sampling.")

    if len(all_rows) < TOTAL_SIZE:
        raise SystemExit(
            f"Only {len(all_rows)} transactions in the database -- need at least "
            f"{TOTAL_SIZE}. Run `python -m scripts.generate_synthetic_data` first."
        )

    ambiguous_rows = [row for row in all_rows if row.raw_description in AMBIGUOUS_DESCRIPTIONS]
    obvious_rows = [row for row in all_rows if row.raw_description not in AMBIGUOUS_DESCRIPTIONS]

    ambiguous_target = min(AMBIGUOUS_TARGET, len(ambiguous_rows))
    sampled_ambiguous = _sample_ambiguous(ambiguous_rows, ambiguous_target)

    obvious_target = TOTAL_SIZE - len(sampled_ambiguous)
    sampled_obvious = random.sample(obvious_rows, min(obvious_target, len(obvious_rows)))

    sampled = sampled_ambiguous + sampled_obvious
    random.shuffle(sampled)

    entries = [_row_to_entry(row) for row in sampled]

    with open(EVAL_SET_PATH, "w") as f:
        json.dump(entries, f, indent=2)

    print(
        f"Wrote {len(entries)} rows to {EVAL_SET_PATH} "
        f"({len(sampled_ambiguous)} ambiguous, {len(sampled_obvious)} obvious, "
        f"{len(set(row.raw_description for row in sampled_ambiguous))} distinct ambiguous descriptions)."
    )


if __name__ == "__main__":
    asyncio.run(main())
