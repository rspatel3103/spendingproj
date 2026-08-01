"""Regression tests for the two tiers of special descriptions.

The original month-placement formula for ambiguous POS strings used a
modular index (`(year*12+month+i) % len(AMBIGUOUS_DESCRIPTIONS)`) that
could collide across different months, silently dropping some of the
deliberately-ambiguous descriptions from the generated dataset entirely.
`_partition_across_months()` replaced that with an explicit partition of
a shuffled copy of the list, so coverage is guaranteed by construction
rather than by chance -- these tests pin that guarantee down for both
tiers.

The second tier (UNRESOLVABLE_DESCRIPTIONS) carries an extra guarantee
worth protecting: those strings must have NO vendor knowledge base entry
and NO ground-truth category, because their whole purpose is to fail
retrieval and land in the human review queue. A well-meaning future edit
that "helpfully" adds a KB entry for them would silently empty the
review queue.
"""

import random

import pytest

from scripts.generate_synthetic_data import (
    AMBIGUOUS_DESCRIPTIONS,
    UNRESOLVABLE_DESCRIPTIONS,
    UNRESOLVABLE_REPEATS,
    VENDOR_KB_SEED,
    ambiguous_assignments_for_months,
    build_oneoff_transactions,
    build_recurring_transactions,
    month_range,
    true_category_for_description,
    unresolvable_assignments_for_months,
)


def test_ambiguous_assignments_cover_all_descriptions_exactly_once():
    random.seed(123)
    months = month_range(13, 2026, 7)

    assignments = ambiguous_assignments_for_months(months)

    assert len(assignments) == len(months)
    flattened = [description for month_descriptions in assignments for description in month_descriptions]
    assert sorted(flattened) == sorted(AMBIGUOUS_DESCRIPTIONS)


def test_unresolvable_assignments_cover_all_descriptions_with_repeats():
    random.seed(123)
    months = month_range(13, 2026, 7)

    assignments = unresolvable_assignments_for_months(months)

    assert len(assignments) == len(months)
    flattened = [d for month_descriptions in assignments for d in month_descriptions]
    assert set(flattened) == set(UNRESOLVABLE_DESCRIPTIONS)
    assert len(flattened) == len(UNRESOLVABLE_DESCRIPTIONS) * UNRESOLVABLE_REPEATS


def test_full_year_generation_places_every_special_description():
    random.seed(123)
    months = month_range(13, 2026, 7)
    ambiguous = ambiguous_assignments_for_months(months)
    unresolvable = unresolvable_assignments_for_months(months)

    seen_ambiguous = set()
    seen_unresolvable = set()
    for (year, month), ambiguous_descriptions, unresolvable_descriptions in zip(
        months, ambiguous, unresolvable
    ):
        recurring_rows = build_recurring_transactions(year, month)
        oneoff_rows = build_oneoff_transactions(
            year,
            month,
            count=200,
            ambiguous_descriptions=ambiguous_descriptions,
            unresolvable_descriptions=unresolvable_descriptions,
        )

        for row in recurring_rows + oneoff_rows:
            if row.raw_description in AMBIGUOUS_DESCRIPTIONS:
                seen_ambiguous.add(row.raw_description)
            if row.raw_description in UNRESOLVABLE_DESCRIPTIONS:
                seen_unresolvable.add(row.raw_description)

    assert seen_ambiguous == set(AMBIGUOUS_DESCRIPTIONS)
    assert seen_unresolvable == set(UNRESOLVABLE_DESCRIPTIONS)


def test_unresolvable_descriptions_have_no_vendor_kb_entry():
    """The review queue only stays populated while these remain unmatchable."""
    kb_text = " ".join(f"{name} {description}" for name, _, _, description, _ in VENDOR_KB_SEED)
    for description in UNRESOLVABLE_DESCRIPTIONS:
        assert description not in kb_text, (
            f"{description!r} appears in VENDOR_KB_SEED -- it is supposed to be "
            f"unresolvable so it queues for human review."
        )


def test_unresolvable_descriptions_have_no_ground_truth():
    """Scoring a row nobody can label would measure guessing, not accuracy."""
    for description in UNRESOLVABLE_DESCRIPTIONS:
        with pytest.raises(ValueError, match="deliberately unresolvable"):
            true_category_for_description(description)


def test_every_resolvable_description_has_a_ground_truth_category():
    """Guards the eval set against a description with no label mapping."""
    for description in AMBIGUOUS_DESCRIPTIONS:
        assert true_category_for_description(description)
