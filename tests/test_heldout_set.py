"""Guards the independence of the held-out generalization set.

The set is only worth running if nothing in it overlaps the vendor
knowledge base. That property is easy to break by accident -- adding a
merchant to VENDOR_KB_SEED that happens to appear here would quietly turn
a generalization test into a lexical-match test, and the score would go
UP while measuring less. Nothing would fail; the number would just stop
meaning what it claims.

These tests make that failure loud.
"""

import re

from app.agents.categorizer import DEFAULT_CATEGORIES
from scripts.generate_synthetic_data import (
    AMBIGUOUS_DESCRIPTIONS,
    CATEGORY_POOLS,
    UNRESOLVABLE_DESCRIPTIONS,
    VENDOR_KB_SEED,
)
from scripts.heldout_set import HELDOUT_TRANSACTIONS


def _tokens(text: str) -> set:
    """Alphabetic tokens of 4+ chars, lowercased -- brand-name granularity."""
    return {t for t in re.findall(r"[A-Za-z]{4,}", text.lower())}


# Common nouns that appear inside KB vendor NAMES without identifying the
# brand ("The Corner Market" contributes "market"). Sharing one of these
# is not leakage; sharing "spotify" is.
_GENERIC = {
    "market", "store", "salon", "studio", "shop", "repair", "bakery",
    "fitness", "insurance", "wholesale", "brokerage", "apartments", "misc",
    "retail", "transfer", "hour", "storage", "water", "grooming", "corner",
    "outlet", "farmers", "payroll", "wireless", "department", "instant",
    "neighborhood", "merchant", "kitchen", "plus", "creative", "cloud",
    "pharmacy", "airlines",
}


def test_no_kb_merchant_name_appears_in_a_heldout_description():
    """The whole point: these merchants must be unknown to retrieval.

    Compares against KB vendor NAMES rather than their descriptions --
    descriptions contain ordinary category words ("fresh", "discount")
    whose presence says nothing about whether the merchant is known.
    A shared brand token does.
    """
    kb_name_tokens = set()
    for vendor_name, *_ in VENDOR_KB_SEED:
        kb_name_tokens |= _tokens(vendor_name)
    kb_name_tokens -= _GENERIC

    overlaps = []
    for description, _, _, _ in HELDOUT_TRANSACTIONS:
        shared = _tokens(description) & kb_name_tokens
        if shared:
            overlaps.append((description, sorted(shared)))

    assert not overlaps, (
        "held-out descriptions share brand tokens with vendor KB names, so this "
        "is no longer a generalization test:\n"
        + "\n".join(f"  {d!r} shares {s}" for d, s in overlaps)
    )


def test_no_heldout_description_duplicates_the_synthetic_dataset():
    """Must not overlap the training-side descriptors either."""
    generated = set(AMBIGUOUS_DESCRIPTIONS) | set(UNRESOLVABLE_DESCRIPTIONS)
    for pool in CATEGORY_POOLS.values():
        generated |= set(pool["descriptions"])

    for description, _, _, _ in HELDOUT_TRANSACTIONS:
        assert description not in generated, (
            f"{description!r} is already in the synthetic dataset"
        )


def test_every_heldout_label_is_a_valid_category():
    """A label outside DEFAULT_CATEGORIES could never be predicted correctly."""
    for description, _, truth, _ in HELDOUT_TRANSACTIONS:
        assert truth in DEFAULT_CATEGORIES, (
            f"{description!r} is labelled {truth!r}, which is not in DEFAULT_CATEGORIES"
        )


def test_difficulty_labels_are_valid():
    for description, _, _, difficulty in HELDOUT_TRANSACTIONS:
        assert difficulty in ("easy", "medium", "hard"), (
            f"{description!r} has difficulty {difficulty!r}"
        )


def test_set_is_large_and_broad_enough_to_mean_something():
    categories = {t[2] for t in HELDOUT_TRANSACTIONS}
    assert len(HELDOUT_TRANSACTIONS) >= 50
    assert len(categories) >= 10, f"only covers {len(categories)} categories"


def test_no_duplicate_descriptions():
    descriptions = [t[0] for t in HELDOUT_TRANSACTIONS]
    assert len(descriptions) == len(set(descriptions))
