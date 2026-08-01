"""Held-out generalization test set.

The main eval set has a structural weakness: `scripts/run_eval.py` scores
predictions against labels produced by the same module that generated the
data AND wrote the vendor knowledge base. Every descriptor in it has a
purpose-built KB entry leading with that exact string. A high score there
shows the pipeline works; it cannot show whether retrieval generalizes,
because there is nothing unfamiliar to generalize to.

This set is deliberately adversarial to that setup:

  * **No merchant here appears in VENDOR_KB_SEED.** Not one of the 103.
    A test enforces this, so the set cannot silently rot into a lexical
    match as the KB grows.
  * **Labels were assigned by hand**, not derived from the generator's
    category tables, so a bug in the generator's taxonomy cannot make a
    prediction look correct.
  * **Descriptor formats vary** the way real bank feeds do -- trailing
    store numbers, city/state suffixes, ALL CAPS truncation, processor
    prefixes, and abbreviations no keyword matcher would resolve.

Categorizing these requires the model to reason from semantic similarity
to *related* merchants plus its own world knowledge. That is the
capability RAG is nominally for, and the one the closed-loop eval cannot
measure.

`difficulty` is a note for reading the results, not an input to scoring:
  easy    -- a well-known brand, just absent from the KB
  medium  -- obscured or abbreviated, but inferable
  hard    -- genuinely ambiguous; a careful human might disagree

Run with: python -m scripts.heldout_set        (prints the set)
Score with: python -m scripts.run_heldout
"""

# (raw_description, amount, true_category, difficulty)
HELDOUT_TRANSACTIONS = [
    # --- Groceries -----------------------------------------------------
    ("PUBLIX SUPER MARKET #1423", -84.19, "Groceries", "easy"),
    ("WEGMANS FOOD MKT 0089", -112.40, "Groceries", "easy"),
    ("KROGER #0455 FUEL CTR", -46.02, "Groceries", "medium"),
    ("BODEGA LA ESQUINA", -12.75, "Groceries", "medium"),
    ("99 RANCH MARKET 221", -63.88, "Groceries", "medium"),
    ("FRESH THYME MKTPL 118", -37.10, "Groceries", "medium"),

    # --- Dining --------------------------------------------------------
    ("CHICK-FIL-A #02291", -14.86, "Dining", "easy"),
    ("PF CHANGS 4471 SEATTLE", -68.20, "Dining", "easy"),
    ("SQ *TACOS EL PRIMO", -18.40, "Dining", "medium"),
    ("TST* THE BREAKFAST NOOK", -31.55, "Dining", "medium"),
    ("WAITR*BIRRIA HOUSE", -27.90, "Dining", "medium"),
    ("DD/BR #340021 Q35", -6.45, "Dining", "hard"),
    ("OLIVIAS TABLE LLC", -22.30, "Dining", "hard"),

    # --- Transport -----------------------------------------------------
    ("EXXONMOBIL 97442110", -52.18, "Transport", "easy"),
    ("MTA*NYCT PAYGO", -2.90, "Transport", "medium"),
    ("CIRCLE K #20991 FUEL", -41.60, "Transport", "medium"),
    ("VIA TRANSPORTATION INC", -9.75, "Transport", "medium"),
    ("PARKMOBILE*ZONE 4412", -6.00, "Transport", "medium"),
    ("DISCOUNT TIRE CO 0231", -218.44, "Transport", "hard"),

    # --- Shopping ------------------------------------------------------
    ("BARNES & NOBLE #2871", -34.99, "Shopping", "easy"),
    ("TEMU.COM ORDER 8821", -19.99, "Shopping", "easy"),
    ("MENARDS 3155 OMAHA", -88.13, "Shopping", "medium"),
    ("SHEIN.COM*ORDER", -42.60, "Shopping", "medium"),
    ("MICRO CENTER #0155", -159.00, "Shopping", "medium"),
    ("ACE HDWE 4471 MAIN ST", -27.35, "Shopping", "medium"),

    # --- Entertainment -------------------------------------------------
    ("REGAL CINEMAS 0812", -23.50, "Entertainment", "easy"),
    ("PARAMOUNT+ 8887779999", -11.99, "Entertainment", "easy"),
    ("EPIC GAMES STORE", -29.99, "Entertainment", "medium"),
    ("STUBHUB*EVENT 771204", -142.00, "Entertainment", "medium"),
    ("TOPGOLF NASHVILLE 021", -76.40, "Entertainment", "hard"),

    # --- Healthcare ----------------------------------------------------
    ("RITE AID #06612", -28.44, "Healthcare", "easy"),
    ("LABCORP 8004452331", -142.00, "Healthcare", "medium"),
    ("ZOCDOC VISIT COPAY", -40.00, "Healthcare", "medium"),
    ("HEB PHARMACY 0231", -18.20, "Healthcare", "hard"),
    ("1800CONTACTS", -96.00, "Healthcare", "hard"),

    # --- Personal Care -------------------------------------------------
    ("ULTA BEAUTY #1120", -54.30, "Personal Care", "easy"),
    ("SUPERCUTS 04412", -32.00, "Personal Care", "easy"),
    ("SQ *THE WAX ROOM", -70.00, "Personal Care", "medium"),
    ("PLANET FIT CLUB FEES", -24.99, "Personal Care", "medium"),
    ("CLASSPASS.COM", -79.00, "Personal Care", "hard"),

    # --- Subscriptions -------------------------------------------------
    ("PANDORA PLUS 8442", -16.99, "Subscriptions", "medium"),
    ("MSFT*MICROSOFT 365", -9.99, "Subscriptions", "easy"),
    ("AUDIBLE*MEMBERSHIP", -14.95, "Subscriptions", "easy"),
    ("BACKBLAZE INC", -9.00, "Subscriptions", "medium"),
    ("SUBSTACK*THE DISPATCH", -8.00, "Subscriptions", "hard"),

    # --- Travel --------------------------------------------------------
    ("SOUTHWEST AIRLINES 5262", -318.40, "Travel", "easy"),
    ("HILTON GARDEN INN AUS", -212.00, "Travel", "easy"),
    ("VRBO*BOOKING 22114", -640.00, "Travel", "medium"),
    ("TURO*TRIP 88123", -184.00, "Travel", "hard"),
    ("AMTRAK .COM 8007458", -87.00, "Travel", "hard"),

    # --- Transfer ------------------------------------------------------
    ("REVOLUT*SEND 4412", -60.00, "Transfer", "medium"),
    ("COINBASE.COM PURCHASE", -250.00, "Transfer", "hard"),
    ("FIDELITY BROKERAGE CONTR", -500.00, "Transfer", "hard"),

    # --- Utilities -----------------------------------------------------
    ("DUKE ENERGY BILLPAY", -148.22, "Utilities", "easy"),
    ("SPECTRUM 8556435465", -79.99, "Utilities", "medium"),
    ("T-MOBILE POSTPAID", -85.00, "Utilities", "easy"),
    ("CITY OF AUSTIN UTILITIES", -132.10, "Utilities", "medium"),

    # --- Housing -------------------------------------------------------
    ("GREYSTAR PROPERTY MGMT", -2150.00, "Housing", "medium"),
    ("LEMONADE INSURANCE CO", -18.00, "Housing", "hard"),

    # --- Income --------------------------------------------------------
    ("PAYCHEX DIR DEP MERIDIAN", 3820.55, "Income", "medium"),
    ("IRS TREAS 310 TAX REF", 1204.00, "Income", "hard"),
]


def main() -> None:
    from collections import Counter

    by_category = Counter(t[2] for t in HELDOUT_TRANSACTIONS)
    by_difficulty = Counter(t[3] for t in HELDOUT_TRANSACTIONS)

    print(f"{len(HELDOUT_TRANSACTIONS)} held-out transactions\n")
    print("by category:")
    for category, n in by_category.most_common():
        print(f"  {category:<16} {n}")
    print("\nby difficulty:")
    for difficulty in ("easy", "medium", "hard"):
        print(f"  {difficulty:<16} {by_difficulty[difficulty]}")


if __name__ == "__main__":
    main()
