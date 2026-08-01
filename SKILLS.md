# Skills

This project has two "skills" — self-contained, versioned modules under
`app/agents/` that any endpoint or agent can import and call without
modification. Each exposes a `SKILL_VERSION` constant and a module
docstring documenting its full interface (inputs, outputs, side effects).
This file is the narrative summary; the docstrings in the source files
are the source of truth if the two ever disagree.

---

## Categorizer

**Module**: [`app/agents/categorizer.py`](app/agents/categorizer.py)
**Version**: `1.0.0`

Classifies a single transaction into a spending category using GPT-4o,
grounded in retrieval-augmented context from the vendor knowledge base
and already-categorized past transactions.

### `categorize_transaction(transaction, allowed_categories=DEFAULT_CATEGORIES) -> CategorizationResult`

- **Takes**: a `Transaction` (needs `.raw_description` and `.amount`;
  `.id` is only used for logging and may be `None`), and an optional
  closed list of allowed top-level categories (14 by default, including
  `Travel` and `Personal Care`).
- **Returns**: a `CategorizationResult` — `category`, `subcategory`, a
  `confidence` score (0–1), and a one-sentence `reasoning` string.
- **Does NOT**: write to the database, mutate the `transaction` it was
  given, decide whether to auto-apply or queue for review, fetch
  transactions itself, or retry on failure (a parse failure raises
  immediately rather than silently returning a bad result).
- **Blocking, on purpose**: this function is synchronous and makes real
  network calls (RAG embedding lookups + one GPT-4o call) with no
  `await` inside. Callers on the FastAPI event loop (background jobs,
  route handlers) must invoke it via `asyncio.to_thread(categorize_transaction, ...)`
  rather than calling it directly — see
  `app.agents.dispatcher.categorize_and_apply`, which does this on every
  loop iteration. Calling it directly from an `async def` on the main
  event loop blocks that loop for the full duration of the call; a
  2,515-row batch this way once made `GET /jobs/{id}` completely
  unresponsive for the ~52 minutes the job ran.

### `apply_categorization_result(session, transaction, result) -> str`

- **Takes**: an open async DB session, a `Transaction`, and a
  `CategorizationResult` (from `categorize_transaction`, or anywhere
  else — it doesn't care where the result came from).
- **Returns**: the action taken, `"auto_applied"` or
  `"queued_for_review"`.
- **Does**: writes `category`/`subcategory`/`confidence_score` onto the
  transaction when confidence is `>= AUTO_APPLY_CONFIDENCE_THRESHOLD`
  (currently `0.80`); otherwise leaves the transaction untouched.
  Either way, adds a `DecisionLog` row (via `build_decision_log`) to the
  session — the caller still has to commit.
- **Does NOT**: call the LLM itself, or decide *which* transactions to
  process — that's the caller's job (see
  `app.agents.dispatcher.categorize_and_apply`).

### Current eval accuracy

**100%** (150/150) on the fixed 150-row eval set
(`eval_set.json` / `scripts/run_eval.py`), vs. a **70.0%** naive
baseline (exact substring match against `VendorKB`, no LLM/RAG).

- Run id: `eval_20260801T173312`, completed 2026-08-01 17:37 UTC.
- Accuracy at `>= AUTO_APPLY_CONFIDENCE_THRESHOLD`: **100% (149/149)**.
  That bucket is the number that matters -- those predictions are written
  to the database with no human in the loop, so its error rate is the one
  a user would actually experience. Overall accuracy flatters the system
  by including rows a human would have caught anyway.

**Read this number skeptically.** It is 100% on synthetic data generated
by `scripts/generate_synthetic_data.py`, scored against ground-truth
labels produced by the same module. That is a closed loop: it
demonstrates the retrieval -> prompt -> structured-output pipeline works
end to end and that the taxonomy has no gaps, but it says very little
about generalization to real bank feeds, where descriptors are messier
and merchants are unbounded. The honest claim is "the pipeline is
correct on data it was designed for," not "the categorizer is 100%
accurate." The naive baseline is the more informative half of the
comparison: 70% means keyword matching alone gets most common merchants
right, and RAG + LLM is worth the cost specifically for the
POS-obscured descriptors it cannot handle.

Deliberately excluded from scoring: the ~64
`UNRESOLVABLE_DESCRIPTIONS` rows (`SQ *MERCHANT 88213`,
`POS DEBIT 4471`, ...). They identify no merchant, so there is no
defensible ground truth -- scoring them would measure whether the model
guesses the same way the label-writer did. They exist to exercise the
review queue instead.

### Confidence distribution (real run, 2,597 transactions)

| confidence | rows | outcome |
|---|---|---|
| 0.95 | 75 | auto-applied |
| 0.90 | 1,831 | auto-applied |
| 0.85 | 271 | auto-applied |
| 0.80 | 352 | auto-applied |
| 0.75 and below | 68 | queued for review |

97.4% auto-applied, 2.6% queued. Of the 68 queued, 56 are the
deliberately-unresolvable descriptors and 10 are `APPLE.COM/BILL` --
genuine real-world ambiguity, since Apple bills iCloud, App Store, and
Music through one descriptor, so hedging at 0.70 is correct behaviour
rather than a failure.

Note that confidence now takes ten distinct values rather than the five
seen previously. Retrieval quality varies meaningfully across
transactions now that vendor entries lead with their statement
descriptor; before, almost everything matched about equally poorly and
the model had little to discriminate on.

### Retrieval quality is a function of how the KB is written

The single biggest lever on this skill's output turned out to be
sentence structure in `VENDOR_KB_SEED`, not model choice, `k`, or the
threshold.

Retrieval embeds a raw transaction string (`WM SUPERCENTER #4521`)
against each vendor document. A long prose description dilutes the
embedding -- every clause about "big-box retailer" pulls the vector away
from the short, specific query. Measured, same vendor, three phrasings:

| vendor document | similarity to `WM SUPERCENTER #4521` |
|---|---|
| no statement descriptor at all | 0.408 |
| descriptor buried in trailing prose | 0.576 |
| **descriptor first, short gloss** | **0.641** |

`AMAZON.COM*A1B2C3D4` moved 0.426 -> 0.668 on the same change. That
similarity is quoted into the prompt, and the prompt instructs the model
to lower confidence when retrieval looks weak -- so KB phrasing
propagates directly into how many transactions land in the review queue.
Rewriting all 103 entries descriptor-first moved the auto-apply rate from
69% to 97.4%.

Two things make this trade cheap. First, the description text never
reaches the LLM (`_format_vendor_hits` passes only vendor name, category,
and similarity), so optimizing it for retrieval instead of readability
costs nothing. Second, it is reversible and measurable in isolation --
no LLM calls needed to evaluate a phrasing change.

Caveat worth stating: this optimizes lexical overlap with descriptors
already known. It does not obviously improve generalization to unseen
merchants, which is what RAG is nominally for. A short gloss is retained
after each descriptor rather than reducing entries to bare strings, to
keep some semantic material for that case.

---

## Forecaster

**Module**: [`app/agents/forecaster.py`](app/agents/forecaster.py)
**Version**: `1.0.0`

Projects a daily cash-flow balance curve from transaction history using
simple time-series arithmetic — no LLM call, so it's fast enough to run
synchronously from a request handler.

### `forecast_cashflow(horizon: int) -> dict`

- **Takes**: `horizon`, a number of days to project forward.
- **Returns**: a dict with `horizon`, `starting_balance`, `lowest_point`
  (+ `lowest_point_date`), `ending_balance`, a plain-language
  `explanation` string, and the full daily `balance_curve` (a list of
  `{"date", "balance"}` points).
- **Does**: reads every `Transaction` row in the database (no account
  scoping); recurring transactions (`Transaction.is_recurring`, a flag
  set upstream, not detected here) get clustered into "legs" and
  projected forward on their historical day-of-month and average
  amount — correctly splitting a twice-monthly vendor like payroll into
  two legs instead of one meaningless blended average; everything else
  contributes a flat average daily rate. Writes one `AgentMetric` row
  recording its own latency, and emits one structured JSON log line.
- **Does NOT**: call any LLM; reason about anomalies or one-off spikes;
  accept a starting balance or account scope (it always sums the full
  table); write to `Transaction` or any other business table; or cache
  / memoize anything between calls.

### Current eval accuracy

**Not applicable — no accuracy benchmark exists for this skill.**
Unlike the categorizer, there's no ground-truth "correct forecast" to
compare against, so there's nothing equivalent to `run_eval.py` for it.
The one thing that has been manually sanity-checked: after the synthetic
data was rebalanced to a plausible household spending pattern (see prior
session notes), a 30-day forecast against the real dataset projected a
starting balance of **-$5,851**, dipping to **-$7,114** mid-month before
partially recovering around payroll dates — directionally sensible given
the known recurring vendors, but not a scored metric.
