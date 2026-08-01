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

The stronger number is **2,533/2,533 on the full dataset** -- 17x the
sample, same ground-truth source, and free to compute since no LLM call
is needed to score already-stored predictions. See the calibration
section below.

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
| 0.95 and above | 85 | auto-applied |
| 0.90 | 1,818 | auto-applied |
| 0.85 | 272 | auto-applied |
| 0.80 | 357 | auto-applied |
| below 0.80 | 57 | queued for review |

97.8% auto-applied, 2.2% queued. Almost all of the queue is the
deliberately-unresolvable descriptors; the remainder is genuine
real-world ambiguity such as `APPLE.COM/BILL`, which Apple uses for
iCloud, App Store, and Music alike -- hedging there is correct behaviour
rather than a failure.

Note that confidence now takes ten distinct values rather than the five
seen previously. Retrieval quality varies meaningfully across
transactions now that vendor entries lead with their statement
descriptor; before, almost everything matched about equally poorly and
the model had little to discriminate on.

### Is the confidence score calibrated? Measured: no

The apply guardrail rests entirely on a number the model reports about
itself, so it is worth knowing what that number is actually worth. This
is measurable for free -- the generator can produce ground truth for
every resolvable transaction, so all 2,533 of them can be scored without
a single API call, rather than only the 150-row eval set.

**Accuracy on the full dataset: 2,533/2,533 (100%).**

| claimed confidence | n | observed accuracy | gap |
|---|---|---|---|
| 1.00 | 3 | 100% | +0.0% |
| 0.95 | 82 | 100% | +5.0% |
| 0.90 | 1,818 | 100% | +10.0% |
| 0.85 | 272 | 100% | +15.0% |
| 0.80 | 357 | 100% | +20.0% |
| 0.70 | 1 | 100% | +30.0% |

Expected Calibration Error: **0.118**, entirely in the *under*-confident
direction. Every bucket outperforms its own claim. That is the opposite
of the usual LLM failure mode, and it is not a virtue -- it means the
number does not mean what it says.

**The guardrail does not discriminate.** Splitting at the 0.80 threshold:
auto-applied rows are 100% accurate and queued rows are 100% accurate.
Confidence carries no signal about correctness on this dataset, so the
threshold currently costs human attention and catches nothing.

That is a statement about the dataset as much as the model: **you cannot
validate a guardrail on data that contains no mistakes.** Demonstrating
that the review queue earns its keep requires harder or real-world data,
and until then the honest claim is that its value is unproven.

**What confidence appears to actually track is retrieval match quality,
not correctness.** Before the taxonomy fix below, the only two errors in
the dataset were `24 HOUR FITNESS` categorized as `Personal Care` against
a ground truth of `Healthcare` -- at 0.90 confidence, auto-applied. The
model was confident because retrieval *had* found a strong vendor match;
the disagreement was over taxonomy, which similarity cannot speak to.

**The sharpest finding: identical inputs produced different answers at
identical confidence.** The same descriptor, same amount, same 0.90
confidence, resolved to `Healthcare` 11 times and `Personal Care` 2
times. `temperature=0` reduces sampling variance but does not eliminate
it, and the model reported the same confidence for the answer it gives
85% of the time as for the one it gives 15% of the time -- its stated
confidence does not track its own instability.

That points at a concrete improvement with evidence behind it:
**self-consistency across N samples would be a better confidence signal
than asking the model to introspect.** Repeated sampling surfaced real
uncertainty that self-reporting never did. Combining it with a similarity
floor and a retrieval-health check (`n_vendor_hits < k` means the index
is degraded) would give a guardrail grounded in observable signals rather
than one.

Root cause of those two errors was a taxonomy inconsistency of my own
making: a gym was `health -> Healthcare` while yoga and pilates were
`personal_care -> Personal Care`. The model flip-flopped because the
boundary genuinely did not exist. Moving the gym alongside yoga took the
error count to zero -- the third instance in this project of a
"model error" that was actually a gap in the category list, after
`Travel` and `Personal Care` themselves.

Reproduce: score `decision_log`'s latest categorizer row per transaction
against `true_category_for_description()`, bucketed by
`confidence_score`.

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
