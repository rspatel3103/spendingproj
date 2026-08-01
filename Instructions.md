# API Instructions

How to run the cashflow-agent API and call every endpoint it exposes,
with example requests and real response shapes.

## Start the server

```bash
cd cashflow-agent
source .venv/bin/activate
uvicorn app.main:app --reload
```

Listens on `http://localhost:8000`. All examples below assume that base
URL. Add `-i` to any `curl` command to see the HTTP status code, or pipe
through `python3 -m json.tool` to pretty-print the response body.

---

## `GET /health`

Liveness check.

```bash
curl localhost:8000/health
```

```json
{"status": "ok"}
```

---

## `POST /transactions/ingest`

Adds new transactions (uncategorized) and kicks off categorization for
just that batch as a background job. Accepts either a JSON body or a CSV
file upload.

**JSON:**
```bash
curl -X POST localhost:8000/transactions/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "account_id": "acct_1",
        "posted_date": "2026-07-30",
        "amount": -12.50,
        "raw_description": "SQ *JOES COFFEE",
        "is_recurring": false
      }
    ]
  }'
```

**CSV** (columns: `account_id,posted_date,amount,raw_description,is_recurring`):
```bash
curl -X POST localhost:8000/transactions/ingest \
  -F "file=@transactions.csv"
```

Response (`202`-style, but returns `200` with a pending job):
```json
{
  "job_id": "3f2b1a9c-...",
  "status": "pending",
  "n_transactions_ingested": 1
}
```

Use the returned `job_id` with `GET /jobs/{job_id}` to check when
categorization finishes.

Errors: `400` for invalid/empty JSON or CSV, `422` for a JSON body that
fails schema validation.

---

## `POST /categorize/run`

Categorizes **every** currently-uncategorized transaction in the table
(not scoped to a recent batch), as a background job. Real, potentially
expensive if the table has a lot of uncategorized rows.

```bash
curl -X POST localhost:8000/categorize/run
```

```json
{"job_id": "8a7c3d21-...", "status": "pending"}
```

Same `job_id` pattern as ingest — check status via `GET /jobs/{job_id}`.

---

## `GET /jobs/{job_id}`

Status/result of a background job started by ingest or categorize/run.

```bash
curl localhost:8000/jobs/3f2b1a9c-...
```

```json
{
  "job_id": "3f2b1a9c-...",
  "job_type": "ingest_categorize",
  "status": "complete",
  "n_categorized": 1,
  "n_auto_applied": 1,
  "n_queued_for_review": 0,
  "error_message": null,
  "created_at": "2026-07-30T22:10:00Z",
  "completed_at": "2026-07-30T22:10:03Z"
}
```

`status` is one of `pending` / `running` / `complete` / `failed`. While
`pending`/`running`, the `n_*` fields are `null`. On failure,
`error_message` is populated instead.

Errors: `404` if `job_id` doesn't exist.

---

## `GET /forecast?horizon=30`

Projects a daily cash-flow balance curve forward `horizon` days (1–365,
default 30) from current transaction history. Synchronous — no LLM
call, just arithmetic, so no job/polling needed.

```bash
curl "localhost:8000/forecast?horizon=30"
```

```json
{
  "horizon": 30,
  "starting_balance": -5851.20,
  "lowest_point": -7114.35,
  "lowest_point_date": "2026-08-14",
  "ending_balance": -4200.10,
  "explanation": "Balance dips to a low of -$7,114.35 around 2026-08-14 as recurring bills land, then partially recovers by payroll dates.",
  "balance_curve": [
    {"date": "2026-07-31", "balance": -5851.20},
    {"date": "2026-08-01", "balance": -5920.45}
  ]
}
```

`balance_curve` has one point per day across the full horizon (truncated
above for brevity).

---

## `GET /review-queue`

Lists transactions the categorizer queued for human review instead of
auto-applying (confidence below `AUTO_APPLY_CONFIDENCE_THRESHOLD`,
currently `0.80`).

These are real unknowns, not arbitrary low scores: the synthetic dataset
deliberately includes statement descriptors that identify no merchant at
all (`SQ *MERCHANT 88213`, `POS DEBIT 4471`), with no vendor knowledge
base entry, so retrieval finds nothing and the model reports low
confidence. That is what keeps this endpoint demonstrable rather than
empty.

```bash
curl localhost:8000/review-queue
```

```json
[
  {
    "transaction_id": 2201,
    "raw_description": "SQ *NEW VENDOR",
    "posted_date": "2026-07-29",
    "amount": -15.00,
    "suggested_category": "Shopping",
    "suggested_subcategory": "online retail",
    "confidence": 0.55,
    "reasoning": "Weak retrieval context; no close vendor match found.",
    "decision_log_id": 2701
  }
]
```

Empty array `[]` means nothing is currently queued.

---

## `POST /review/{transaction_id}/approve`

Approves a queued suggestion — writes the categorizer's suggested
category/subcategory/confidence onto the transaction and logs the
approval.

```bash
curl -X POST localhost:8000/review/2201/approve
```

```json
{
  "transaction_id": 2201,
  "status": "approved",
  "category": "Shopping",
  "subcategory": "online retail"
}
```

Errors: `404` if the transaction or its categorizer decision doesn't
exist; `409` if the transaction's latest decision isn't currently
`queued_for_review` (e.g. it was already approved, or was auto-applied).

There is currently no reject/disapprove endpoint — only approve.

---

## `GET /audit/{transaction_id}`

Full decision history for one transaction — every `DecisionLog` row
about it, in chronological order (categorizer decisions, review
approvals, backfills, etc., all show up here).

```bash
curl localhost:8000/audit/376
```

```json
{
  "transaction_id": 376,
  "raw_description": "PAYPAL *AIRBNB",
  "posted_date": "2026-03-12",
  "amount": -412.00,
  "category": "Travel",
  "subcategory": "lodging",
  "decisions": [
    {
      "id": 526,
      "agent_name": "categorizer",
      "decision_type": "categorization",
      "reasoning": "The transaction is with Airbnb...",
      "confidence_score": 0.8,
      "suggested_category": "Housing",
      "suggested_subcategory": null,
      "action_taken": "queued_for_review",
      "created_at": "2026-07-30T21:05:00Z"
    },
    {
      "id": 2701,
      "agent_name": "categorizer",
      "decision_type": "categorization",
      "reasoning": "The transaction is with Airbnb, which is commonly associated with travel and lodging...",
      "confidence_score": 0.85,
      "suggested_category": "Travel",
      "suggested_subcategory": "lodging",
      "action_taken": "auto_applied",
      "created_at": "2026-07-31T02:13:07Z"
    }
  ]
}
```

Errors: `404` if the transaction doesn't exist.

---

## `GET /metrics`

Aggregate stats over real agent activity (`agent_metrics` table) — not
Prometheus-grade, just real numbers.

```bash
curl localhost:8000/metrics
```

```json
{
  "n_categorization_calls": 2515,
  "avg_categorization_latency_ms": 812.4,
  "avg_confidence": 0.87,
  "pct_auto_applied": 100.0,
  "pct_queued_for_review": 0.0,
  "n_forecast_calls": 4,
  "avg_forecast_latency_ms": 6.2
}
```

All fields are `null`/`0` (not an error) if no calls have happened yet.
Note: this only counts calls made through the live app (ingest,
categorize/run, forecast) — `scripts/run_eval.py` and the one-off
`scripts/backfill_queued_transactions.py` deliberately don't feed this
table, since it's meant to reflect real production traffic.

---

## Typical workflow

1. `POST /transactions/ingest` a new batch → get a `job_id`.
2. Poll `GET /jobs/{job_id}` until `status` is `complete` (or `failed`).
3. Check `GET /review-queue` for anything that needs a human look.
4. `POST /review/{id}/approve` for each one you're satisfied with.
5. `GET /forecast?horizon=30` to see the resulting cash-flow projection.
6. `GET /audit/{transaction_id}` any time you want the full reasoning
   trail behind a specific transaction's category.
7. `GET /metrics` periodically to sanity-check latency/confidence/
   auto-apply rate over time.
