# cashflow-agent

A FastAPI service for categorizing spending transactions and forecasting cash
flow, backed by LangChain agents and a Chroma vector store for retrieval-augmented
vendor categorization.

This is currently a scaffold: routers, agents, and the RAG layer are stubbed out
with docstrings describing their intended purpose, and no business logic has been
implemented yet. The app boots and serves a `/health` check.

## Structure

```
app/
  main.py              # FastAPI app entrypoint
  config.py            # settings via pydantic-settings, reads .env
  db/
    models.py          # SQLAlchemy models
    session.py         # async engine + session factory
  agents/
    dispatcher.py       # routes intents; owns bulk categorization
    categorizer.py       # classifies transactions into categories
    forecaster.py         # projects future cash flow
  rag/
    vector_store.py     # Chroma vector store client
    vendor_kb.py          # curated vendor -> category mappings
  schemas.py           # Pydantic request/response models
  routers/
    transactions.py
    categorize.py
    forecast.py
scripts/
  generate_synthetic_data.py   # fake transactions for local dev
tests/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

Then check `curl localhost:8000/health` -> `{"status": "ok"}`.

## Test

```bash
pytest
```
