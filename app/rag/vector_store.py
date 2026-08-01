"""Vector store client.

Wraps two local, on-disk Chroma collections (persisted at
`settings.chroma_persist_dir`, default `./chroma_data`), both embedded
with `settings.openai_embedding_model` (default OpenAI's
`text-embedding-3-small`):

  - `vendor_kb`: one vector per `VendorKB` row, so a raw transaction
    description can be matched against known vendors and their
    canonical categories.
  - `categorized_transactions`: one vector per already-categorized
    `Transaction` row, so the categorizer agent can also ground on
    "transactions like this one I've already seen," not just the seed
    knowledge base.

`build_vendor_index()` / `build_transaction_index()` (re)populate those
collections from the current database state -- upserts are keyed by row
id, so re-running them after new data lands is safe. `retrieve_similar()`
/ `retrieve_similar_transactions()` embed a query string (e.g. a raw
transaction description) and return the top-k nearest neighbors with
their categories and a similarity score.
"""

import asyncio
import threading
from functools import lru_cache
from typing import Sequence

import chromadb
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import select

from app.config import settings
from app.db.models import Transaction, VendorKB
from app.db.session import async_session_factory

VENDOR_COLLECTION_NAME = "vendor_kb"
TRANSACTION_COLLECTION_NAME = "categorized_transactions"


class EmptyIndexError(RuntimeError):
    """Raised when a collection that must be populated is empty.

    An empty Chroma collection is otherwise silent: `get_or_create_collection`
    happily fabricates one, `_query` returns `[]`, and the categorizer goes on
    to prompt the LLM with "(no similar known vendors found)". The model still
    answers, often confidently, so the confidence guardrail does NOT catch it --
    it measures the model's uncertainty, not whether retrieval actually worked.
    Failing loudly at startup is the only thing that turns this into a visible
    problem.
    """


# Guards lazy construction of the Chroma client and collection handles.
#
# `functools.lru_cache` is safe against cache corruption but does NOT hold a
# lock while the wrapped function runs -- on a cold cache, N threads all
# execute the body concurrently and only one result gets stored. That is
# harmless for a cheap pure function and fatal here: the categorizer fans
# out across worker threads (see CATEGORIZATION_CONCURRENCY in
# app/agents/dispatcher.py), so 8 threads each tried to build a
# PersistentClient against the same directory at once and raced on tenant
# initialization ("Could not connect to tenant default_tenant").
#
# Serializing construction costs one uncontended lock acquire per call --
# nanoseconds against a network round trip -- and removes the race.
# `get_or_create_collection` is guarded for the same reason: concurrent
# create attempts on one collection name are the same hazard.
_chroma_lock = threading.Lock()


@lru_cache
def _build_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_client() -> chromadb.ClientAPI:
    """Memoized Chroma client -- one instance per process, safe under threads."""
    with _chroma_lock:
        return _build_client()


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """Memoized OpenAIEmbeddings client -- one instance per process, no `global` mutation."""
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
        max_retries=settings.openai_max_retries,
    )


def _get_collection(name: str):
    # cosine distance so `1 - distance` below is a similarity in [-1, 1]
    with _chroma_lock:
        return _build_client().get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


async def build_vendor_index() -> int:
    """Embed every VendorKB row and upsert into the vendor_kb collection."""
    async with async_session_factory() as session:
        rows = (await session.execute(select(VendorKB))).scalars().all()

    if not rows:
        return 0

    documents = [f"{row.vendor_name}: {row.description}" for row in rows]
    embeddings = get_embeddings().embed_documents(documents)

    _get_collection(VENDOR_COLLECTION_NAME).upsert(
        ids=[str(row.id) for row in rows],
        embeddings=embeddings,
        documents=documents,
        metadatas=[
            {
                "vendor_name": row.vendor_name,
                "canonical_category": row.canonical_category,
                "canonical_subcategory": row.canonical_subcategory or "",
            }
            for row in rows
        ],
    )
    return len(rows)


async def build_transaction_index() -> int:
    """Embed already-categorized Transaction rows and upsert into the categorized_transactions collection."""
    async with async_session_factory() as session:
        stmt = select(Transaction).where(Transaction.category.is_not(None))
        rows = (await session.execute(stmt)).scalars().all()

    if not rows:
        return 0

    documents = [row.raw_description for row in rows]
    embeddings = get_embeddings().embed_documents(documents)

    _get_collection(TRANSACTION_COLLECTION_NAME).upsert(
        ids=[str(row.id) for row in rows],
        embeddings=embeddings,
        documents=documents,
        metadatas=[
            {
                "vendor_name": row.vendor_name or "",
                "category": row.category or "",
            }
            for row in rows
        ],
    )
    return len(rows)


def verify_vendor_index() -> int:
    """Assert the vendor collection is populated; raise EmptyIndexError if not.

    Called from the app's startup lifespan so an empty/missing
    `chroma_persist_dir` (fresh container, ephemeral filesystem, wrong working
    directory -- the path is relative) fails the boot instead of silently
    degrading every categorization to an ungrounded LLM guess.

    Only the vendor collection is checked. It is seeded and should never be
    empty, so zero is unambiguously broken. `categorized_transactions` is
    legitimately empty on a fresh database, so it can't be asserted the same
    way.

    Deliberately does NOT rebuild the index itself: that would make startup
    depend on the OpenAI API being reachable, turning a third-party outage into
    a crash loop. A clear error naming the fix is better than a boot that hangs
    on someone else's availability.
    """
    count = _get_collection(VENDOR_COLLECTION_NAME).count()
    if count == 0:
        raise EmptyIndexError(
            f"Vendor index at {settings.chroma_persist_dir!r} is empty -- retrieval would "
            f"silently return nothing and every categorization would be ungrounded. "
            f"Rebuild it with: python -m scripts.build_indexes"
        )
    return count


async def index_transactions(transaction_ids: Sequence[int]) -> int:
    """Embed and upsert only the given transactions -- the incremental path.

    `build_transaction_index()` re-embeds every categorized row in the table,
    which is correct for a one-off rebuild and far too expensive to run after
    each ingest batch (ingesting 10 rows into a 2,500-row table would re-embed
    all 2,510). This embeds just the rows named, so the cost scales with the
    batch instead of the corpus.

    Rows without a category are skipped -- a transaction still queued for
    review has no confirmed label to teach the retriever.
    """
    if not transaction_ids:
        return 0

    async with async_session_factory() as session:
        stmt = select(Transaction).where(
            Transaction.id.in_(transaction_ids), Transaction.category.is_not(None)
        )
        rows = (await session.execute(stmt)).scalars().all()

    if not rows:
        return 0

    documents = [row.raw_description for row in rows]
    # embed_documents is a blocking network call -- off the event loop, same
    # reasoning as the categorizer's asyncio.to_thread wrapper.
    embeddings = await asyncio.to_thread(get_embeddings().embed_documents, documents)

    _get_collection(TRANSACTION_COLLECTION_NAME).upsert(
        ids=[str(row.id) for row in rows],
        embeddings=embeddings,
        documents=documents,
        metadatas=[
            {"vendor_name": row.vendor_name or "", "category": row.category or ""} for row in rows
        ],
    )
    return len(rows)


def _query(collection_name: str, query_text: str, k: int) -> list[dict]:
    collection = _get_collection(collection_name)
    if collection.count() == 0:
        return []

    query_embedding = get_embeddings().embed_query(query_text)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    hits = []
    for id_, distance, metadata, document in zip(
        results["ids"][0], results["distances"][0], results["metadatas"][0], results["documents"][0]
    ):
        hits.append({"id": id_, "document": document, "similarity": 1 - distance, **metadata})
    return hits


def retrieve_similar(query_text: str, k: int = 5) -> list[dict]:
    """Top-k VendorKB entries most similar to query_text, with categories and similarity scores."""
    return _query(VENDOR_COLLECTION_NAME, query_text, k)


def retrieve_similar_transactions(query_text: str, k: int = 3) -> list[dict]:
    """Top-k already-categorized transactions most similar to query_text."""
    return _query(TRANSACTION_COLLECTION_NAME, query_text, k)
