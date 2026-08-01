"""Builds (or rebuilds) the two Chroma vector indexes used for RAG-assisted
categorization: the vendor knowledge base and already-categorized
transactions. See `app/rag/vector_store.py` for what gets embedded and how.

Upserts are keyed by row id, so this is safe to re-run any time new
VendorKB entries or newly-categorized transactions land in the database --
it will just add/update the relevant vectors, not duplicate them.

Requires a real OPENAI_API_KEY in `.env` (embedding calls hit the OpenAI
API).

Run with: python -m scripts.build_indexes
"""

import asyncio

from app.rag.vector_store import build_transaction_index, build_vendor_index


async def main() -> None:
    vendor_count = await build_vendor_index()
    transaction_count = await build_transaction_index()
    print(f"Indexed {vendor_count} vendor_kb vectors and {transaction_count} categorized-transaction vectors.")


if __name__ == "__main__":
    asyncio.run(main())
