"""Application settings.

Defines a pydantic-settings `Settings` class that reads configuration from
environment variables and a local `.env` file (see `.env.example`). Other
modules import the shared `settings` instance from here instead of reading
`os.environ` directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    database_url: str = "sqlite+aiosqlite:///./cashflow.db"
    chroma_persist_dir: str = "./chroma_data"

    # How many transactions the bulk categorizer processes in parallel.
    #
    # The binding constraint is the OpenAI org's tokens-per-minute limit,
    # not local resources. Worked example for the 30,000 TPM tier this was
    # developed against: one categorization costs ~440 tokens (~390 in,
    # ~50 out), so the ceiling is 30000/440 = ~68 calls/min. At ~1.3s per
    # call one worker sustains 60/1.3 = ~46 calls/min, so the limit is
    # reached at roughly 1.5 workers.
    #
    # 2 therefore runs slightly hot on purpose: it saturates the quota,
    # and openai_max_retries below absorbs the 429s with exponential
    # backoff, which paces the run to the ceiling. Rejected requests are
    # not billed, so the cost of overshooting is latency, not money. Set
    # to 1 to stay strictly under the limit and never see a 429; raise it
    # well above 2 on a higher tier.
    categorization_concurrency: int = 2

    # Retries for rate-limited/transient OpenAI errors. The SDK applies
    # exponential backoff between attempts, which is what actually keeps
    # a long bulk run under a TPM ceiling instead of failing the whole
    # batch on the first 429.
    openai_max_retries: int = 8

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
