"""Structured logging.

One JSON line per agent call to stdout: timestamp, level, logger name,
message, plus whatever event-specific fields the call site passes. Uses
stdlib `logging` with a small custom formatter -- no structlog dependency,
since flat structured lines with no context-binding chains don't need it.

`GET /metrics` (app/routers/metrics.py) does NOT read these logs -- it
aggregates a separate, lighter DB-backed `AgentMetric` table
(app/db/models.py) written by callers that already own an async session.
These logs are for humans/log tooling; the DB table is for the endpoint.
"""

import json
import logging
import sys
import warnings
from datetime import datetime, timezone

# LangChain's `.with_structured_output(..., include_raw=True)` returns a
# wrapper whose `parsed` field is declared Optional but is populated on
# success, which trips a Pydantic serializer warning on EVERY LLM call.
# It is cosmetic -- the parsed object is correct, and categorizer.py
# raises explicitly when it really is None -- but at one warning per
# transaction it buries genuine output during a bulk run. Suppressed here
# rather than at each call site because this module already owns what the
# process writes to stderr/stdout.
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    logger.info(event, extra={"fields": fields})
