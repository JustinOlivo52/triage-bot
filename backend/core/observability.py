"""
backend/core/observability.py — structured logging and optional error tracking.

Kept out of backend/core/config.py on purpose: that module is pure env-var
reads plus the one normalization DATABASE_URL needs. This module has real
side effects (mutating logging's global state, initializing a third-party
SDK) that need to run once at process startup, not whenever a settings
value happens to get imported.
"""

import json
import logging
import sys

from backend.core.config import DEMO_MODE, LOG_LEVEL, SENTRY_DSN


class JsonFormatter(logging.Formatter):
    """
    Minimal JSON-lines formatter — stdlib only, no dependency pulled in for
    something logging.Formatter already gives the pieces for. Render (and
    most log viewers/aggregators) read JSON lines directly; this replaces
    one unstructured text blob per line with fields a query can filter on.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    """
    Replaces the root logger's handlers with a single JSON stdout handler.
    Idempotent — safe to call more than once (tests re-invoke it under
    different LOG_LEVEL values via importlib.reload).
    """
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]


def init_error_tracking() -> None:
    """
    No-op with no SENTRY_DSN set — error tracking is an upgrade, not a
    requirement to run, same as retrieval degrading to lexical search with
    no Voyage key. Imported lazily so a backend that never sets SENTRY_DSN
    never pays for more than the import.
    """
    if not SENTRY_DSN:
        return

    import sentry_sdk

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment="demo" if DEMO_MODE else "production",
        # Error capture only — no paid performance tracing/sampling.
        traces_sample_rate=0.0,
    )
