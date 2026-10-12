"""
backend/core/rate_limit.py — in-memory request-rate limiting.

Scoped to the check-in endpoint specifically: that's the only route that
triggers a real LLM call (check_in_and_triage -> the triage pipeline), so
it's the only one an API-level abuse script can use to run up real spend.
Everything else in encounters.py is a cheap DB read/write — rate limiting
them would just be operational friction for no cost-control benefit.

In-memory, not Redis-backed, on purpose. This matches how the app actually
runs, not some hypothetical scaled-out deployment: Render's free web
service defaults to WEB_CONCURRENCY=1 (confirmed in a real deploy log), one
process, one copy of this state. A multi-instance deployment would need a
shared store instead — that's a real future requirement, not something to
fake with a dependency this app doesn't need yet.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from fastapi import HTTPException, Request, status

from backend.core.config import CHECK_IN_RATE_LIMIT, CHECK_IN_RATE_LIMIT_WINDOW_SECONDS


@dataclass
class _Bucket:
    hits: list[float] = field(default_factory=list)


class RateLimiter:
    """Sliding-window limiter: at most `limit` calls per `window_seconds`, per key."""

    def __init__(self, limit: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._buckets: dict[str, _Bucket] = defaultdict(_Bucket)

    def allow(self, key: str) -> bool:
        now = self._clock()
        bucket = self._buckets[key]
        cutoff = now - self._window
        bucket.hits = [hit for hit in bucket.hits if hit > cutoff]
        if len(bucket.hits) >= self._limit:
            return False
        bucket.hits.append(now)
        return True

    def reset(self) -> None:
        """Test-only: clears all state. Production never needs this."""
        self._buckets.clear()


check_in_limiter = RateLimiter(CHECK_IN_RATE_LIMIT, CHECK_IN_RATE_LIMIT_WINDOW_SECONDS)


def _client_ip(request: Request) -> str:
    """
    Render (like most PaaS) terminates the connection at a reverse proxy, so
    request.client.host alone would be the proxy's address for every
    visitor, not theirs — collapsing everyone into one shared bucket.
    X-Forwarded-For's first entry is the original client when present.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_check_in_rate_limit(request: Request) -> None:
    key = _client_ip(request)
    if not check_in_limiter.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Too many check-ins from this address — limit is "
                f"{CHECK_IN_RATE_LIMIT} per {int(CHECK_IN_RATE_LIMIT_WINDOW_SECONDS)}s. "
                "Try again shortly."
            ),
        )
