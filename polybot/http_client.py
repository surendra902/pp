from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

from .ratelimit import TokenBucket
from .retry import Backoff, CircuitBreaker, retry_async


@dataclass
class HttpConfig:
    timeout_s: float = 10.0
    # Budget applies to this client instance (e.g., RPC vs CLOB REST can use separate clients).
    rate_per_s: float = 5.0
    burst: float = 10.0


class HttpClient:
    """Thin aiohttp wrapper with:
    - token bucket budgeting
    - exponential backoff on transient failures
    - fail-closed circuit breaker

    Chairman rule: if critical endpoints are unstable, we stop trading.
    This layer makes that decision measurable.
    """

    def __init__(self, cfg: HttpConfig):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None
        self._bucket = TokenBucket(rate_per_s=cfg.rate_per_s, capacity=cfg.burst)
        self._breaker = CircuitBreaker(max_failures=5, window_s=30.0, cooloff_s=60.0)

    async def __aenter__(self) -> "HttpClient":
        timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("HttpClient not started")
        return self._session

    def is_open(self) -> bool:
        return self._breaker.allow()

    async def request_json(self, method: str, url: str, *, headers: dict[str, str] | None = None, json_body: Any | None = None) -> Any:
        if not self._breaker.allow():
            raise RuntimeError(f"circuit_open: cooloff={self._breaker.remaining_cooloff_s():.1f}s")

        await self._bucket.acquire(1.0)

        async def _do() -> Any:
            async with self.session.request(method, url, headers=headers, json=json_body) as resp:
                # If we are rate-limited, we want the caller to back off.
                if resp.status in (429, 500, 502, 503, 504):
                    text = await resp.text()
                    raise RuntimeError(f"http_transient status={resp.status} url={url} body={text[:200]}")
                if resp.status >= 400:
                    text = await resp.text()
                    # Non-transient: do not retry blindly.
                    raise ValueError(f"http_error status={resp.status} url={url} body={text[:200]}")
                return await resp.json()

        try:
            out = await retry_async(_do, attempts=4, backoff=Backoff(base_s=0.25, factor=2.0, max_s=4.0))
            self._breaker.record_success()
            return out
        except Exception:
            self._breaker.record_failure()
            raise
