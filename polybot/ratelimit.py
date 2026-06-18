from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Async token bucket for request budgeting.

    This is a hard requirement for Polygon RPC + CLOB REST safety.
    """

    def __init__(self, *, rate_per_s: float, capacity: float | None = None):
        self.rate = float(rate_per_s)
        self.capacity = float(capacity if capacity is not None else max(1.0, rate_per_s))
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        dt = now - self.updated
        if dt <= 0:
            return
        self.updated = now
        self.tokens = min(self.capacity, self.tokens + dt * self.rate)

    async def acquire(self, n: float = 1.0) -> None:
        n = float(n)
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= n:
                    self.tokens -= n
                    return
                deficit = n - self.tokens
                # time until deficit is refilled
                wait_s = max(0.0, deficit / self.rate) if self.rate > 0 else 1.0
            await asyncio.sleep(min(1.0, wait_s))
