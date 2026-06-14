from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Type


@dataclass
class Backoff:
    base_s: float = 0.25
    factor: float = 2.0
    max_s: float = 10.0
    jitter: float = 0.20  # +/- 20%

    def delay(self, attempt: int) -> float:
        d = min(self.max_s, self.base_s * (self.factor ** max(0, attempt)))
        j = d * self.jitter
        return max(0.0, d + random.uniform(-j, +j))


async def retry_async(
    fn: Callable[[], Awaitable[object]],
    *,
    attempts: int,
    backoff: Backoff,
    retry_on: tuple[Type[BaseException], ...] = (Exception,),
) -> object:
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return await fn()
        except retry_on as e:
            last = e
            if i == attempts - 1:
                raise
            await asyncio.sleep(backoff.delay(i))
    raise last or RuntimeError("retry_async: exhausted")


class CircuitBreaker:
    """Fail-closed circuit breaker.

    Once too many failures occur within the window, the breaker opens and
    blocks calls until cooloff elapses.
    """

    def __init__(self, *, max_failures: int = 5, window_s: float = 30.0, cooloff_s: float = 60.0):
        self.max_failures = max_failures
        self.window_s = window_s
        self.cooloff_s = cooloff_s
        self._fails: list[float] = []
        self._open_until: float = 0.0

    def allow(self) -> bool:
        now = time.monotonic()
        if now < self._open_until:
            return False
        # prune old failures
        cutoff = now - self.window_s
        self._fails = [t for t in self._fails if t >= cutoff]
        return True

    def record_success(self) -> None:
        # On success we do not clear history entirely; just let it decay.
        pass

    def record_failure(self) -> None:
        now = time.monotonic()
        self._fails.append(now)
        cutoff = now - self.window_s
        self._fails = [t for t in self._fails if t >= cutoff]
        if len(self._fails) >= self.max_failures:
            self._open_until = now + self.cooloff_s

    def remaining_cooloff_s(self) -> float:
        return max(0.0, self._open_until - time.monotonic())
