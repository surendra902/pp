from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class Supervisor:
    """Minimal task supervisor.

    Chairman rule: explicit tasks, explicit cancellation, no orphan loops.
    """

    tasks: set[asyncio.Task] = field(default_factory=set)

    def spawn(self, coro, *, name: str) -> None:
        t = asyncio.create_task(coro, name=name)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    async def cancel_all(self) -> None:
        for t in list(self.tasks):
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


async def run_until_cancelled() -> None:
    """Block forever until cancelled."""
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise
