from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import websockets

from .retry import Backoff


@dataclass
class WsConfig:
    url: str
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0
    max_stale_s: float = 60.0


OnConnectFn = Callable[[websockets.WebSocketClientProtocol], Awaitable[None]]


class WsClient:
    """Minimal websocket client with reconnect + stale detection.

    Supports an optional `on_connect` coroutine that can send subscription/auth
    messages on every (re)connect.
    """

    def __init__(self, cfg: WsConfig, *, on_connect: Optional[OnConnectFn] = None):
        self.cfg = cfg
        self._last_msg_mono: float = 0.0
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._on_connect = on_connect

    @property
    def last_msg_age_s(self) -> float:
        if self._last_msg_mono <= 0:
            return 1e9
        return time.monotonic() - self._last_msg_mono

    def is_effectively_up(self) -> bool:
        return self._ws is not None and self.last_msg_age_s < self.cfg.max_stale_s

    async def messages(self) -> AsyncIterator[dict[str, Any] | list[Any]]:
        backoff = Backoff(base_s=0.5, factor=2.0, max_s=15.0, jitter=0.25)
        attempt = 0
        while True:
            try:
                async with websockets.connect(
                    self.cfg.url,
                    ping_interval=self.cfg.ping_interval_s,
                    ping_timeout=self.cfg.ping_timeout_s,
                    close_timeout=5,
                    max_queue=1024,
                ) as ws:
                    self._ws = ws
                    attempt = 0
                    # (re)subscribe/auth
                    if self._on_connect is not None:
                        await self._on_connect(ws)

                    async for raw in ws:
                        self._last_msg_mono = time.monotonic()
                        try:
                            yield json.loads(raw)
                        except Exception:
                            continue
            except asyncio.CancelledError:
                raise
            except Exception:
                self._ws = None
                await asyncio.sleep(backoff.delay(attempt))
                attempt += 1
