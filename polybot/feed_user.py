from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .ws_client import WsClient, WsConfig


@dataclass
class UserFeedConfig:
    url: str


class UserFeed:
    """Authenticated user websocket feed.

    Handles fills / order updates.

    Chairman rule:
    - idempotent fill processing (dedupe)
    - no silent drops
    """

    def __init__(self, cfg: UserFeedConfig):
        self.cfg = cfg
        self.ws = WsClient(WsConfig(url=cfg.url))

    async def run(self, on_message: Callable[[dict[str, Any]], None]) -> None:
        async for msg in self.ws.messages():
            on_message(msg)
