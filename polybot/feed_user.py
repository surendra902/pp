from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from .ws_client import WsClient, WsConfig


@dataclass
class UserFeedConfig:
    url: str


class UserFeed:
    """Authenticated user websocket feed.

    Extraction target: `bot_LIVE.py` UserFeed.run() parsing.

    This module only parses message envelopes and forwards raw dict payloads
    to `on_message`.
    """

    def __init__(self, cfg: UserFeedConfig):
        self.cfg = cfg
        self.ws = WsClient(WsConfig(url=cfg.url))

    async def run(self, on_message: Callable[[dict[str, Any]], Any]) -> None:
        async for msg in self.ws.messages():
            msgs = msg if isinstance(msg, list) else [msg]
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                on_message(m)
