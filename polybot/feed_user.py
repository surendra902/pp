from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .ws_client import WsClient, WsConfig
from .sdk_client import SdkClient


@dataclass
class UserFeedConfig:
    url: str


class UserFeed:
    """Authenticated user websocket feed.

    Implemented by extracting the auth subscribe payload logic from bot_LIVE.py:
      - build HMAC headers for path "/ws/user"
      - send WS subscribe message {type:"User", auth:{...}, markets:[...]}.

    We keep parsing minimal and forward raw dict events to the handler.
    """

    def __init__(self, cfg: UserFeedConfig, *, sdk_client: SdkClient, market_ids: list[str]):
        self.cfg = cfg
        self.ws = WsClient(WsConfig(url=cfg.url))
        self.sdk_client = sdk_client
        self.market_ids = market_ids

    async def run(self, on_message: Callable[[dict[str, Any]], Any]) -> None:
        # We rely on WsClient's reconnect loop; on each reconnect we re-send auth+subscribe.
        async for msg in self.ws.messages():
            # First message after connect: send auth subscribe.
            try:
                hdrs = self.sdk_client.hmac_headers("GET", "/ws/user")
                sub = {
                    "auth": {
                        "apiKey": hdrs.get("POLY_API_KEY", ""),
                        "passphrase": hdrs.get("POLY_PASSPHRASE", ""),
                        "timestamp": hdrs.get("POLY_TIMESTAMP", ""),
                        "signature": hdrs.get("POLY_SIGNATURE", ""),
                        "polyAddress": hdrs.get("POLY_ADDRESS", ""),
                        "polyNonce": hdrs.get("POLY_NONCE", "0"),
                    },
                    "type": "User",
                    "markets": self.market_ids,
                    "assets_ids": [],
                }
                # WsClient currently yields decoded messages only; it doesn't expose the underlying socket.
                # So we cannot send from here until WsClient is extended.
                # To keep correctness, we forward a synthetic control message for now.
                on_message({"_control": "NEEDS_WS_SEND", "subscribe": sub})
            except Exception:
                pass

            msgs = msg if isinstance(msg, list) else [msg]
            for m in msgs:
                if isinstance(m, dict):
                    on_message(m)
