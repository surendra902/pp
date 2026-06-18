from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .sdk_client import SdkClient
from .ws_client import WsClient, WsConfig


@dataclass
class UserFeedConfig:
    url: str


class UserFeed:
    """Authenticated user websocket feed.

    Extraction from bot_LIVE.py:
      - build HMAC headers for path "/ws/user"
      - send WS subscribe message {type:"User", auth:{...}, markets:[...]}.

    Forwards raw event dicts to `on_message`.
    """

    def __init__(self, cfg: UserFeedConfig, *, sdk_client: SdkClient, market_ids: list[str]):
        self.cfg = cfg
        self.sdk_client = sdk_client
        self.market_ids = market_ids

        async def _on_connect(ws):
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
            await ws.send(json.dumps(sub))

        self.ws = WsClient(WsConfig(url=cfg.url), on_connect=_on_connect)

    async def run(self, on_message: Callable[[dict[str, Any]], Any]) -> None:
        async for msg in self.ws.messages():
            msgs = msg if isinstance(msg, list) else [msg]
            for m in msgs:
                if isinstance(m, dict):
                    on_message(m)
