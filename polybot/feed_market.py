from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .book import OrderBook
from .ws_client import WsClient, WsConfig


@dataclass
class MarketFeedConfig:
    url: str


class MarketFeed:
    """Public market websocket feed.

    Consumes raw WS messages and updates in-memory order books.

    Chairman rule:
    - treat stale-data WS as down
    - isolate parsing errors
    - never mutate books from unknown payload shapes
    """

    def __init__(self, cfg: MarketFeedConfig):
        self.cfg = cfg
        self.ws = WsClient(WsConfig(url=cfg.url))
        self.books: Dict[str, OrderBook] = {}

    def get_book(self, token_id: str) -> OrderBook:
        if token_id not in self.books:
            self.books[token_id] = OrderBook(token_id=token_id)
        return self.books[token_id]

    async def run(self, on_message: Optional[Callable[[dict[str, Any]], None]] = None) -> None:
        async for msg in self.ws.messages():
            if on_message:
                on_message(msg)
            # TODO: Extract exact message schema handling from bot_LIVE.py
            # For now, keep the spine only.
            # Example structure in bot_LIVE.py: price_change deltas that call book.apply_delta(...)
            _ = msg
