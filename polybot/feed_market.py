from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .book import OrderBook
from .ws_client import WsClient, WsConfig


@dataclass
class MarketFeedConfig:
    url: str


class MarketFeed:
    """Public market websocket feed.

    Extraction from `bot_LIVE.py` HyperPolyFeed._handle() with the same event
    types:
      - event_type == "book" (full snapshot)
      - event_type == "price_change" (delta)
      - event_type == "last_trade_price" (last trade)

    This feed is intentionally minimal: it only maintains per-token OrderBook
    and emits callbacks for consumers.
    """

    def __init__(self, cfg: MarketFeedConfig):
        self.cfg = cfg
        self.ws = WsClient(WsConfig(url=cfg.url))

        self.books: Dict[str, OrderBook] = {}
        self._snapshot_received: set[str] = set()

        # last trade EWMA (optional consumer use)
        self._trade_ewma: Dict[str, float] = {}
        self._trade_ts: Dict[str, float] = {}
        self.TRADE_ALPHA = 0.3
        self.TRADE_TTL_S = 30.0

        self._callbacks: list[Callable[[str, OrderBook], Any]] = []

    def on_update(self, cb: Callable[[str, OrderBook], Any]) -> None:
        self._callbacks.append(cb)

    def get_book(self, token_id: str) -> OrderBook:
        if token_id not in self.books:
            self.books[token_id] = OrderBook(token_id=token_id)
        return self.books[token_id]

    def last_trade(self, token_id: str) -> Optional[float]:
        ts = self._trade_ts.get(token_id)
        if not ts:
            return None
        if time.monotonic() - ts > self.TRADE_TTL_S:
            return None
        return self._trade_ewma.get(token_id)

    async def run(self) -> None:
        async for msg in self.ws.messages():
            # bot_LIVE.py sometimes emits a list of messages
            msgs = msg if isinstance(msg, list) else [msg]
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                tid = str(m.get("asset_id") or "")
                if not tid:
                    continue
                bk = self.get_book(tid)

                et = str(m.get("event_type") or "")
                if et == "book":
                    bids = []
                    for x in m.get("bids", []) or []:
                        try:
                            p = float(x["price"])
                            s = float(x["size"])
                        except Exception:
                            continue
                        if s > 0:
                            bids.append((p, s))
                    asks = []
                    for x in m.get("asks", []) or []:
                        try:
                            p = float(x["price"])
                            s = float(x["size"])
                        except Exception:
                            continue
                        if s > 0:
                            asks.append((p, s))
                    # Our polybot.OrderBook does not currently expose
                    # replace_snapshot; we approximate via deltas:
                    # clear then apply.
                    bk._bids.clear(); bk._asks.clear()
                    bk._bid_total = 0; bk._ask_total = 0
                    bk._best_bid = None; bk._best_ask = None
                    for p, s in bids:
                        bk.apply_delta(p, s, is_bid=True)
                    for p, s in asks:
                        bk.apply_delta(p, s, is_bid=False)
                    bk.ts = time.monotonic()
                    self._snapshot_received.add(tid)

                elif et == "price_change":
                    if tid not in self._snapshot_received:
                        continue
                    delta_ts = time.monotonic()
                    for c in m.get("changes", []) or []:
                        try:
                            side = str(c.get("side", "")).upper()
                            p = float(c["price"])
                            s = float(c["size"])
                        except Exception:
                            continue
                        bk.apply_delta(p, s, is_bid=(side == "BID"))
                    bk.ts = delta_ts

                elif et == "last_trade_price":
                    try:
                        price = float(m["price"])
                    except Exception:
                        continue
                    if 0 < price < 1:
                        prev = self._trade_ewma.get(tid)
                        self._trade_ewma[tid] = (
                            self.TRADE_ALPHA * price + (1.0 - self.TRADE_ALPHA) * prev
                            if prev is not None else price
                        )
                        self._trade_ts[tid] = time.monotonic()

                else:
                    continue

                # Fire callbacks without blocking WS parser loop
                for cb in self._callbacks:
                    try:
                        res = cb(tid, bk)
                        if asyncio.iscoroutine(res):
                            asyncio.create_task(res)
                    except Exception:
                        continue
