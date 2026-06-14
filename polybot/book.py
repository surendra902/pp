from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import heapq
import math
import time


@dataclass
class OrderBook:
    # Keep the same scale as bot_LIVE.py
    PRICE_SCALE: int = 10_000
    SIZE_SCALE: int = 1_000_000

    token_id: str = ""
    _bids: Dict[int, int] = field(default_factory=dict)
    _asks: Dict[int, int] = field(default_factory=dict)
    ts: float = field(default_factory=time.monotonic)

    _bid_total: int = 0
    _ask_total: int = 0
    _best_bid: Optional[int] = None
    _best_ask: Optional[int] = None

    def age_ms(self) -> float:
        return (time.monotonic() - self.ts) * 1000.0

    def is_stale(self, max_ms: float) -> bool:
        return self.age_ms() > max_ms

    @classmethod
    def _pkey(cls, p: float) -> int:
        return int(round(p * cls.PRICE_SCALE))

    @classmethod
    def _sint(cls, s: float) -> int:
        return int(round(s * cls.SIZE_SCALE))

    def apply_delta(self, price: float, size: float, is_bid: bool) -> None:
        if not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0):
            return
        if not (isinstance(size, (int, float)) and math.isfinite(size)):
            return

        key = self._pkey(price)
        new = self._sint(size) if size > 0 else 0
        book = self._bids if is_bid else self._asks
        old = book.get(key, 0)

        if new <= 0:
            if key in book:
                del book[key]
            new = 0
        else:
            book[key] = new

        delta = new - old

        if is_bid:
            self._bid_total += delta
            if new == 0 and self._best_bid == key:
                self._best_bid = max(self._bids) if self._bids else None
            elif new > 0 and (self._best_bid is None or key > self._best_bid):
                self._best_bid = key
        else:
            self._ask_total += delta
            if new == 0 and self._best_ask == key:
                self._best_ask = min(self._asks) if self._asks else None
            elif new > 0 and (self._best_ask is None or key < self._best_ask):
                self._best_ask = key

    def best_bid(self) -> Optional[float]:
        return None if self._best_bid is None else self._best_bid / self.PRICE_SCALE

    def best_ask(self) -> Optional[float]:
        return None if self._best_ask is None else self._best_ask / self.PRICE_SCALE

    def micro_price(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if not self._bids or not self._asks:
            if bb is None:
                return ba
            if ba is None:
                return bb
            return (bb + ba) / 2

        inv_p = 1.0 / self.PRICE_SCALE
        inv_s = 1.0 / self.SIZE_SCALE
        bkeys = heapq.nlargest(3, self._bids.keys())
        akeys = heapq.nsmallest(3, self._asks.keys())
        bid_levels = [(k * inv_p, self._bids[k] * inv_s) for k in bkeys]
        ask_levels = [(k * inv_p, self._asks[k] * inv_s) for k in akeys]
        bv = sum(s for _, s in bid_levels)
        av = sum(s for _, s in ask_levels)
        if bv <= 0 or av <= 0:
            return (bb + ba) / 2 if (bb is not None and ba is not None) else None
        bvw = sum(p * s for p, s in bid_levels) / bv
        avw = sum(p * s for p, s in ask_levels) / av
        return (bvw * av + avw * bv) / (bv + av)

    def imbalance(self) -> float:
        total = self._bid_total + self._ask_total
        return self._bid_total / total if total > 0 else 0.5
