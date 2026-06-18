from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple


class LegState(str, Enum):
    PENDING = "PENDING"      # order submitted, not confirmed
    LIVE = "LIVE"            # resting/working
    PARTIAL = "PARTIAL"      # partially filled
    FILLED = "FILLED"        # fully filled
    CLOSED = "CLOSED"        # position flattened / resolved
    UNKNOWN = "UNKNOWN"      # invariant broken -> must halt


@dataclass
class FillDeduper:
    """Idempotency set for fills/trades."""

    seen: Set[str] = field(default_factory=set)

    def is_new(self, trade_id: str) -> bool:
        if trade_id in self.seen:
            return False
        self.seen.add(trade_id)
        return True


@dataclass
class ReconcileCursor:
    """Monotonic cursor for REST trade walks."""

    last_ts: int = 0

    def advance(self, ts: int) -> None:
        if ts > self.last_ts:
            self.last_ts = ts


@dataclass
class Reconciler:
    """Spine of reconciliation engine.

    This file intentionally defines *state + invariants* first.
    Then we can extract bot_LIVE.py's exact REST/WS parsing into here.
    """

    cursor: ReconcileCursor = field(default_factory=ReconcileCursor)
    dedupe: FillDeduper = field(default_factory=FillDeduper)

    # market_id+token -> state
    states: Dict[Tuple[str, str], LegState] = field(default_factory=dict)

    halted: bool = False
    halt_reason: str = ""

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def on_ws_fill(self, *, market_id: str, token_id: str, trade_id: str, ts: int, payload: dict[str, Any]) -> None:
        if not self.dedupe.is_new(trade_id):
            return
        self.cursor.advance(ts)
        key = (market_id, token_id)
        st = self.states.get(key, LegState.PENDING)
        # minimal transition: any fill implies LIVE->PARTIAL or ->FILLED
        if st in (LegState.UNKNOWN, LegState.CLOSED):
            self.halt(f"fill_in_invalid_state key={key} state={st}")
            return
        self.states[key] = LegState.PARTIAL

    def on_ws_order_update(self, *, market_id: str, token_id: str, order_id: str, status: str, payload: dict[str, Any]) -> None:
        key = (market_id, token_id)
        st = self.states.get(key, LegState.PENDING)
        if st == LegState.UNKNOWN:
            return
        # TODO: map exact statuses from bot_LIVE.py
        if status.lower() in ("open", "live"):
            self.states[key] = LegState.LIVE
        elif status.lower() in ("filled", "matched"):
            self.states[key] = LegState.FILLED

    def on_rest_trade(self, *, market_id: str, token_id: str, trade_id: str, ts: int, payload: dict[str, Any]) -> None:
        # same idempotency as WS
        self.on_ws_fill(market_id=market_id, token_id=token_id, trade_id=trade_id, ts=ts, payload=payload)
