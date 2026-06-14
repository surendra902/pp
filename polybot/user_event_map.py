from __future__ import annotations

from typing import Any, Optional


def extract_trade_id(m: dict[str, Any]) -> str:
    return str(m.get("trade_id") or m.get("id") or m.get("match_id") or "")


def extract_token_id(m: dict[str, Any]) -> str:
    return str(m.get("asset_id") or m.get("token_id") or "")


def extract_side(m: dict[str, Any]) -> str:
    return str(m.get("side") or "").upper()


def extract_price(m: dict[str, Any]) -> Optional[float]:
    try:
        p = float(m.get("price") or 0)
    except Exception:
        return None
    return p if p > 0 else None


def extract_size(m: dict[str, Any]) -> Optional[float]:
    # user WS in bot_LIVE.py uses size/quantity fields
    for k in ("size", "quantity"):
        if k in m:
            try:
                s = float(m.get(k) or 0)
            except Exception:
                continue
            return s if s > 0 else None
    return None


def extract_ts(m: dict[str, Any]) -> int:
    # WS events often omit ts; reconcile cursor can advance on REST.
    # If present, prefer numeric seconds.
    for k in ("match_time", "timestamp", "ts"):
        if k in m:
            try:
                v = float(m.get(k) or 0)
            except Exception:
                continue
            # normalize ms/us
            while v > 1e11:
                v /= 1000.0
            return int(v)
    return 0
