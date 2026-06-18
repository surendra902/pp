from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .logging_setup import get_logger
from .sdk_client import SdkClient


log = get_logger("clob")


@dataclass
class OrderRequest:
    token_id: str
    side: str  # "BUY" | "SELL"
    price: float
    size: float  # shares (SDK expects shares)
    order_type: str = "GTC"  # "GTC" | "FOK"
    neg_risk: bool = False
    tick_size: float = 0.01


class ClobApi:
    """SDK-backed execution wrapper (non-hallucinated schema).

    This fixes the main missing piece of the refactor branch: the ability to
    place/cancel orders using the venue-supported SDK.

    NOTE: market discovery/strategy are still not fully migrated; this module
    only supplies execution.
    """

    def __init__(self, sdk_client: SdkClient):
        self.sdk_client = sdk_client

    async def place_order(self, req: OrderRequest) -> Optional[str]:
        sdk = self.sdk_client.sdk
        try:
            from py_clob_client_v2.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions  # type: ignore
            from py_clob_client_v2.order_builder.constants import BUY as SDK_BUY, SELL as SDK_SELL  # type: ignore
        except Exception as e:
            raise RuntimeError("py-clob-client-v2 required for execution") from e

        sdk_side = SDK_BUY if req.side.upper() == "BUY" else SDK_SELL
        args = OrderArgs(token_id=req.token_id, price=req.price, size=req.size, side=sdk_side)
        try:
            opts = PartialCreateOrderOptions(neg_risk=req.neg_risk, tick_size=str(req.tick_size))
        except TypeError:
            opts = PartialCreateOrderOptions(neg_risk=req.neg_risk)

        loop = __import__("asyncio").get_running_loop()
        signed = await loop.run_in_executor(None, lambda: sdk.create_order(args, opts))
        ot = OrderType.FOK if req.order_type.upper() == "FOK" else OrderType.GTC
        resp = await loop.run_in_executor(None, lambda: sdk.post_order(signed, ot))
        oid = (resp or {}).get("orderID")
        return oid

    async def cancel_order(self, order_id: str) -> bool:
        sdk = self.sdk_client.sdk
        loop = __import__("asyncio").get_running_loop()
        cancel_one = getattr(sdk, "cancel_order", None)
        try:
            if cancel_one:
                payload = type("P", (), {"orderID": order_id})
                await loop.run_in_executor(None, lambda: cancel_one(payload))
            else:
                await loop.run_in_executor(None, lambda: sdk.cancel(order_id))
            return True
        except Exception:
            return False

    async def cancel_all(self) -> None:
        sdk = self.sdk_client.sdk
        loop = __import__("asyncio").get_running_loop()
        try:
            await loop.run_in_executor(None, sdk.cancel_all)
        except Exception:
            pass
