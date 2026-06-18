from __future__ import annotations

import asyncio
import sys

from polybot.config import Config
from polybot.logging_setup import get_logger
from polybot.protocol import WS_MARKET_DEFAULT, WS_USER_DEFAULT
from polybot.runtime import Supervisor, run_until_cancelled
from polybot.feed_market import MarketFeed, MarketFeedConfig
from polybot.feed_user import UserFeed, UserFeedConfig
from polybot.reconcile import Reconciler
from polybot.sdk_client import SdkClient
from polybot.user_event_map import (
    extract_price,
    extract_side,
    extract_size,
    extract_token_id,
    extract_trade_id,
    extract_ts,
)


log = get_logger("main")


async def async_main(cfg: Config) -> None:
    sup = Supervisor()

    # SDK init (needed for user WS auth + later execution)
    sdk = SdkClient(
        clob_url=cfg.clob_url,
        chain_id=cfg.chain_id,
        private_key=cfg.private_key,
        signature_type=cfg.signature_type,
        proxy_address=cfg.proxy_address,
    )
    await sdk.initialize()

    market_feed = MarketFeed(MarketFeedConfig(url=getattr(cfg, "ws_market_url", WS_MARKET_DEFAULT)))
    reconciler = Reconciler()

    def _on_user_msg(msg: dict) -> None:
        et = str(msg.get("event_type") or "")
        if et not in ("trade", "order_fill", "match"):
            return

        tid = extract_token_id(msg)
        side = extract_side(msg)
        price = extract_price(msg)
        size = extract_size(msg)
        trade_id = extract_trade_id(msg)
        ts = extract_ts(msg)

        if not tid or side not in ("BUY", "SELL") or price is None or size is None:
            return
        # market_id is not always present on WS event; keep empty string.
        reconciler.on_ws_fill(market_id=str(msg.get("market_id") or ""), token_id=tid, trade_id=trade_id or f"ws-{tid}-{ts}", ts=ts, payload=msg)

    user_feed = UserFeed(
        UserFeedConfig(url=getattr(cfg, "ws_user_url", WS_USER_DEFAULT)),
        sdk_client=sdk,
        market_ids=cfg.market_ids,
    )

    sup.spawn(market_feed.run(), name="ws_market")
    sup.spawn(user_feed.run(_on_user_msg), name="ws_user")

    log.info("Supervisor running. dry_run=%s markets=%d", cfg.dry_run, len(cfg.market_ids))
    try:
        await run_until_cancelled()
    finally:
        await sup.cancel_all()


def main() -> None:
    cfg = Config.from_env()
    errs = cfg.validate()
    if errs:
        for e in errs:
            print("ERROR:", e)
        sys.exit(1)

    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
