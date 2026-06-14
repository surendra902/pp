from __future__ import annotations

import asyncio
import sys

from polybot.config import Config
from polybot.logging_setup import get_logger
from polybot.protocol import WS_MARKET_DEFAULT, WS_USER_DEFAULT, CLOB_REST_DEFAULT
from polybot.runtime import Supervisor, run_until_cancelled
from polybot.feed_market import MarketFeed, MarketFeedConfig
from polybot.feed_user import UserFeed, UserFeedConfig
from polybot.reconcile import Reconciler


log = get_logger("main")


async def async_main(cfg: Config) -> None:
    sup = Supervisor()

    # Spine components
    market_feed = MarketFeed(MarketFeedConfig(url=getattr(cfg, "ws_market_url", WS_MARKET_DEFAULT)))
    user_feed = UserFeed(UserFeedConfig(url=getattr(cfg, "ws_user_url", WS_USER_DEFAULT)))
    reconciler = Reconciler()

    def _on_user_msg(msg: dict) -> None:
        # TODO: extract exact schemas from bot_LIVE.py and map to reconciler
        # reconciler.on_ws_fill(...)
        # reconciler.on_ws_order_update(...)
        _ = msg

    sup.spawn(market_feed.run(), name="ws_market")
    sup.spawn(user_feed.run(_on_user_msg), name="ws_user")

    log.info("Supervisor running. dry_run=%s", cfg.dry_run)
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
