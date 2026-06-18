from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    private_key: str = ""
    proxy_address: str = ""
    signature_type: int = 2

    clob_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137

    coins: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])

    dry_run: bool = True
    log_level: str = "INFO"

    # User WS subscription (required for fills)
    market_ids: List[str] = field(default_factory=list)

    # Risk
    min_order_usdc: float = 2.0
    max_order_usdc: float = 25.0
    max_position_usdc: float = 100.0
    max_bankroll_fraction: float = 0.10

    # Market-data freshness
    book_max_age_ms: float = 500.0

    # Loops
    discovery_interval_s: float = 10.0
    reconcile_interval_s: float = 30.0
    drift_check_concurrency: int = 4

    @classmethod
    def from_env(cls) -> "Config":
        def g(k: str, d: str = "") -> str:
            return os.environ.get(k, d).strip().strip('"\'')

        def gf(k: str, d: float) -> float:
            try:
                return float(g(k, str(d)))
            except Exception:
                return d

        def gi(k: str, d: int) -> int:
            try:
                return int(g(k, str(d)))
            except Exception:
                return d

        pk = g("POLYMARKET_PRIVATE_KEY")
        if pk and not pk.startswith("0x"):
            pk = "0x" + pk

        mids_raw = g("MARKET_IDS", "")
        mids = [m.strip() for m in mids_raw.split(",") if m.strip()]

        return cls(
            private_key=pk,
            proxy_address=g("POLYMARKET_PROXY_ADDRESS"),
            signature_type=gi("POLYMARKET_SIGNATURE_TYPE", 2),
            clob_url=g("CLOB_URL", "https://clob.polymarket.com").rstrip("/ "),
            gamma_url=g("GAMMA_URL", "https://gamma-api.polymarket.com").rstrip("/ "),
            chain_id=gi("CHAIN_ID", 137),
            coins=[c.strip().upper() for c in g("COINS", "BTC,ETH,SOL").split(",") if c.strip()],
            dry_run=g("DRY_RUN", "true").lower() in ("1", "true", "yes"),
            log_level=g("LOG_LEVEL", "INFO"),
            market_ids=mids,
            min_order_usdc=gf("MIN_ORDER_USDC", 2.0),
            max_order_usdc=gf("MAX_ORDER_USDC", 25.0),
            max_position_usdc=gf("MAX_POSITION_USDC", 100.0),
            max_bankroll_fraction=gf("MAX_BANKROLL_FRACTION", 0.10),
            book_max_age_ms=gf("MAX_BOOK_AGE_MS", 500.0),
            discovery_interval_s=gf("DISCOVERY_INTERVAL_S", 10.0),
            reconcile_interval_s=gf("RECONCILE_INTERVAL_S", 30.0),
            drift_check_concurrency=gi("DRIFT_CHECK_CONCURRENCY", 4),
        )

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.private_key:
            errs.append("POLYMARKET_PRIVATE_KEY is required")
        if not (0.0 < self.max_bankroll_fraction <= 1.0):
            errs.append("MAX_BANKROLL_FRACTION must be in (0, 1]")
        if self.min_order_usdc <= 0 or self.max_order_usdc < self.min_order_usdc:
            errs.append("Order sizing invalid")
        if not self.market_ids:
            errs.append("MARKET_IDS (comma-separated market ids) is required for user WS fills")
        return errs
