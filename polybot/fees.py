from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeCurve:
    """Polymarket US fee curve (documented).

    Docs specify: Fee = Θ * C * p * (1 - p)
    where C is number of contracts and p is price.

    This module intentionally models the curve only; category-specific theta
    must be configured externally.
    """

    theta: float  # e.g. taker theta

    def fee_usdc(self, *, contracts: float, price: float) -> float:
        # Conservative guardrails: clamp p into [0.0, 1.0]
        p = 0.0 if price < 0.0 else 1.0 if price > 1.0 else price
        c = max(0.0, contracts)
        return self.theta * c * p * (1.0 - p)


# Default from docs.polymarket.us (effective 2026-04-03): taker theta=0.05
DEFAULT_TAKER_CURVE = FeeCurve(theta=0.05)
# Maker rebate is negative (rebate). Docs show -0.0125.
DEFAULT_MAKER_REBATE_CURVE = FeeCurve(theta=-0.0125)
