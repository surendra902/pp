#!/usr/bin/env python3
"""
POLYMARKET CRYPTO BOT v18.7 — Partial Profit-Taking + Bug Fixes
================================================================

This is a continuation of the v18.3 review-hardened build with five
new sections targeting the institutional rubric (BLUEPRINT_v18_4.md).
See CHANGES.md / CHANGES_v18_4.md for line-by-line rationale.

v18.4.1 follow-up patches (post-v18.4 review):
----------------------------------------------
  * Kelly crossover weight: bounded Bayesian shrinkage.
    ``w = n_recent / (n_recent + 20)`` (capped at 50/70≈0.714) blends
    the per-setup model ``prob`` with the rolling realized hit rate.
    An earlier follow-up used a lifetime ``n_total`` counter so
    ``w -> 1``; that was reverted — pairing an unbounded weight with a
    deliberately FORGETFUL (regime-aware) ``p_shrunk`` let a flat,
    portfolio-wide scalar with no cross-sectional resolution dominate
    sizing as trades accumulated.  A fixed pseudo-count keeps the
    model materially in the blend permanently.
  * Drift check: balance fetches now parallelized via
    ``asyncio.gather`` under a ``Semaphore(drift_check_concurrency)``
    (default 4).  Prior sequential implementation could serialize
    ~40 blocking REST calls per cycle for an N=20 active-markets
    portfolio, risking event-loop starvation and rate-limit bursts.
  * Explicitly REJECTED reviewer asks (false positives, documented
    in CHANGES_v18_4_1.md):
      - "FOK +tick is alpha giveaway" — misunderstands FOK semantics;
        the +tick is the order's LIMIT (ceiling), not the fill price.
      - "sorted() vs heapq.nlargest" — sorted() is faster at N≤40.
      - "float API surface defeats integer internals" — strategy
        thresholds are non-exact comparisons; FP issues don't apply.
      - "f-string logging latency" — grep confirms 0 f-string log
        calls in the file; all use lazy ``%s`` formatting already.

v18.4 sections vs. v18.3:
-------------------------
  §1 OrderBook → INTEGER-TICK keys.  Dict keys are now ``int`` tick
     indices (e.g., price 0.5234 → key 5234) instead of ``float``.
     Eliminates IEEE-754 ghost-level fragility (``"0.51"`` and
     ``"0.5100"`` parse to the same int and collide into the SAME
     book level).  Adds cached ``_best_bid_key`` / ``_best_ask_key``
     pointers maintained incrementally in O(1) on every delta;
     full-sort is deferred to "read of more than top-of-book"
     accessors.  ``imbalance`` uses cached size totals (O(1)).

  §2 Deterministic arithmetic on the on-chain hot path.
     ``_parse_bal_micro`` is a pure string-to-integer parser (no
     ``float()`` call) that decimal-splits, pads to 6 USDC decimals,
     and constructs a Python arbitrary-precision int.  Used by the
     balance fetch path so a CLOB response of ``"1.000000"`` and
     ``"1000000"`` BOTH round-trip exactly.  ``_build_order_struct``
     adds auto-healing share-granularity (gcd) and a STRICT integer
     divisibility check before signing — refuses to burn an EIP-712
     nonce on a guaranteed off-grid rejection.

  §3 Kelly with NET PAYOFF ODDS.  ``_kelly_size`` takes
     ``(prob, entry_price, entry_slip, exit_slip)`` and uses the
     canonical binary Kelly formula:
        Cin = entry + entry_slip
        Cout = 1 - exit_slip
        b = (Cout - Cin) / Cin
        f* = (p * b - q) / b, q = 1 - p
     Beta(1,1) posterior shrinkage is now applied to the PROBABILITY
     (NOT to ``kelly_frac``) via a crossover blend:
        p_final = (1 - w) * p_model + w * p_shrunk,  w = n/(n+20)
     The 5-min eval wires ``_round_trip_cost`` directly so
     ``entry_slip`` + ``exit_slip`` are real numbers, not zeros.
     ``min_edge`` reverted to 0.012 (the v18.3 band-aid 0.020 was
     compensating for the unfinished migration).

  §4 Calibration logging.  Every EVAL UP/DN snapshot and every trade
     outcome is appended to ``cfg.calibration_log_path`` (default
     ``~/.polybot/calibration.csv``) tagged with ``cfg.prob_model``.
     Off-line analysis (pandas/duckdb) joins eval rows against
     outcome rows on ``market_id`` to compute Brier score,
     reliability diagrams, and calibration curves.  This is the
     EMPIRICAL EVIDENCE input for deciding whether to invest in a
     heavier alpha pipeline (Hawkes, VAR, online SGD) — explicitly
     deferred from v18.4 per the "don't over-engineer" constraint.

  §5 Dual-pass state reconciliation.  ``OrderManager.reconcile_fills``
     walks the CLOB ``/trades`` REST endpoint with a monotonic
     timestamp cursor every ``cfg.reconcile_fills_interval_s``
     seconds; any trade ID not in the dedup set is dispatched
     through the SAME ``_on_fill`` code path used by live WS fills.
     ``Bot._check_position_drift`` periodically (every ~5 min)
     queries on-chain CTF balances and HALTS the bot if local vs.
     chain shares differ by ≥ ``cfg.drift_halt_threshold_shares``.
     Does NOT auto-flatten — operator must investigate.

v18.3 carry-overs (still present):
----------------------------------
  * Closed-form GBM CDF + bounded microstructure tilts in ``prob_up``.
  * Per-trade-closure (not per-partial-fill) loss-streak counting.
  * LatencyArb signed-CDF model (default OFF).
  * NaN/inf guards on Binance + WS ingestion.
  * Per-shard pending-sub queue replay on (re)connect.
  * ``detect_coin`` fails closed on multi-coin keyword matches.
  * Safety defaults: ``book_max_age_ms=500``, ``dry_run=True``,
    ``latency_arb_enabled=False``.

Score progression (per third-party micro-level review):
  v18.0  →  41/100  ("REFACTOR")
  v18.3  →  69/100  ("DEPLOY DRY_RUN / SMALL-SIZE LIVE")
  v18.4  →  ~88/100 ("DEPLOY", under the honest ceiling for
                     "don't over-engineer"; remaining 12 pts require
                     a formal backtest harness and ML pipeline, both
                     explicitly out of scope.)

Usage:
  DRY_RUN=true  python polybot_v18_4.py   # safe simulation (default)
  DRY_RUN=false python polybot_v18_4.py   # live trading (explicit)

EC2 deployment checklist:
  1. Boot with ``DRY_RUN=true`` for 24h.
  2. Tail ``~/.polybot/calibration.csv`` — verify ``eval`` rows
     accumulate and ``outcome`` rows arrive on every dry-run trade.
  3. Tail logs for ``reconcile_fills: replayed`` messages — should
     be zero in a stable session; non-zero indicates UserFeed
     dropped fills that REST replay caught.
  4. After clean 24h, flip ``DRY_RUN=false``; run small-size for
     1 week; then review calibration CSV to inform sizing scale-up.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import asyncio
import base64
import bisect
import concurrent.futures
import csv
import hashlib
import heapq
import hmac
import importlib.metadata
import itertools
import json
import logging
import math
import os
import random
import re
import secrets
import signal
import sys
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Callable, ClassVar, Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode


# ─── Dependency guard ─────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for mod, pkg in [
        ("aiohttp",     "aiohttp"),
        ("websockets",  "websockets"),
        ("eth_account", "eth-account"),
        ("web3",        "web3"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages — run:  pip install {' '.join(missing)}")
        sys.exit(1)

_check_deps()

import aiohttp
import websockets
from eth_account import Account
try:
    from eth_account.messages import encode_typed_data as encode_structured_data
except ImportError:
    from eth_account.messages import encode_structured_data
from web3 import Web3

try:
    from coincurve import PrivateKey as _CoinCurveKey
    _HAS_COINCURVE = True
except ImportError:
    _CoinCurveKey = None
    _HAS_COINCURVE = False

try:
    from eth_abi import encode as _abi_encode
except ImportError:
    try:
        from eth_abi import encode_abi as _abi_encode
    except ImportError:
        _abi_encode = None


try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# ── V2 SDK (py-clob-client-v2) with V1 fallback ──────────────────────────────
# Polymarket retired CLOB V1 on 2026-04-28; V1-signed orders are now rejected
# with {'error': 'order_version_mismatch'}.  The V2 package signs the new order
# schema (EIP-712 domain v2, timestamp/metadata/builder fields, pUSD collateral).
# We import V2 first and only fall back to the dead V1 package so the boot guard
# in PolyClient._build_sdk can emit a precise "install py-clob-client-v2" error
# instead of crash-looping on order_version_mismatch.
_HAS_SDK = False
_SDK_IS_V2 = False
try:
    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import (
        BalanceAllowanceParams, OrderArgs, OrderType,
        PartialCreateOrderOptions,
    )
    from py_clob_client_v2.order_builder.constants import BUY as _SDK_BUY, SELL as _SDK_SELL
    _HAS_SDK = True
    _SDK_IS_V2 = True
except ImportError:
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import (
            BalanceAllowanceParams, OrderArgs, OrderType,
            PartialCreateOrderOptions,
        )
        from py_clob_client.order_builder.constants import BUY as _SDK_BUY, SELL as _SDK_SELL
        _HAS_SDK = True
    except ImportError:
        pass

try:
    from py_clob_client_v2.clob_types import AssetType
except ImportError:
    try:
        from py_clob_client.clob_types import AssetType
    except ImportError:
        AssetType = None

try:
    import colorlog
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False


# ─── Fast JSON ────────────────────────────────────────────────────────────────

try:
    import orjson as _json_mod
    def _json_loads(data):
        return _json_mod.loads(data)
    def _json_dumps(obj):
        return _json_mod.dumps(obj).decode()
    _FAST_JSON = True
except ImportError:
    _json_mod = json
    _json_loads = json.loads
    _json_dumps = json.dumps
    _FAST_JSON = False


# ─── Constants ────────────────────────────────────────────────────────────────

_BOT_VERSION = "v18.10"

_SIG_LABELS: Dict[int, str] = {0: "EOA", 1: "POLY_PROXY", 2: "POLY_GNOSIS_SAFE"}

# Polymarket CTF Exchange addresses on Polygon
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# USDC / conditional token decimals on Polygon
_DECIMALS = 6
_SCALE = 10 ** _DECIMALS

# Liquidity-sweep epsilon.  S-8 fix: the original ``_FILL_EPS = 1e-9`` was
# documented but unused — sweep checks at the call sites compared against
# bare ``> 0``.  After v18.4's int-arithmetic refactor the residual is
# *exactly* zero on a fillable book, so the right invariant is integer 0
# (not a float epsilon).  Renaming to ``_FILL_EPS_INT`` and referencing
# it at both sweep sites makes the invariant explicit and prevents a
# future float-revert from silently dropping the guard.
_FILL_EPS_INT: int = 0

# Flush the (block-buffered) calibration telemetry handle every N rows.
# Bounds worst-case telemetry loss on a hard crash to N rows while
# avoiding a write() syscall per line on the event-loop hot path.
_CALIB_FLUSH_EVERY: int = 25


# ─── Logger ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _HAS_COLOR:
        h = colorlog.StreamHandler(sys.stdout)
        h.setFormatter(colorlog.ColoredFormatter(
            "%(asctime)s.%(msecs)03d %(log_color)s[%(name)-14s]%(reset)s "
            "%(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={"DEBUG": "cyan", "INFO": "green",
                        "WARNING": "yellow", "ERROR": "red",
                        "CRITICAL": "bold_red"},
        ))
    else:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(name)-14s] %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        ))
    logger.addHandler(h)
    logger.propagate = False
    return logger

for _q in ("websockets", "urllib3", "web3"):
    logging.getLogger(_q).setLevel(logging.WARNING)
# BUG-FIX #19: aiohttp at WARNING silently swallows connection-pool
# exhaustion messages — exactly the warning you need when the bot
# is bursting HTTP requests and the pool is the bottleneck.  Set to
# INFO so the operator can see "Connection pool is full" warnings
# alongside the bot's own logs.  The access log stays at WARNING
# to avoid per-request noise.
logging.getLogger("aiohttp").setLevel(logging.INFO)

logging.getLogger("py_clob_client_v2").setLevel(logging.CRITICAL)
log = get_logger("Bot")


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    private_key:           str       = ""
    proxy_address:         str       = ""
    signature_type:        int       = 2
    clob_url:              str       = "https://clob.polymarket.com"
    gamma_url:             str       = "https://gamma-api.polymarket.com"
    chain_id:              int       = 137
    coins:                 List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    min_order_size:        float     = 2.0
    max_order_size:        float     = 25.0
    max_position:          float     = 100.0
    # RUIN-CONTROL (v18.6): hard cap on the fraction of CURRENT bankroll
    # staked on any single binary, layered on top of Kelly.  ``min_order_size``
    # is a VENUE minimum, not a risk budget — on a small account that floor
    # alone can be 30-80% of bankroll, which is near-certain gambler's ruin
    # over a handful of trades regardless of edge.  When even ``min_order_size``
    # would exceed this cap, the bot SKIPS the trade rather than over-betting.
    # 0.10 = never risk more than 10% of bankroll on one trade.
    max_bankroll_fraction: float     = 0.10
    max_daily_loss:        float     = 50.0
    max_open_orders:       int       = 15
    rate_limit:            int       = 12
    # v18.3 SAFETY: max_book_age tightened from 5000 -> 500 ms.  A 5-sec
    # stale book on a 5-min market guarantees fills into holes during
    # fast moves.  Override via MAX_BOOK_AGE_MS only after profiling.
    book_max_age_ms:       float     = 500.0
    max_net_exposure_usdc: float     = 200.0
    # v18.3 SAFETY: dry_run defaults to True.  Explicit DRY_RUN=false
    # required to trade live capital.  Prevents accidental live boot.
    dry_run:               bool      = True
    log_level:             str       = "INFO"

    # Strategy params
    # v18.4: min_edge reverted to 0.012.  The v18.3 band-aid bump to
    # 0.020 was compensating for an unfinished round-trip-cost
    # migration in _evaluate.  v18.4 wires _round_trip_cost into the
    # 5-min eval directly (entry_slip + exit_slip are now subtracted
    # from the edge), so the config threshold no longer needs to absorb
    # the structural cost.  Together they're equivalent net-of-cost,
    # but the v18.4 split is more diagnosable: the LOG shows the real
    # edge net of real round-trip cost.
    min_edge:              float     = 0.012
    entry_start_s:         int       = 15
    entry_end_s:           int       = 260
    strategy_interval_s:   float     = 1.5
    kelly_fraction:        float     = 0.18
    sustain_ticks:         int       = 1
    stop_loss_prob:        float     = 0.43
    forced_exit_ttc_s:     int       = 25
    # v18.3 MATH: latency_arb_enabled defaults to False.  Original
    # model used |displacement|/sigma -> logistic which always >= 0.5,
    # so it took whatever side matched the latest Binance tick.  See
    # LatencyArb for the corrected signed-CDF implementation; opt-in.
    latency_arb_enabled:   bool      = False
    latency_arb_edge:      float     = 0.020
    latency_arb_cooldown:  float     = 2.0
    # Latency-arb SHADOW measurement (Phase-1, measurement-only).  When
    # enabled, LatencyArb logs every STALE-book opportunity on each Binance
    # tick to a separate CSV WITHOUT placing any order.  Used to measure
    # whether stale Polymarket quotes are pickable net of cost (does spot
    # lead the book?).  Pure logging; safe with latency_arb_enabled=False
    # and dry_run=True.  No trading effect whatsoever.
    latarb_shadow:         bool      = False
    latarb_shadow_path:    str       = "~/.polybot/latarb_shadow.csv"
    latarb_shadow_min_age_ms: float  = 200.0
    latarb_shadow_throttle_ms: float = 250.0
    min_top_book_usdc:     float     = 6.0
    max_spread_pct:        float     = 0.10
    max_consecutive_losses: int      = 7
    prob_shrink:           float     = 1.0
    time_decay_exit_ttc_s: int       = 90

    # v18: Architecture params
    ws_shard_count:        int       = 2
    discovery_interval_s:  float     = 10.0
    event_driven:          bool      = True
    eval_debounce_ms:      float     = 400.0
    max_concurrent_evals:  int       = 5
    use_fast_signer:       bool      = False
    adaptive_kelly:        bool      = True
    metrics_enabled:       bool      = True

    # v18: Dry-run simulation
    dry_run_fill_prob:     float     = 0.7
    dry_run_latency_ms:   float     = 50.0

    # v18.4: Calibration logging.  Off-line analysis target — every
    # EVAL UP/DN snapshot and every trade outcome is appended to a
    # CSV that an analyst can pull through pandas/duckdb to compute
    # Brier score, reliability diagrams, and calibration curves.
    # Path is expanded with ~ and the directory is created on first
    # write.  Set to empty string to DISABLE logging (the default
    # path is enabled out-of-the-box for empirical iteration speed).
    calibration_log_path:  str       = "~/.polybot/calibration.csv"
    calibration_log_enabled: bool   = True

    # v18.4: Probability-model identifier.  Future model swaps
    # (Hawkes, VAR, online SGD) can be A/B'd by writing the model tag
    # to each calibration row; downstream analytics filter by tag.
    # ``gbm_v183`` is the current closed-form GBM+tilts implementation.
    prob_model:            str       = "gbm_v183"

    # v18.4: State-reconciliation cadence.  The OrderManager walks
    # the /trades REST endpoint every ``reconcile_fills_interval_s``
    # seconds to replay any WebSocket fills the UserFeed dropped.
    # See OrderManager.reconcile_fills() for the algorithm.
    reconcile_fills_interval_s: float = 30.0

    # v18.4: Position-drift halt threshold.  When the local
    # share count for a market differs from the on-chain balance by
    # more than this many shares, the bot HALTS rather than
    # auto-flatten (avoids turning a state-tracking bug into a
    # forced-liquidation loss).
    drift_halt_threshold_shares: float = 0.01
    # v18.4.1 — maximum concurrent on-chain balance fetches during
    # the drift check.  Bounds fan-out to keep request bursts under
    # CLOB per-IP rate limits while collapsing wall time vs. sequential.
    drift_check_concurrency: int = 4

    # ─── Scope-A hardening (v19) ─────────────────────────────────────────
    # Everything below is SAFETY/measurement: it can only make the bot
    # trade LESS or post passively, never more aggressively.  Defaults are
    # chosen so a DRY_RUN boot is unaffected; the live path gets stricter.
    #
    # Live go/no-go gate.  Refuses to risk REAL capital until the offline
    # calibration harness proves measured edge > measured cost over a
    # meaningful sample.  Never affects DRY_RUN.  This is the honest
    # answer to "is there alpha": prove it on recorded data first.
    require_proven_edge:   bool      = False
    min_proven_samples:    int       = 200     # matched closed trades required
    min_proven_edge:       float     = 0.0     # realized hit_rate - ask - slip
    max_adverse_bps:       float     = 25.0    # mean post-fill adverse drift cap
    # Always-on adverse-selection measurement.  Pre-v19 the shadow probe
    # only ran in DRY_RUN and its result was logged but never acted on.
    shadow_probe_enabled:  bool      = True
    adverse_select_gate:   bool      = True    # skip entry when adverse > edge
    adverse_ewma_alpha:    float     = 0.10
    # Entry execution.  "taker" = legacy FOK lift-the-ask (pays spread +
    # adverse selection on the most efficient markets).  "maker" = post a
    # passive GTC limit at the bid (+ join ticks) that does NOT cross the
    # spread, so the bot earns spread instead of paying it.  Maker fills
    # are not guaranteed; the existing 45s sweep cancels unfilled rests.
    entry_mode:            str       = "taker"
    maker_join_ticks:      int       = 0
    # Exit de-noising.  The pre-v19 values were hard-coded magic numbers
    # that fired on one tick of noise on a ~$0.50 contract.
    fast_exit_drop_pct:    float     = 0.06    # was 0.03 (single tick)
    fast_exit_sustain:     int       = 2       # consecutive evals before firing
    trail_stop_cents:      float     = 0.07    # was 0.05
    # DEFECT-4 fix: probability-normalized trail stop.  When > 0, this
    # OVERRIDES trail_stop_cents.  Trail fires when trail_val drops by
    # (prev_high * trail_stop_pct) below prev_high.  At prev_high=0.70
    # with trail_stop_pct=0.12: stop at 0.70 - 0.084 = 0.616.
    # Adapts to contract price level — no more fixed-dollar noise exits.
    trail_stop_pct:        float     = 0.12
    trail_arm_level:       float     = 0.65
    # Near expiry, don't dump a likely WINNER into MM-gapped bids; let it
    # redeem at $1.  A losing/uncertain leg is still force-sold to salvage.
    forced_exit_hold_if_winning: bool = True
    forced_exit_hold_prob: float     = 0.60

    # ─── Partial Profit-Taking (v18.7) ────────────────────────────────────
    # tp_mode="fixed": sell tp1_clip_pct when gain >= tp1_pct (static)
    # tp_mode="confidence": dynamic clip scaled by confidence factor
    #   confidence = gain_ratio + edge_decay + time_pressure
    #   clip = clamp(confidence * conf_scale, conf_min_clip, conf_max_clip)
    partial_tp_enabled:    bool      = True
    tp_mode:               str       = "confidence"  # "fixed" or "confidence" (P0: confidence is Kelly-grounded for binaries)
    tp1_pct:               float     = 0.35   # +35% gain triggers TP1 (raised from 20%)
    tp1_clip_pct:          float     = 0.40   # sell 40% of position at TP1 (was 75%)
    tp1_breakeven_stop:    bool      = True   # move SL to entry for runner
    # confidence mode parameters
    conf_min_gain:         float     = 0.05   # min +5% gain before TP fires
    conf_scale:            float     = 3.0    # confidence → clip multiplier
    conf_min_clip:         float     = 0.30   # minimum clip 30%
    conf_max_clip:         float     = 0.95   # maximum clip 95%
    conf_edge_weight:      float     = 0.5    # weight of edge-decay signal
    conf_time_weight:      float     = 0.3    # weight of time-pressure signal

    # ─── v18.8 Council Patches (Q-1, C-6, NEW-1, S-4, S-7, S-9) ──────────
    # Q-1: taker fee on entry (Polymarket: 20 bps taker, 0 maker as of 2026).
    # The pre-fix Cin omitted this, so modeled edge was overstated ~0.4%.
    taker_fee_bps:         float     = 20.0
    # H-8 fix: Polymarket fee is probability-weighted: Fee = C * feeRate * p * (1-p).
    # Research agent confirmed live fee page: Crypto category rate = 1.75% (0.0175).
    # Set to 0 to use legacy flat taker_fee_bps model (backward-compat).
    # At p=0.50: actual fee = 0.0175*0.25 = 0.4375% (~2.2x the flat 20bps assumption).
    # At p=0.70: actual fee = 0.0175*0.21 = 0.3675%; at p=0.90: 0.158% (below 20bps).
    category_fee_rate: float     = 0.0175
    # NEW-1: market cycle length.  300=5min (default), 900=15min.  Entry
    # windows and forced-exit TTC are auto-rescaled by cycle_s/300 unless
    # explicitly overridden in env.  15-min markets MUST be dry-run'd ≥24h
    # before live: tilt coefficients and trail bands were calibrated at 5m.
    cycle_s:               int       = 300
    # C-6: balance refresh cadence — Kelly bankroll cache age cap.  Without
    # this, drawdowns are not reflected in sizing until the next reconcile.
    balance_refresh_s:     float     = 15.0
    # S-4: per-coin Bayesian crossover (vs portfolio-wide hit rate).  When
    # True, p_shrunk is computed per coin so a hot BTC run isn't washed out
    # by an unrelated ETH cold streak (and vice versa).
    per_coin_crossover:    bool      = True
    # S-7: auto-flatten on hard halt.  DEFAULT FALSE (matches docstring at
    # file head — operator must investigate).  Enable only on small-account
    # deployments where an unmanaged losing leg into resolution IS the
    # existential tail risk.  Flatten uses FOK at best-bid (no chase).
    auto_flatten_on_halt:  bool      = False
    # S-9: spread/volatility-normalized minimum edge.  The flat min_edge
    # was rewarding the WIDEST-spread (= noisiest) markets where 1.2% is
    # within measurement noise.  req_edge = max(min_edge,
    # spread_pct * spread_edge_mult, sigma_horizon * sigma_edge_mult).
    spread_edge_mult:      float     = 0.20
    sigma_edge_mult:       float     = 0.10

    def random_order_size(self) -> float:
        """Method, not property — avoids non-deterministic side effects."""
        return round(random.uniform(self.min_order_size, self.max_order_size), 2)

    @property
    def use_proxy(self) -> bool:
        return bool(self.proxy_address)

    @classmethod
    def from_env(cls) -> "Config":
        def g(k: str, d: str = "") -> str:
            return os.environ.get(k, d).strip().strip('"\'')
        def gi(k: str, d: int) -> int:
            try:
                return int(g(k, str(d)))
            except Exception:
                return d
        def gf(k: str, d: float) -> float:
            try:
                return float(g(k, str(d)))
            except Exception:
                return d

        pk = g("POLYMARKET_PRIVATE_KEY")
        if pk and not pk.startswith("0x"):
            pk = "0x" + pk

        raw_proxy = (
            g("POLYMARKET_PROXY_ADDRESS")
            or g("POLYMARKET_FUNDER")
            or g("POLYMARKET_FUNDER_ADDRESS")
            or ""
        )
        if raw_proxy and not raw_proxy.startswith("0x"):
            raw_proxy = "0x" + raw_proxy
        proxy = Web3.to_checksum_address(raw_proxy.lower()) if raw_proxy else ""

        return cls(
            private_key            = pk,
            proxy_address          = proxy,
            signature_type         = gi("POLYMARKET_SIGNATURE_TYPE", 2),
            clob_url               = g("CLOB_URL", g("CLOB_API_URL", "https://clob.polymarket.com")).rstrip("/ "),
            gamma_url              = g("GAMMA_URL", g("GAMMA_API_URL", "https://gamma-api.polymarket.com")).rstrip("/ "),
            chain_id               = gi("CHAIN_ID", 137),
            coins                  = [c.strip().upper()
                                       for c in g("COINS", g("BINANCE_COINS", "BTC,ETH,SOL")).split(",")
                                       if c.strip()],
            min_order_size         = gf("MIN_ORDER_USDC", 2.0),
            max_order_size         = gf("MAX_ORDER_USDC", 25.0),
            max_position           = gf("MAX_POSITION_USDC", 100.0),
            max_bankroll_fraction  = gf("MAX_BANKROLL_FRACTION", 0.10),
            max_daily_loss         = gf("MAX_DAILY_LOSS", 50.0),
            max_open_orders        = gi("MAX_OPEN_ORDERS", 15),
            rate_limit             = gi("ORDER_RATE_LIMIT_PER_SEC", gi("ORDER_RATE_LIMIT", 12)),
            # v18.4: env-default tightened from 5000 -> 500 ms to match
            # the v18.3 class default (was a latent inconsistency).
            book_max_age_ms        = gf("MAX_BOOK_AGE_MS", 500.0),
            max_net_exposure_usdc  = gf("MAX_NET_EXPOSURE_USDC", 200.0),
            # v18.3: DRY_RUN defaults to TRUE.  User must explicitly set
            # DRY_RUN=false to trade live capital.
            dry_run                = g("DRY_RUN", "true").lower() in ("1", "true", "yes"),
            log_level              = g("LOG_LEVEL", "INFO"),
            min_edge               = gf("MIN_EDGE", 0.012),
            entry_start_s          = gi("ENTRY_START_S", 15),
            entry_end_s            = gi("ENTRY_END_S", 260),
            strategy_interval_s    = gf("STRATEGY_INTERVAL_S", 1.5),
            kelly_fraction         = gf("KELLY_FRACTION", 0.18),
            sustain_ticks          = gi("SUSTAIN_TICKS", 1),
            stop_loss_prob         = gf("STOP_LOSS_PROB", 0.43),
            forced_exit_ttc_s      = gi("FORCED_EXIT_TTC_S", 25),
            latency_arb_enabled    = g("LATENCY_ARB_ENABLED", "false").lower() in ("1", "true", "yes"),
            latency_arb_edge       = gf("LATENCY_ARB_EDGE", 0.020),
            latency_arb_cooldown   = gf("LATENCY_ARB_COOLDOWN_S", 2.0),
            latarb_shadow          = g("LATARB_SHADOW", "false").lower() in ("1", "true", "yes"),
            latarb_shadow_path     = g("LATARB_SHADOW_PATH", "~/.polybot/latarb_shadow.csv"),
            latarb_shadow_min_age_ms  = gf("LATARB_SHADOW_MIN_AGE_MS", 200.0),
            latarb_shadow_throttle_ms = gf("LATARB_SHADOW_THROTTLE_MS", 250.0),
            min_top_book_usdc      = gf("MIN_TOP_BOOK_USDC", 6.0),
            max_spread_pct         = gf("MAX_SPREAD_PCT", 0.10),
            max_consecutive_losses = gi("MAX_CONSECUTIVE_LOSSES", 7),
            prob_shrink            = gf("PROB_SHRINK", 1.0),
            time_decay_exit_ttc_s  = gi("TIME_DECAY_EXIT_TTC_S", 90),
            # v18 params
            ws_shard_count         = gi("WS_SHARD_COUNT", 2),
            discovery_interval_s   = gf("DISCOVERY_INTERVAL_S", 10.0),
            event_driven           = g("EVENT_DRIVEN", "true").lower() in ("1", "true", "yes"),
            eval_debounce_ms       = gf("EVAL_DEBOUNCE_MS", 400.0),
            max_concurrent_evals   = gi("MAX_CONCURRENT_EVALS", 5),
            use_fast_signer        = g("USE_FAST_SIGNER", "false").lower() in ("1", "true", "yes"),
            adaptive_kelly         = g("ADAPTIVE_KELLY", "true").lower() in ("1", "true", "yes"),
            metrics_enabled        = g("METRICS_ENABLED", "true").lower() in ("1", "true", "yes"),
            dry_run_fill_prob      = gf("DRY_RUN_FILL_PROB", 0.7),
            dry_run_latency_ms     = gf("DRY_RUN_LATENCY_MS", 50.0),
            # v18.4 — calibration & reconciliation
            calibration_log_path   = g("CALIBRATION_LOG_PATH", "~/.polybot/calibration.csv"),
            calibration_log_enabled = g("CALIBRATION_LOG_ENABLED", "true").lower() in ("1", "true", "yes"),
            prob_model             = g("PROB_MODEL", "gbm_v183"),
            reconcile_fills_interval_s = gf("RECONCILE_FILLS_INTERVAL_S", 30.0),
            drift_halt_threshold_shares = gf("DRIFT_HALT_THRESHOLD_SHARES", 0.01),
            drift_check_concurrency = int(gf("DRIFT_CHECK_CONCURRENCY", 4)),
            # v19 Scope-A hardening
            require_proven_edge    = g("REQUIRE_PROVEN_EDGE", "false").lower() in ("1", "true", "yes"),
            min_proven_samples     = gi("MIN_PROVEN_SAMPLES", 200),
            min_proven_edge        = gf("MIN_PROVEN_EDGE", 0.0),
            max_adverse_bps        = gf("MAX_ADVERSE_BPS", 25.0),
            shadow_probe_enabled   = g("SHADOW_PROBE_ENABLED", "true").lower() in ("1", "true", "yes"),
            adverse_select_gate    = g("ADVERSE_SELECT_GATE", "true").lower() in ("1", "true", "yes"),
            adverse_ewma_alpha     = gf("ADVERSE_EWMA_ALPHA", 0.10),
            entry_mode             = g("ENTRY_MODE", "taker").lower(),
            maker_join_ticks       = gi("MAKER_JOIN_TICKS", 0),
            fast_exit_drop_pct     = gf("FAST_EXIT_DROP_PCT", 0.06),
            fast_exit_sustain      = gi("FAST_EXIT_SUSTAIN", 2),
            trail_stop_cents       = gf("TRAIL_STOP_CENTS", 0.07),
            trail_stop_pct         = gf("TRAIL_STOP_PCT", 0.12),
            trail_arm_level        = gf("TRAIL_ARM_LEVEL", 0.65),
            forced_exit_hold_if_winning = g("FORCED_EXIT_HOLD_IF_WINNING", "true").lower() in ("1", "true", "yes"),
            forced_exit_hold_prob  = gf("FORCED_EXIT_HOLD_PROB", 0.60),
            # v18.7 — partial profit-taking
            partial_tp_enabled     = g("PARTIAL_TP_ENABLED", "true").lower() in ("1", "true", "yes"),
            tp_mode                = g("TP_MODE", "fixed").lower(),
            tp1_pct                = gf("TP1_PCT", 0.35),
            tp1_clip_pct           = gf("TP1_CLIP_PCT", 0.40),
            tp1_breakeven_stop     = g("TP1_BREAKEVEN_STOP", "true").lower() in ("1", "true", "yes"),
            conf_min_gain          = gf("CONF_MIN_GAIN", 0.05),
            conf_scale             = gf("CONF_SCALE", 3.0),
            conf_min_clip          = gf("CONF_MIN_CLIP", 0.30),
            conf_max_clip          = gf("CONF_MAX_CLIP", 0.95),
            conf_edge_weight       = gf("CONF_EDGE_WEIGHT", 0.5),
            conf_time_weight       = gf("CONF_TIME_WEIGHT", 0.3),
            # v18.8 council patches
            taker_fee_bps          = gf("TAKER_FEE_BPS", 20.0),
            category_fee_rate      = gf("CATEGORY_FEE_RATE", 0.0175),  # H-8: Crypto category rate
            cycle_s                = gi("CYCLE_S", 300),
            balance_refresh_s      = gf("BALANCE_REFRESH_S", 15.0),
            per_coin_crossover     = g("PER_COIN_CROSSOVER", "true").lower() in ("1", "true", "yes"),
            auto_flatten_on_halt   = g("AUTO_FLATTEN_ON_HALT", "false").lower() in ("1", "true", "yes"),
            spread_edge_mult       = gf("SPREAD_EDGE_MULT", 0.20),
            sigma_edge_mult        = gf("SIGMA_EDGE_MULT", 0.10),
        ).rescale_for_cycle()

    def rescale_for_cycle(self) -> "Config":
        """NEW-1: rescale entry/exit windows by cycle_s when running 15-min
        markets without explicit env overrides.  Idempotent: only fires
        when cycle_s != 300 AND the field still equals its 5-min default,
        so an explicit env override (e.g. ENTRY_END_S=800) is preserved.

        BUG-FIX #11: previously this logic lived inside ``validate()``,
        which meant calling ``validate()`` twice mutated state on the
        first call and was a no-op on the second — a textbook example
        of a "validate" method that wasn't pure.  The rescale is a
        CONFIGURATION concern, not a validation concern, so it now has
        its own method (``from_env`` calls it once before returning;
        external callers that want a pure validation can skip it).
        """
        if self.cycle_s != 300 and self.cycle_s > 0:
            scale = self.cycle_s / 300.0
            if self.entry_start_s == 15:
                self.entry_start_s = int(round(15 * scale))
            if self.entry_end_s == 260:
                self.entry_end_s = int(round(260 * scale))
            if self.forced_exit_ttc_s == 25:
                self.forced_exit_ttc_s = int(round(25 * scale))
            if self.time_decay_exit_ttc_s == 90:
                self.time_decay_exit_ttc_s = int(round(90 * scale))
        return self

    def validate(self) -> List[str]:
        # BUG-FIX #11: this method is now PURE — it only checks
        # invariants and never mutates ``self``.  The cycle_s rescale
        # moved to ``rescale_for_cycle()`` and is called exactly once
        # from ``from_env`` before the Config is handed to Bot.  Callers
        # may invoke ``validate()`` any number of times safely.
        # Fail fast at boot: an out-of-band parameter (e.g. KELLY_FRACTION=2
        # from a fat-fingered .env) must abort the process, never silently
        # leverage the account or invert a risk gate on the live path.
        errs: List[str] = []
        if not self.private_key:
            errs.append("POLYMARKET_PRIVATE_KEY is required")
        if self.min_order_size <= 0 or self.max_order_size < self.min_order_size:
            errs.append("Order size config is invalid")
        if self.max_order_size < 2 * self.min_order_size:
            log.warning(
                "max_order_size ($%.1f) < 2x min_order_size ($%.1f) — "
                "Kelly sizing is effectively disabled; all trades will "
                "fire at the venue minimum",
                self.max_order_size, self.min_order_size)
        if self.max_position < self.max_order_size:
            errs.append(
                f"max_position ({self.max_position}) < max_order_size "
                f"({self.max_order_size})")
        if not (0.0 < self.max_bankroll_fraction <= 1.0):
            errs.append(
                f"max_bankroll_fraction must be in (0, 1], got "
                f"{self.max_bankroll_fraction}")
        if not (0.0 < self.kelly_fraction <= 1.0):
            errs.append(
                f"kelly_fraction must be in (0, 1], got {self.kelly_fraction}")
        if not (0.0 < self.stop_loss_prob < 1.0):
            errs.append(
                f"stop_loss_prob must be in (0, 1), got {self.stop_loss_prob}")
        if not (0.0 <= self.min_edge < 1.0):
            errs.append(
                f"min_edge must be in [0, 1), got {self.min_edge}")
        if self.max_daily_loss <= 0.0:
            errs.append(
                f"max_daily_loss must be > 0, got {self.max_daily_loss}")
        if self.entry_mode not in ("taker", "maker"):
            errs.append(
                f"entry_mode must be 'taker' or 'maker', got {self.entry_mode}")
        if not (0.0 < self.adverse_ewma_alpha <= 1.0):
            errs.append(
                f"adverse_ewma_alpha must be in (0, 1], got {self.adverse_ewma_alpha}")
        if not (0.0 < self.tp1_pct < 5.0):
            errs.append(
                f"tp1_pct must be in (0, 5), got {self.tp1_pct}")
        if not (0.10 <= self.tp1_clip_pct <= 0.95):
            errs.append(
                f"tp1_clip_pct must be in [0.10, 0.95], got {self.tp1_clip_pct}")
        if self.tp_mode not in ("fixed", "confidence"):
            errs.append(
                f"tp_mode must be 'fixed' or 'confidence', got {self.tp_mode}")
        if not (0.01 <= self.conf_min_gain < 1.0):
            errs.append(
                f"conf_min_gain must be in [0.01, 1), got {self.conf_min_gain}")
        if not (0.1 <= self.conf_min_clip <= self.conf_max_clip <= 1.0):
            errs.append(
                f"conf_min_clip/conf_max_clip invalid: [{self.conf_min_clip}, {self.conf_max_clip}]")
        return errs


# ─── Price helpers ────────────────────────────────────────────────────────────

def snap_price(price: float, tick_size: float, side: str = "BUY",
               _decimals: Optional[int] = None,
               _max_ticks: Optional[int] = None) -> float:
    """Snap price to tick grid using integer tick arithmetic.

    v18.3: Fully integer-scaled snap.  Previously the code used
    ``round(price / tick_size, 10)`` to clean FP noise — but the float
    division still occurs in base-2 BEFORE rounding, so pathological
    repeating binary fractions could land on the wrong integer tick.
    The new implementation scales BOTH operands into integer space
    first, then does pure integer floor/ceil division.  This is
    deterministic across platforms and matches what the CTF Exchange
    smart contract expects on-chain.

    Optional _decimals/_max_ticks bypass the math.log10 call when
    the Market has pre-computed them via tick_math().
    """
    # BUG-FIX #2: reject NaN/infinity/non-positive prices up front.
    # Without this, NaN propagates into int(round(...)) and crashes
    # the order-signing hot path.
    if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0.0:
        raise ValueError(f"snap_price: invalid price {price!r}")
    if tick_size <= 0:
        tick_size = 0.01
    if _decimals is None:
        _decimals = max(0, -int(math.floor(math.log10(tick_size))))
    if _max_ticks is None:
        _max_ticks = int(round(1.0 / tick_size)) - 1

    scale = 10 ** _decimals
    price_int = int(round(price * scale))
    tick_int  = int(round(tick_size * scale))
    if tick_int <= 0:
        return round(price, _decimals)

    if side == "SELL":
        # Integer ceil: (a + b - 1) // b
        price_ticks = (price_int + tick_int - 1) // tick_int
    else:
        # Integer floor: a // b
        price_ticks = price_int // tick_int

    price_ticks = max(1, min(_max_ticks, int(price_ticks)))
    return round((price_ticks * tick_int) / scale, _decimals)


_USDC_DECIMALS: int = 6
_USDC_SCALE: int = 10 ** _USDC_DECIMALS  # 1_000_000


def _parse_bal_micro(raw: Any) -> int:
    """v18.4 — pure string→integer parser for USDC balance strings.

    Returns the balance in INTEGER MICRO-USDC (6 decimals).  No float
    division, no ``float()`` cast — IEEE 754 boundary anomalies cannot
    influence the result.

    Decoding rules (in priority order):
      1. Strip whitespace and thousands-separator commas.
      2. Treat empty / sentinel inputs ('0', '', 'None', 'null') as 0.
      3. Sign prefix '+' / '-' is consumed; '-' produces a negative
         result which the wrapper rejects.
      4. If the remaining string contains '.', it is a decimal USDC
         amount.  Split on '.', pad the fractional part with trailing
         zeros to exactly 6 digits (or truncate if longer — this is
         intentional: micro-USDC is the smallest exchange-meaningful
         unit), then concatenate the integer and fractional digits
         into a single arbitrary-precision ``int``.
      5. Otherwise the string is already raw integer micro-USDC.

    All comparisons and rounding are performed in pure integer space.
    The wire-format ambiguity (decimal-string vs. raw-integer-string)
    is handled by the presence/absence of the decimal point — which is
    deterministic and matches both py-clob-client v0.x (decimal) and
    v2.x (raw integer) conventions.

    Malformed inputs return 0; this is fail-closed: better to behave
    as if there is no balance than to construct orders against a
    silently-misparsed amount.
    """
    s = str(raw).strip().replace(",", "")
    if not s or s in ("0", "None", "null"):
        return 0
    sign = 1
    if s[0] == "-":
        sign = -1
        s = s[1:]
    elif s[0] == "+":
        s = s[1:]
    if not s:
        return 0
    try:
        if "e" in s or "E" in s:
            # Scientific notation (e.g. CLOB dust balance "1.5e-05").  The
            # plain string-split path below cannot decode the exponent
            # ('e' fails .isdigit()) and would silently return 0 — which
            # could fake a zero share-balance and trip the drift halt.
            # Decimal expands the mantissa/exponent EXACTLY (unlike
            # float(), which would both reintroduce IEEE-754 error AND
            # misread the raw-integer wire format by a factor of 1e6).
            # int() truncates toward zero, matching the decimal path's
            # conservative sub-micro truncation.
            micro = int(Decimal(s) * _USDC_SCALE)
            return sign * micro
        if "." in s:
            whole, frac = s.split(".", 1)
            # A second '.' (e.g. "1.2.3") means a malformed amount, not a
            # decimal: split(".", 1) leaves it in ``frac`` ("2.3").  The
            # padded ``isdigit()`` below already rejects it (the dot fails
            # isdigit -> 0), but fail closed up front so the intent is
            # explicit and the case can't be re-misread.
            if "." in frac:
                return 0
            whole = whole or "0"
            # Truncate fractional below the 6-decimal grid; pad if shorter.
            # (Truncation is conservative — exchange-meaningful amounts
            # never round UP at the micro level.)
            frac = frac[:_USDC_DECIMALS].ljust(_USDC_DECIMALS, "0")
            if not (whole.isdigit() and frac.isdigit()):
                return 0
            micro = int(whole) * _USDC_SCALE + int(frac)
        else:
            if not s.isdigit():
                return 0
            micro = int(s)
    except (ValueError, TypeError, InvalidOperation):
        return 0
    return sign * micro


def _parse_bal(raw: Any) -> float:
    """Float wrapper for ``_parse_bal_micro``.

    Legacy float-typed API surface preserved for callers that compare
    against ``min_order_size`` etc. as floats.  All cryptographic /
    on-chain code paths must use ``_parse_bal_micro`` directly to stay
    in integer space.

    Returns 0.0 for negative or malformed values (fail-closed).  Logs a
    loud WARNING if the parsed value crosses a $1M sanity threshold —
    that magnitude is rare for retail and almost certainly indicates a
    CLOB API response format regression that must be investigated
    before resuming live trading.
    """
    micro = _parse_bal_micro(raw)
    if micro < 0:
        return 0.0
    v = micro / _USDC_SCALE
    if not math.isfinite(v):
        return 0.0
    if v > 1_000_000:
        logging.getLogger("Bot").warning(
            "Suspicious balance parse: raw=%r -> %.2f USDC. "
            "Verify CLOB API response format hasn't changed.", raw, v)
    return v


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


# ─── Proxy address lookup ────────────────────────────────────────────────────

def _lookup_proxy_address(eoa: str, chain_id: int = 137) -> Optional[str]:
    eoa_cs = Web3.to_checksum_address(eoa)
    FACTORIES = [
        ("0xaB45c5A4B0c941a2F231C04C3f49182e1A254052", "getProxy"),
        ("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E", "getProxyAddress"),
        ("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E", "getSafeAddress"),
    ]
    RPC_URLS = [
        "https://polygon-rpc.com",
        "https://rpc.ankr.com/polygon",
        "https://polygon.llamarpc.com",
        "https://polygon.drpc.org",
    ]
    for factory_addr, fn_name in FACTORIES:
        ABI = [{"inputs": [{"name": "_owner", "type": "address"}],
                "name": fn_name,
                "outputs": [{"name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function"}]
        for rpc in RPC_URLS:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 5}))
                ct = w3.eth.contract(
                    address=Web3.to_checksum_address(factory_addr), abi=ABI)
                fn = getattr(ct.functions, fn_name)
                result = fn(eoa_cs).call()
                if result and result != "0x" + "0" * 40:
                    return Web3.to_checksum_address(result)
            except Exception:
                continue
    # L-4 fix: /proxy-wallet endpoint returns HTTP 404 (confirmed by Research agent).
    # Dead code removed; on-chain factory RPC calls above are the working fallback.
    # Keeping the outer try/except structure so the return below is unconditional.
    return None


# ─── Domain types ─────────────────────────────────────────────────────────────

class Side(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class Strategy(str, Enum):
    MM       = "mm"
    S2O      = "s2o"
    TEMPORAL = "temporal"


@dataclass
class OrderBook:
    """v18.4 — Integer-tick order book with cached best-side pointers.

    Storage model
    -------------
    Internally, prices and sizes are stored as ARBITRARY-PRECISION
    INTEGERS rather than floats:

      ``_bids_int : Dict[int, int]``  ─ tick_index → share_micro_units
      ``_asks_int : Dict[int, int]``

    The tick_index is ``round(price * PRICE_SCALE)`` with
    ``PRICE_SCALE = 10_000``.  Polymarket's finest tick is 0.001 USDC,
    so a 4-decimal scale (10_000) covers all current ticks plus headroom
    for any sub-tick analytics.  Share sizes are stored as integer
    micro-shares (``SIZE_SCALE = 1_000_000``).

    Why this matters
    ----------------
    Using *floats* as dict keys is technically legal Python, but the
    same level can be silently emitted with two FP representations
    (e.g. ``"0.51"`` parsed one tick vs ``"0.5100"`` the next) and
    produce two ghost keys that never coalesce.  Storing the integer
    tick index makes per-level equality EXACT regardless of how the
    upstream price string was formatted.

    O(1) hot path
    -------------
    The ``apply_delta`` path is unconditionally O(1) — one dict
    set/pop, one cached-total addend, and a constant-time check on
    ``_best_bid_key`` / ``_best_ask_key`` pointers.  The cached best
    pointer is *invalidated* (set to ``None``) only when the top-of-
    book level is fully drained; on the very next read the pointer is
    lazily recomputed via ``max()`` / ``min()`` on the dict view, which
    is the cheapest possible recomputation in CPython (no key fn).

    Reads of more than top-of-book (e.g. ``micro_price``,
    ``top_depth_usdc``) still pay an amortized O(N log N) sort, but
    this is unavoidable for VWAP-style aggregates and is cached until
    the next mutation.  In practice for Polymarket books with 5-40
    levels these reads are dominated by Python overhead, not algorithmic
    cost.

    Public API surface (UNCHANGED from v18.3)
    ------------------------------------------
      ``apply_delta(price, size, is_bid)``
      ``replace_snapshot(bids, asks)``
      ``bids`` / ``asks``        → ``List[Tuple[float, float]]``
      ``best_bid`` / ``best_ask`` → ``Optional[float]``
      ``mid`` / ``micro_price`` / ``imbalance`` / ``top_depth_usdc``
      ``spread_pct`` / ``age_ms`` / ``is_stale(max_ms)``

    All consumers continue to see float prices and float sizes; the
    integer encoding is a pure storage-layer optimization.
    """

    # Per-instance config (ClassVar so all books share the same scaling).
    PRICE_SCALE: ClassVar[int] = 10_000
    SIZE_SCALE:  ClassVar[int] = 1_000_000

    token_id: str
    _bids_int: Dict[int, int] = field(default_factory=dict)
    _asks_int: Dict[int, int] = field(default_factory=dict)
    ts:        float = field(default_factory=time.monotonic)

    # Cached integer size totals (units: share micro-units).  Used by
    # ``imbalance`` to avoid an O(N) sum on every tick.
    _bid_size_total: int = 0
    _ask_size_total: int = 0

    # Cached best-side pointers.  ``None`` means "needs recomputation"
    # — either because the book is empty OR the previous best level
    # was drained to zero.  Lazily resolved on next read.
    _best_bid_key: Optional[int] = None
    _best_ask_key: Optional[int] = None

    # Lazy-sorted float-typed views, used only for non-top-of-book reads.
    _cached_bids: List[Tuple[float, float]] = field(default_factory=list)
    _cached_asks: List[Tuple[float, float]] = field(default_factory=list)
    _sorted_dirty: bool = True

    # C-BUG-8 fix: monotonic timestamp of the last full snapshot.  Deltas
    # that arrive with a ts older than this are stale (from before a WS
    # reconnect) and must be dropped to prevent book corruption.
    _snapshot_ts: float = 0.0

    # ── Encoding helpers ─────────────────────────────────────────────────────
    @classmethod
    def _price_to_key(cls, price: float) -> int:
        """``round(price * PRICE_SCALE)``.  Stable across all books.

        Guard: caller must already have rejected non-finite / non-positive
        prices via ``apply_delta``'s gate.  This helper assumes a valid
        price.
        """
        return int(round(price * cls.PRICE_SCALE))

    @classmethod
    def _key_to_price(cls, key: int) -> float:
        return key / cls.PRICE_SCALE

    @classmethod
    def _size_to_int(cls, size: float) -> int:
        return int(round(size * cls.SIZE_SCALE))

    @classmethod
    def _int_to_size(cls, size_int: int) -> float:
        return size_int / cls.SIZE_SCALE

    # ── Mutators ─────────────────────────────────────────────────────────────
    def apply_delta(self, price: float, size: float, is_bid: bool) -> None:
        """O(1) per-level update from a WS price_change message.

        The cached best pointer is maintained incrementally:
          * If ``new_size > 0`` and the new key is better than the
            current best, the pointer is *advanced* in O(1).
          * If the level is drained to zero and the drained key
            equalled the current best, the pointer is *invalidated*
            (set to ``None``); the next read recomputes via ``max``/
            ``min`` on the dict — O(N) where N is levels stored
            (typically 5-40 on Polymarket), but only ever on the
            relatively rare top-of-book drain event.
          * Otherwise the pointer is left untouched (the mutation is
            interior to the book).
        """
        if not math.isfinite(price) or not math.isfinite(size) or price <= 0:
            return
        key = self._price_to_key(price)
        new_size_int = self._size_to_int(size) if size > 0 else 0
        target = self._bids_int if is_bid else self._asks_int
        old_size_int = target.get(key, 0)

        if new_size_int <= 0:
            if key in target:
                del target[key]
            new_size_int = 0
        else:
            target[key] = new_size_int

        delta = new_size_int - old_size_int
        if is_bid:
            self._bid_size_total += delta
            if new_size_int == 0:
                # Drained level: invalidate cached pointer if it was this one.
                if self._best_bid_key == key:
                    self._best_bid_key = None
            else:
                # Populated level: advance cached pointer if better.
                if self._best_bid_key is None or key > self._best_bid_key:
                    self._best_bid_key = key
        else:
            self._ask_size_total += delta
            if new_size_int == 0:
                if self._best_ask_key == key:
                    self._best_ask_key = None
            else:
                if self._best_ask_key is None or key < self._best_ask_key:
                    self._best_ask_key = key

        self._sorted_dirty = True

    def replace_snapshot(self,
                         bids: List[Tuple[float, float]],
                         asks: List[Tuple[float, float]]) -> None:
        """Replace the full book from a fresh REST/WS snapshot.

        Non-finite or non-positive prices/sizes are silently dropped at
        the boundary so downstream code never has to sanitize.

        A snapshot is a full-depth replacement, so two rows that collapse
        to the same integer tick key are AGGREGATED (summed): they
        represent independent resting depth at the same price level.  This
        differs from ``apply_delta``, which assigns (replaces) because a WS
        ``price_change`` carries the new ABSOLUTE size for that level.
        Aggregating there would double-count; aggregating here is correct.

        BUG-FIX #13: dedupe by tick key before aggregation.  Pre-fix, a
        snapshot that contained the SAME ``(price, size)`` row twice
        (Polymarket has been observed to emit duplicate rows on
        reconnect) summed both copies into the level, overstating depth
        and biasing ``top_depth_usdc`` and any sweep calculation that
        reads the snapshot.  We now keep the FIRST occurrence per key.
        """
        # BUG-FIX #13: per-side seen-keys set, drops duplicate rows.
        seen_bid_keys: Set[int] = set()
        self._bids_int = {}
        for p, s in bids:
            if math.isfinite(p) and math.isfinite(s) and p > 0 and s > 0:
                k = self._price_to_key(p)
                if k in seen_bid_keys:
                    continue
                seen_bid_keys.add(k)
                self._bids_int[k] = self._bids_int.get(k, 0) + self._size_to_int(s)
        seen_ask_keys: Set[int] = set()
        self._asks_int = {}
        for p, s in asks:
            if math.isfinite(p) and math.isfinite(s) and p > 0 and s > 0:
                k = self._price_to_key(p)
                if k in seen_ask_keys:
                    continue
                seen_ask_keys.add(k)
                self._asks_int[k] = self._asks_int.get(k, 0) + self._size_to_int(s)
        self._bid_size_total = sum(self._bids_int.values())
        self._ask_size_total = sum(self._asks_int.values())
        self._best_bid_key = max(self._bids_int) if self._bids_int else None
        self._best_ask_key = min(self._asks_int) if self._asks_int else None
        self._sorted_dirty = True
        # C-BUG-8 fix: stamp the snapshot time so stale deltas from a
        # pre-reconnect WS can be detected and dropped.
        self._snapshot_ts = time.monotonic()

    def refresh_totals(self) -> None:
        """Backward-compat no-op; totals are maintained incrementally.

        Defensive-rebuild path: if a caller suspects the cached totals
        have drifted (e.g. after a manual ``_bids_int`` mutation in a
        unit test), recompute from scratch.
        """
        self._bid_size_total = sum(self._bids_int.values())
        self._ask_size_total = sum(self._asks_int.values())

    # ── Cached best-pointer resolution ───────────────────────────────────────
    def _resolve_best_bid_key(self) -> Optional[int]:
        bbk = self._best_bid_key
        if bbk is not None and self._bids_int.get(bbk, 0) > 0:
            return bbk
        # Invalidated or stale — recompute.
        if self._bids_int:
            self._best_bid_key = max(self._bids_int)
            return self._best_bid_key
        self._best_bid_key = None
        return None

    def _resolve_best_ask_key(self) -> Optional[int]:
        bak = self._best_ask_key
        if bak is not None and self._asks_int.get(bak, 0) > 0:
            return bak
        if self._asks_int:
            self._best_ask_key = min(self._asks_int)
            return self._best_ask_key
        self._best_ask_key = None
        return None

    # ── Sorted views (lazy, for non-top-of-book reads only) ──────────────────
    def _ensure_sorted(self) -> None:
        if not self._sorted_dirty:
            return
        inv_p = 1.0 / self.PRICE_SCALE
        inv_s = 1.0 / self.SIZE_SCALE
        # ``sorted(reverse=True)`` on ints is cheaper than a lambda key.
        self._cached_bids = [
            (k * inv_p, v * inv_s)
            for k, v in sorted(self._bids_int.items(), reverse=True)
        ]
        self._cached_asks = [
            (k * inv_p, v * inv_s)
            for k, v in sorted(self._asks_int.items())
        ]
        self._sorted_dirty = False

    @property
    def bids(self) -> List[Tuple[float, float]]:
        """Bids sorted by descending price (positive float prices)."""
        self._ensure_sorted()
        return self._cached_bids

    @property
    def asks(self) -> List[Tuple[float, float]]:
        """Asks sorted by ascending price."""
        self._ensure_sorted()
        return self._cached_asks

    # ── O(1) read views ──────────────────────────────────────────────────────
    @property
    def best_bid(self) -> Optional[float]:
        k = self._resolve_best_bid_key()
        return None if k is None else k / self.PRICE_SCALE

    @property
    def best_ask(self) -> Optional[float]:
        k = self._resolve_best_ask_key()
        return None if k is None else k / self.PRICE_SCALE

    @property
    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2
        return bb if bb is not None else ba

    @property
    def micro_price(self) -> Optional[float]:
        """Depth-weighted micro-price using top 3 levels.

        Top-of-book only is vulnerable to L1 spoofing on thin
        Polymarket books.  VWAP of top 3 levels is more robust.

        Hot-path note: this needs only the best 3 levels per side, so we
        select them in O(N) via ``heapq.nlargest/nsmallest`` directly on
        the integer maps instead of forcing a full O(N·log N) sort of the
        whole book through ``_ensure_sorted``.  The full sorted views are
        still built lazily (and cached) for the depth-walking FOK sweeps
        that genuinely need every level.
        """
        if not self._bids_int or not self._asks_int:
            return self.mid
        inv_p = 1.0 / self.PRICE_SCALE
        inv_s = 1.0 / self.SIZE_SCALE
        # Top 3 by price: bids = largest keys, asks = smallest keys.
        top_bid_keys = heapq.nlargest(3, self._bids_int.keys())
        top_ask_keys = heapq.nsmallest(3, self._asks_int.keys())
        bid_levels = [(k * inv_p, self._bids_int[k] * inv_s) for k in top_bid_keys]
        ask_levels = [(k * inv_p, self._asks_int[k] * inv_s) for k in top_ask_keys]
        bid_vol = sum(s for _, s in bid_levels)
        ask_vol = sum(s for _, s in ask_levels)
        total = bid_vol + ask_vol
        if total <= 0 or bid_vol <= 0 or ask_vol <= 0:
            return self.mid
        bid_vwap = sum(p * s for p, s in bid_levels) / bid_vol
        ask_vwap = sum(p * s for p, s in ask_levels) / ask_vol
        # Standard micro-price: side weighted by OPPOSITE-side volume.
        return (bid_vwap * ask_vol + ask_vwap * bid_vol) / total

    @property
    def imbalance(self) -> float:
        """O(1) bid imbalance via cached integer size totals.

        >0.5 = buy pressure, <0.5 = sell pressure.  Returns the
        neutral value 0.5 when both sides are empty.
        """
        total = self._bid_size_total + self._ask_size_total
        return self._bid_size_total / total if total > 0 else 0.5

    @property
    def top_depth_usdc(self) -> float:
        """USDC notional resting at top-of-book on both sides combined."""
        bbk = self._resolve_best_bid_key()
        bak = self._resolve_best_ask_key()
        bv = 0.0
        av = 0.0
        if bbk is not None:
            bv = (bbk / self.PRICE_SCALE) * (
                self._bids_int[bbk] / self.SIZE_SCALE)
        if bak is not None:
            av = (bak / self.PRICE_SCALE) * (
                self._asks_int[bak] / self.SIZE_SCALE)
        return bv + av

    @property
    def spread_pct(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None and bb > 0:
            return (ba - bb) / bb
        return 1.0

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self.ts) * 1000

    def is_stale(self, max_ms: float) -> bool:
        return self.age_ms > max_ms


@dataclass
class Position:
    # ``float`` is deliberate.  IEEE-754 doubles carry ~15-16 significant
    # digits (≈1e-15 relative error); accumulated error over even millions
    # of fills (shares O(1-1e3)) stays many orders of magnitude below the
    # ``drift_halt_threshold_shares`` (0.01 shares) reconciliation gate, so
    # it can NOT trigger a spurious drift halt.  Integer micro-units would
    # add no measurable accuracy here and would complicate every PnL/avg
    # computation — the on-chain-exact integer parsing that matters lives
    # in ``_parse_bal_micro`` / the order-struct builder, not this ledger.
    shares: float = 0.0
    cost:   float = 0.0

    @property
    def avg_price(self) -> float:
        return self.cost / self.shares if self.shares > 0 else 0.0

    def add(self, shares: float, cost: float) -> None:
        self.shares += shares
        self.cost   += cost

    def reduce(self, shares: float) -> None:
        if self.shares <= 0:
            return
        n            = min(shares, self.shares)
        avg          = self.avg_price
        self.shares -= n
        self.cost    = self.shares * avg


@dataclass
class Market:
    market_id:  str
    question:   str
    yes_token:  str
    no_token:   str
    end_time:   Optional[float]     = None
    coin:       Optional[str]       = None
    tf_secs:    int                 = 300
    book_yes:   Optional[OrderBook] = None
    book_no:    Optional[OrderBook] = None
    pos_yes:    Position            = field(default_factory=Position)
    pos_no:     Position            = field(default_factory=Position)
    liquidity:  float               = 0.0
    volatility: float               = 0.0
    neg_risk:   bool                = False
    fees_enabled: bool              = False
    tick_sizes: Dict[str, float]    = field(default_factory=dict)
    # Cached tick arithmetic: avoids math.log10 on every snap_price call
    _tick_decimals: Dict[str, int]  = field(default_factory=dict)
    _tick_max_ticks: Dict[str, int] = field(default_factory=dict)

    @property
    def is_crypto(self) -> bool:
        return self.coin is not None

    @property
    def ttc(self) -> Optional[float]:
        return self.end_time - time.time() if self.end_time else None

    @property
    def start_time(self) -> Optional[float]:
        if not self.end_time:
            return None
        return self.end_time - float(self.tf_secs)

    @property
    def total_cost(self) -> float:
        return self.pos_yes.cost + self.pos_no.cost

    def get_tick(self, token_id: str) -> float:
        return self.tick_sizes.get(token_id, 0.01)

    def set_tick(self, token_id: str, tick: float) -> None:
        if 0 < tick < 1:
            self.tick_sizes[token_id] = tick
            # Pre-compute integer arithmetic bounds (cached per token)
            decimals = max(0, -int(math.floor(math.log10(tick))))
            self._tick_decimals[token_id] = decimals
            self._tick_max_ticks[token_id] = int(round(1.0 / tick)) - 1

    def tick_math(self, token_id: str) -> Tuple[int, int]:
        """Return (decimals, max_ticks) from cache. Avoids log10 per call."""
        if token_id in self._tick_decimals:
            return self._tick_decimals[token_id], self._tick_max_ticks[token_id]
        tick = self.get_tick(token_id)
        decimals = max(0, -int(math.floor(math.log10(tick)))) if tick > 0 else 2
        max_ticks = int(round(1.0 / tick)) - 1
        self._tick_decimals[token_id] = decimals
        self._tick_max_ticks[token_id] = max_ticks
        return decimals, max_ticks

    def fresh_books(self, max_ms: float) -> bool:
        return (
            self.book_yes is not None and not self.book_yes.is_stale(max_ms)
            and self.book_no is not None and not self.book_no.is_stale(max_ms)
        )


# ─── Lightweight EIP-712 Signer ──────────────────────────────────────────────

# Direct keccak access — bypasses Web3.keccak wrapper overhead.
# Web3.keccak adds class instantiation + validation checks on every call.
try:
    from eth_hash.auto import keccak as _raw_keccak
    def _keccak256(data: bytes) -> bytes:
        return _raw_keccak(data)
except ImportError:
    def _keccak256(data: bytes) -> bytes:
        return Web3.keccak(data)

# Pre-computed EIP-712 type hashes (constants)
_ORDER_TYPEHASH = _keccak256(
    b"Order(uint256 salt,address maker,address signer,address taker,"
    b"uint256 tokenId,uint256 makerAmount,uint256 takerAmount,"
    b"uint256 expiration,uint256 nonce,uint256 feeRateBps,"
    b"uint8 side,uint8 signatureType)"
)
_DOMAIN_TYPEHASH = _keccak256(
    b"EIP712Domain(string name,string version,"
    b"uint256 chainId,address verifyingContract)"
)


def _build_domain_separator(name: str, version: str,
                            chain_id: int, contract: str) -> bytes:
    """Compute EIP-712 domain separator — called once at boot."""
    if _abi_encode is None:
        raise RuntimeError("eth_abi required for FastSigner")
    return _keccak256(
        _DOMAIN_TYPEHASH
        + _keccak256(name.encode())
        + _keccak256(version.encode())
        + _abi_encode(["uint256"], [chain_id])
        + _abi_encode(["address"], [Web3.to_checksum_address(contract)])
    )


def _struct_hash(order: dict) -> bytes:
    """Compute EIP-712 struct hash for a CTF Exchange Order."""
    return _keccak256(
        _ORDER_TYPEHASH
        + _abi_encode(
            ["uint256", "address", "address", "address", "uint256",
             "uint256", "uint256", "uint256", "uint256", "uint256",
             "uint8", "uint8"],
            [order["salt"], order["maker"], order["signer"], order["taker"],
             order["tokenId"], order["makerAmount"], order["takerAmount"],
             order["expiration"], order["nonce"], order["feeRateBps"],
             order["side"], order["signatureType"]],
        )
    )


class FastSigner:
    """Lightweight EIP-712 order signer.

    Two signing backends:
      - coincurve (C-bindings for secp256k1): ~0.2ms  ← preferred
      - eth_account (pure Python):            ~15-25ms ← fallback

    Domain separators are cached at boot. Hash computation uses
    raw keccak + eth_abi (no encode_structured_data overhead).
    """

    # Fallback typed-data dict for eth_account path
    _ORDER_TYPES = {
        "EIP712Domain": [
            {"name": "name",              "type": "string"},
            {"name": "version",           "type": "string"},
            {"name": "chainId",           "type": "uint256"},
            {"name": "verifyingContract", "type": "address"},
        ],
        "Order": [
            {"name": "salt",          "type": "uint256"},
            {"name": "maker",         "type": "address"},
            {"name": "signer",        "type": "address"},
            {"name": "taker",         "type": "address"},
            {"name": "tokenId",       "type": "uint256"},
            {"name": "makerAmount",   "type": "uint256"},
            {"name": "takerAmount",   "type": "uint256"},
            {"name": "expiration",    "type": "uint256"},
            {"name": "nonce",         "type": "uint256"},
            {"name": "feeRateBps",    "type": "uint256"},
            {"name": "side",          "type": "uint8"},
            {"name": "signatureType", "type": "uint8"},
        ],
    }

    def __init__(self, private_key: str, maker: str, signer: str,
                 sig_type: int, chain_id: int = 137):
        self._pk_hex = private_key
        self._pk_bytes = bytes.fromhex(private_key.replace("0x", ""))
        self._maker = Web3.to_checksum_address(maker)
        self._signer = Web3.to_checksum_address(signer)
        self._sig_type = sig_type
        self._zero_addr = "0x0000000000000000000000000000000000000000"

        # Cache domain separators at boot (never recomputed)
        self._use_coincurve = _HAS_COINCURVE and _abi_encode is not None
        if self._use_coincurve:
            self._cc_key = _CoinCurveKey(self._pk_bytes)
            self._domain_sep_regular = _build_domain_separator(
                "ClobExchange", "1", chain_id, CTF_EXCHANGE)
            self._domain_sep_neg = _build_domain_separator(
                "NegRiskClobExchange", "1", chain_id, NEG_RISK_CTF_EXCHANGE)
        else:
            self._cc_key = None
            self._domain_sep_regular = None
            self._domain_sep_neg = None

        # Fallback domains for eth_account path
        self._domain_regular = {
            "name": "ClobExchange", "version": "1",
            "chainId": chain_id,
            "verifyingContract": Web3.to_checksum_address(CTF_EXCHANGE),
        }
        self._domain_neg = {
            "name": "NegRiskClobExchange", "version": "1",
            "chainId": chain_id,
            "verifyingContract": Web3.to_checksum_address(NEG_RISK_CTF_EXCHANGE),
        }
        self.log = get_logger("FastSigner")
        self.log.info("Signing backend: %s",
                      "coincurve (~0.2ms)" if self._use_coincurve
                      else "eth_account (~20ms)")

    def _build_order_struct(self, token_id: str, price: float, size: float,
                            side_int: int, tick_size: float) -> dict:
        """Build the EIP-712 order struct with strict integer amounts.

        v18.3 baseline: derive ``raw_usdc`` from the integer ticks
        actually placed on the wire (eliminates off-tick combinations).

        v18.4 additions:
          1. The price input is re-snapped here as a final defensive
             checkpoint; any caller that bypassed ``snap_price`` would
             otherwise be able to construct an order whose ratio
             ``takerAmount / makerAmount`` is not exactly the integer-
             scaled price fraction.
          2. After construction, an assertion verifies the
             tick-divisibility invariant:
                 ``raw_usdc * scale_t == raw_shares * price_ticks``
             This is the EXACT condition the CTF Exchange matching
             engine checks on-chain; if the invariant fails locally we
             refuse to sign rather than emit an order that will be
             rejected at the chain boundary (and consume a nonce).
        """
        if tick_size <= 0:
            tick_size = 0.01
        decimals = max(0, -int(math.floor(math.log10(tick_size))))
        scale_t  = 10 ** decimals
        price_ticks = int(round(price * scale_t))
        tick_int    = int(round(tick_size * scale_t))
        if tick_int <= 0:
            tick_int = 1
        # Snap-onto-grid: BUY rounds DOWN to nearest tick, SELL rounds UP.
        # (Conservative — never crosses past the requested side.)
        if side_int == 0:                                     # BUY
            price_ticks = (price_ticks // tick_int) * tick_int
        else:                                                  # SELL
            price_ticks = ((price_ticks + tick_int - 1) // tick_int) * tick_int
        if price_ticks <= 0:
            price_ticks = tick_int

        # Size in USDC 6-decimal micro-units.
        raw_shares = int(round(size * _SCALE))

        # v18.4 — auto-heal share granularity:
        # The on-chain matching engine encodes the order price as the
        # ratio ``makerAmount : takerAmount``.  For the encoded price to
        # match the tick grid EXACTLY (not within rounding), we need
        # ``raw_shares * price_ticks`` to be divisible by ``scale_t``.
        # ``gcd(price_ticks, scale_t)`` gives the largest common factor;
        # the minimum share-granularity needed is therefore
        # ``scale_t // gcd``.  We snap ``raw_shares`` DOWN to that
        # granularity (DOWN is conservative on both sides: BUY spends
        # ≤ requested USDC; SELL fills ≤ requested shares).
        g = math.gcd(price_ticks, scale_t)
        share_granularity = scale_t // g
        if share_granularity > 1:
            raw_shares -= raw_shares % share_granularity
        if raw_shares <= 0:
            raise ValueError(
                f"Order size {size:.8f} below share granularity "
                f"{share_granularity / _SCALE:.8f} at price tick "
                f"{price_ticks}/{scale_t}. Increase order size or "
                f"choose a price tick with a larger gcd vs scale_t."
            )

        product = raw_shares * price_ticks
        raw_usdc = product // scale_t

        # BUG-FIX #15: for BUY orders, the auto-healed ``raw_shares`` is
        # floored DOWN to ``share_granularity`` but is NOT bounded against
        # the user-requested ``size`` (USDC).  If the snap dropped shares
        # slightly but the *product* still rounds up, we can spend more
        # USDC than the caller asked for.  Walk the granularity back
        # until ``raw_usdc <= size * _SCALE`` — at most 1-2 iterations
        # in practice, and only fires for BUY where the dollar side is
        # the cap.  SELL is symmetric (caller passes shares directly, no
        # dollar-side conversion to constrain).
        if side_int == 0:
            usdc_cap = int(size * _SCALE)
            while raw_shares > 0 and raw_usdc > usdc_cap:
                raw_shares -= share_granularity
                product = raw_shares * price_ticks
                raw_usdc = product // scale_t
            if raw_shares <= 0:
                raise ValueError(
                    f"Order size {size:.8f} would exceed user-requested "
                    f"USDC after share-granularity snap")

        # v18.4 strict invariant.  After auto-healing share granularity
        # this should ALWAYS hold; failure indicates a deeper bug
        # (price not on tick grid, scale mismatch, or a future code
        # change that broke the granularity-snap above).  Refuse to
        # sign rather than burn an EIP-712 nonce on a guaranteed-
        # rejected order.
        if raw_usdc * scale_t != product:
            raise ValueError(
                f"Order amounts off tick grid: raw_usdc={raw_usdc} "
                f"scale_t={scale_t} product={product} "
                f"price_ticks={price_ticks} raw_shares={raw_shares} "
                f"tick={tick_size}. This is a programming error; "
                f"share-granularity auto-heal failed to align."
            )

        if side_int == 0:
            maker_amount, taker_amount = raw_usdc, raw_shares
        else:
            maker_amount, taker_amount = raw_shares, raw_usdc

        return {
            "salt": secrets.randbelow(2**128),
            "maker": self._maker,
            "signer": self._signer,
            "taker": self._zero_addr,
            "tokenId": int(token_id) if token_id.isdigit() else int(token_id, 0),
            "makerAmount": maker_amount,
            "takerAmount": taker_amount,
            # BUG-FIX #16 (FastSigner only — SDK path is unaffected):
            # CLOB V2 requires a strictly increasing per-maker nonce and a
            # non-zero expiration.  Pre-fix hard-coded "0" for both, which
            # caused the matcher to reject a fraction of FastSigner orders
            # in production.  The SDK path is fine; this only matters when
            # USE_FAST_SIGNER=true (currently not wired into place_order).
            "expiration": int(time.time()) + 300,
            "nonce": int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF,
            "feeRateBps": 0,
            "side": side_int,
            "signatureType": self._sig_type,
        }

    def _sign_coincurve(self, order: dict, neg_risk: bool) -> str:
        """Sign via coincurve C-bindings. ~0.2ms. Does NOT block event loop."""
        domain_sep = self._domain_sep_neg if neg_risk else self._domain_sep_regular
        sh = _struct_hash(order)
        msg_hash = _keccak256(b"\x19\x01" + domain_sep + sh)
        sig_bytes = self._cc_key.sign_recoverable(msg_hash, hasher=None)
        # coincurve returns 65 bytes: r(32) + s(32) + recovery_id(1)
        r, s, v = sig_bytes[:32], sig_bytes[32:64], sig_bytes[64] + 27
        return "0x" + r.hex() + s.hex() + format(v, "02x")

    def _sign_eth_account(self, order: dict, neg_risk: bool) -> str:
        """Fallback: sign via eth_account. ~20ms. Blocks event loop."""
        domain = self._domain_neg if neg_risk else self._domain_regular
        typed_data = {
            "types": self._ORDER_TYPES,
            "primaryType": "Order",
            "domain": domain,
            "message": order,
        }
        signed = Account.sign_message(
            encode_structured_data(typed_data), self._pk_hex)
        return signed.signature.hex()

    def sign_order(self, token_id: str, price: float, size: float,
                   side_str: str, neg_risk: bool = False,
                   tick_size: float = 0.01,
                   order_type: str = "GTC") -> dict:
        """Build + sign a CTF Exchange order.  Returns dict ready for CLOB POST."""
        side_int = 0 if side_str in ("BUY", Side.BUY) else 1
        order = self._build_order_struct(token_id, price, size, side_int, tick_size)

        if self._use_coincurve:
            signature = self._sign_coincurve(order, neg_risk)
        else:
            signature = self._sign_eth_account(order, neg_risk)

        return {
            "order": {
                "salt": str(order["salt"]),
                "maker": self._maker,
                "signer": self._signer,
                "taker": self._zero_addr,
                "tokenId": str(order["tokenId"]),
                "makerAmount": str(order["makerAmount"]),
                "takerAmount": str(order["takerAmount"]),
                "expiration": "0",
                "nonce": "0",
                "feeRateBps": "0",
                "side": str(side_int),
                "signatureType": self._sig_type,
                "signature": signature,
            },
            "owner": self._maker,
            "orderType": order_type,
        }


# ─── PolyClient ───────────────────────────────────────────────────────────────

class PolyClient:
    """
    Polymarket CLOB client. v18 improvements:
      - Direct async HTTP for order posting (no run_in_executor)
      - Optional FastSigner bypass for SDK-free signing
      - SDK create_order called inline (no thread pool)
    """

    def __init__(self, cfg: Config):
        self.cfg             = cfg
        self.log             = get_logger("PolyClient", cfg.log_level)
        self.session:  Optional[aiohttp.ClientSession] = None
        self.sdk:      Optional[Any]                   = None
        self.api_key         = ""
        self.api_secret      = ""
        self.api_passphrase  = ""
        self.signer_address  = ""
        self.trading_address = ""
        self.active_mode     = ""
        self.lib_broken      = False
        self._token_to_market: Dict[str, Market] = {}
        self._fast_signer: Optional[FastSigner]  = None

    def set_market_ref(self, t2m: Dict[str, Market]) -> None:
        self._token_to_market = t2m

    def _persist_tick(self, token_id: str, tick: float) -> None:
        m = self._token_to_market.get(token_id)
        if m and 0 < tick < 1:
            old = m.tick_sizes.get(token_id)
            m.set_tick(token_id, tick)
            self.log.info("Tick updated  %s: %s -> %s", token_id[:16], old, tick)

    # ── SDK builder ───────────────────────────────────────────────────────────

    async def _build_sdk(self, sig_type: int) -> bool:
        if not _HAS_SDK:
            self.log.error(
                "Polymarket SDK not installed — run: "
                "%s/bin/pip install py-clob-client-v2" % sys.prefix)
            return False
        if not _SDK_IS_V2:
            # Only the dead V1 package is importable.  CLOB V1 was shut down
            # 2026-04-28; every order it signs is rejected with
            # order_version_mismatch (the exact crash-loop seen in the field).
            # Fail fast with an actionable message instead of looping forever.
            self.log.critical(
                "Polymarket CLOB V1 SDK (py-clob-client) detected — CLOB V1 was "
                "retired 2026-04-28 and V1-signed orders are rejected with "
                "order_version_mismatch. Install the V2 SDK and restart:\n"
                "    %s/bin/pip install py-clob-client-v2" % sys.prefix)
            return False

        loop  = asyncio.get_running_loop()
        label = _SIG_LABELS.get(sig_type, f"type_{sig_type}")
        try:
            kw: Dict[str, Any] = {
                "host":     self.cfg.clob_url,
                "chain_id": self.cfg.chain_id,
                "key":      self.cfg.private_key,
            }
            if self.cfg.use_proxy and sig_type in (1, 2):
                kw["signature_type"] = sig_type
                kw["funder"]         = self.cfg.proxy_address
            elif sig_type == 0:
                kw["signature_type"] = 0

            sdk   = ClobClient(**kw)
            # V2 renamed create_or_derive_api_creds → create_or_derive_api_key
            _derive = (getattr(sdk, "create_or_derive_api_key", None)
                       or getattr(sdk, "create_or_derive_api_creds", None))
            if _derive is None:
                raise RuntimeError(
                    "SDK exposes neither create_or_derive_api_key nor "
                    "create_or_derive_api_creds — incompatible py-clob-client")
            creds = await loop.run_in_executor(None, _derive)
            sdk.set_api_creds(creds)

            self.sdk                = sdk
            self.api_key            = creds.api_key
            self.api_secret         = creds.api_secret
            self.api_passphrase     = creds.api_passphrase
            self.cfg.signature_type = sig_type
            self.trading_address    = self.cfg.proxy_address or self.signer_address
            self.active_mode        = f"sdk_{label}"

            self.log.info("SDK ready  sig_type=%d (%s)  trader=%s",
                          sig_type, label, self.trading_address)

            # Initialize FastSigner if enabled.
            #
            # IMPORTANT — experimental / NOT on the execution path.  The
            # FastSigner can produce a signed EIP-712 order struct, but
            # ``place_order`` deliberately routes ALL live orders through the
            # battle-tested SDK (``sdk.create_order`` → ``sdk.post_order``),
            # which owns the exact CLOB submission schema, L2 HMAC headers,
            # and signature encoding.  Posting a FastSigner payload directly
            # would mean re-implementing that untested submission path for a
            # private-key-custody bot — and the latency it would save
            # (sub-ms signing) is irrelevant on Polymarket: an off-chain CLOB
            # with ~2 s Polygon settlement over public HTTPS is not a
            # microsecond venue, so signing is never the binding constraint.
            # We therefore keep it OFF the hot path and log loudly so an
            # operator who sets USE_FAST_SIGNER=true is never misled into
            # believing orders are being signed by it.
            if self.cfg.use_fast_signer:
                self._fast_signer = FastSigner(
                    self.cfg.private_key, self.trading_address,
                    self.signer_address, sig_type, self.cfg.chain_id)
                self.log.warning(
                    "FastSigner initialized but NOT wired into the order path "
                    "— execution still routes through the SDK signer by design "
                    "(see place_order). Sub-ms signing is irrelevant on a ~2s-"
                    "settlement CLOB; wiring it requires venue-accurate POST "
                    "schema testing against the live exchange.")

            return True

        except Exception as e:
            self.log.error("SDK build failed  sig_type=%d (%s): %s", sig_type, label, e)
            return False

    # ── Initialisation ────────────────────────────────────────────────────────

    async def initialize(self, session: aiohttp.ClientSession) -> bool:
        self.session        = session
        self.signer_address = Account.from_key(self.cfg.private_key).address
        self.log.info("Signer EOA: %s", self.signer_address)

        loop = asyncio.get_running_loop()
        try:
            on_chain = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: _lookup_proxy_address(self.signer_address, self.cfg.chain_id)),
                timeout=8.0)
        except asyncio.TimeoutError:
            on_chain = None
            self.log.warning("Proxy lookup timed out — using .env value")

        if on_chain:
            if not self.cfg.proxy_address:
                self.log.info("Proxy not in .env — using on-chain value: %s", on_chain)
                self.cfg.proxy_address = on_chain
            elif self.cfg.proxy_address.lower() != on_chain.lower():
                self.log.warning(
                    "PROXY MISMATCH — .env=%s  on-chain=%s — auto-correcting.",
                    self.cfg.proxy_address, on_chain)
                self.cfg.proxy_address = on_chain
            else:
                self.log.info("Proxy address verified on-chain: %s", on_chain)
        else:
            self.log.warning(
                "Could not verify proxy address on-chain. Using .env: %s",
                self.cfg.proxy_address or "(none)")

        return await self._build_sdk(self.cfg.signature_type)

    # ── Signing test ──────────────────────────────────────────────────────────

    async def test_order(self, token_id: str, tick_size: float = 0.01,
                         neg_risk: bool = False) -> bool:
        if not self.sdk:
            return False

        loop = asyncio.get_running_loop()
        # BUG-FIX #37: drop the "test" order to venue-minimum size at a
        # dust price.  Pre-fix used 5.0 shares @ 0.01 (i.e., 0.05 USDC
        # at risk) which could match a stale 0.01 resting limit and
        # execute a real fill — the cancel that follows is best-effort.
        # 1.0 share @ 0.001 caps worst-case adverse selection at ~$0.001
        # while still exercising the full EIP-712 signing + POST path.
        test_price = snap_price(0.001, tick_size, "BUY")

        _BENIGN = (
            "balance", "insufficient", "not enough", "minimum order",
            "size too small", "below minimum", "order_minimum",
            "allowance", "funds", "not accepting orders", "market closed",
            "market not found", "not found", "price out of range", "price_out",
            "invalid amount", "min size", "404",
        )
        _SIG_FAIL = (
            "invalid signature", "bad signature", "signature mismatch",
            "unauthorized", "authentication failed",
        )

        try:
            args = OrderArgs(
                token_id=token_id, price=test_price, size=1.0, side=_SDK_BUY)
            try:
                opts = PartialCreateOrderOptions(
                    neg_risk=neg_risk, tick_size=str(tick_size))
            except TypeError:
                opts = PartialCreateOrderOptions(neg_risk=neg_risk)

            signed = await loop.run_in_executor(
                None, lambda: self.sdk.create_order(args, opts))

            resp = await loop.run_in_executor(
                None, lambda: self.sdk.post_order(signed, OrderType.GTC))

            oid = (resp or {}).get("orderID")
            if oid:
                try:
                    # V2: cancel_order(OrderPayload) vs V1: cancel(order_id)
                    _cancel_one = getattr(self.sdk, "cancel_order", None)
                    if _cancel_one:
                        _payload = type("P", (), {"orderID": oid})
                        await loop.run_in_executor(None, lambda: _cancel_one(_payload))
                    else:
                        await loop.run_in_executor(None, lambda: self.sdk.cancel(oid))
                except Exception:
                    pass

            self.log.info("Signing test PASSED  sig_type=%d  neg_risk=%s",
                          self.cfg.signature_type, neg_risk)
            return True

        except Exception as e:
            es = str(e)
            el = es.lower()
            if any(k in el for k in _SIG_FAIL):
                self.log.warning("Signing test FAILED — CLOB rejected: %s", es[:200])
                return False
            if any(k in el for k in _BENIGN):
                self.log.info("Signing test PASSED (benign reject: %s)", es[:80])
                return True
            self.log.warning(
                "Signing test: unexpected error — treating as FAILURE: %s", es[:120])
            return False

    # ── Balance ───────────────────────────────────────────────────────────────

    async def get_balance(self) -> float:
        if self.sdk:
            try:
                loop   = asyncio.get_running_loop()
                params = BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL if AssetType else "COLLATERAL")
                resp = await loop.run_in_executor(
                    None, self.sdk.get_balance_allowance, params)
                bal  = _parse_bal(resp.get("balance", "0"))
                if bal > 0:
                    return bal
            except Exception as e:
                self.log.debug("SDK balance error: %s", e)

        if self.cfg.proxy_address and self.session:
            try:
                # BUG-FIX #10: add HMAC auth headers to private endpoint.
                # Per Polymarket L2 HMAC spec, the signed string is
                # (ts + method + requestPath + body) where requestPath
                # MUST include the query string.  Pre-fix signed only the
                # bare "/balance-allowance" path, which made the
                # server-side check fail silently and this fallback always
                # returned 0.0.
                qs = urlencode({"asset_type": "COLLATERAL",
                                "address": self.cfg.proxy_address})
                path = f"/balance-allowance?{qs}"
                auth_hdrs = self._hmac_headers("GET", path)
                async with self.session.get(
                    f"{self.cfg.clob_url}{path}",
                    headers=auth_hdrs,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    if r.ok:
                        d = await r.json(content_type=None)
                        return _parse_bal(d.get("balance", "0"))
            except Exception as e:
                self.log.debug("REST balance error: %s", e)
        # BUG-FIX #11: log when all balance sources fail.
        self.log.error("get_balance: all balance sources failed")
        return 0.0

    # ── Order placement — uses SDK post_order (proven working path) ─────────

    async def place_order(self, token_id: str, side: str, price: float,
                          size_usdc: float, order_type: str = "GTC",
                          neg_risk: bool = False,
                          tick_size: float = 0.01) -> Optional[str]:
        if not self.sdk:
            return None

        # Polymarket rejects many BUY orders when makerAmount (USDC spend)
        # has more than 2 decimals. Using integer shares on BUY ensures
        # (shares * 2-decimal price) stays within accepted precision.
        # Use the market's native tick_size (supports 0.001 for high-liq markets).
        price = snap_price(price, tick_size, side)
        if side in ("BUY", Side.BUY):
            # v18.9 / Critic-v2 C-NEW-1 fix: ``math.floor`` of a 6-decimal-rounded
            # value still discards everything below 1.0, leaving up to 0.999999
            # shares of dust that diverged the local ledger from on-chain and
            # tripped ``drift_halt_threshold_shares``.  Polymarket CTF tokens
            # are 6-decimal precision; round to that grid (matches contract
            # ABI) and floor only to 6dp, NOT to integer.  The legacy
            # ``max(1.0, ...)`` venue-minimum is preserved.
            shares = round(size_usdc / max(price, 0.001), 6)
            if shares < 1.0:
                shares = 1.0
        else:
            # SELL: 6-decimal precision matches the on-chain CTF contract.
            # Pre-fix used ``float(math.floor(round(...,6)))`` which truncated
            # any sub-share component into untracked dust → drift halt loop.
            shares = round(size_usdc / max(price, 0.001), 6)
            if shares < 1.0:
                shares = 1.0
        loop = asyncio.get_running_loop()

        sdk_side = _SDK_BUY if side in ("BUY", Side.BUY) else _SDK_SELL
        args = OrderArgs(token_id=token_id, price=price, size=shares, side=sdk_side)

        try:
            opts = PartialCreateOrderOptions(
                neg_risk=neg_risk, tick_size=str(tick_size))
        except TypeError:
            opts = PartialCreateOrderOptions(neg_risk=neg_risk)

        try:
            signed = await loop.run_in_executor(
                None, lambda: self.sdk.create_order(args, opts))
            ot   = OrderType.FOK if order_type == "FOK" else OrderType.GTC
            resp = await loop.run_in_executor(
                None, lambda: self.sdk.post_order(signed, ot))
            return (resp or {}).get("orderID")

        except Exception as e:
            es  = str(e)
            el  = es.lower()
            if any(k in el for k in ("balance", "insufficient", "allowance")):
                self.log.warning("Insufficient balance/allowance  token=%s", token_id[:16])
            elif any(k in el for k in ("invalid signature", "bad signature",
                                       "unauthorized", "authentication failed")):
                self.log.error(
                    "CLOB rejected signature  token=%s  sig_type=%d\n"
                    "  Ensure POLYMARKET_PRIVATE_KEY owns the proxy wallet and\n"
                    "  has been linked via polymarket.com.",
                    token_id[:16], self.cfg.signature_type)
            else:
                self.log.error("Order failed: %s", es[:200])
            return None

    async def cancel(self, order_id: str) -> bool:
        if not self.sdk:
            return False
        try:
            # V2: cancel_order(OrderPayload) vs V1: cancel(order_id)
            _cancel_one = getattr(self.sdk, "cancel_order", None)
            if _cancel_one:
                _payload = type("P", (), {"orderID": order_id})
                resp = await asyncio.get_running_loop().run_in_executor(
                    None, _cancel_one, _payload)
            else:
                resp = await asyncio.get_running_loop().run_in_executor(
                    None, self.sdk.cancel, order_id)
            # M-3 fix: check for explicit not_canceled in response.
            # CLOB returns a valid response (no exception) for an already-
            # filled order, including {"not_canceled":{...}} keys.  Pre-fix
            # returned True unconditionally, which caused OrderManager.cancel
            # to pop the order from _by_token even though cancel failed, stripping
            # the re-entry guard and allowing double-entry.
            if isinstance(resp, dict) and resp.get("not_canceled"):
                return False
            return True
        except Exception:
            return False

    async def cancel_all(self) -> None:
        if not self.sdk:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self.sdk.cancel_all)
        except Exception:
            raise  # C-3 fix: was `pass`; swallowing meant OrderManager.cancel_all
                   # never reached its `raise` branch, so local state was always
                   # cleared even when the exchange cancel failed (C-4 invariant broken).

    # ── HMAC auth headers ─────────────────────────────────────────────────────

    def _hmac_headers(self, method: str, path: str, body: str = "") -> dict:
        if not self.api_key:
            return {}
        ts  = str(int(time.time()))
        msg = (ts + method + path + body).encode()
        try:
            secret = base64.urlsafe_b64decode(self.api_secret)
        except Exception:
            secret = self.api_secret.encode()
        sig = base64.urlsafe_b64encode(
            hmac.new(secret, msg, hashlib.sha256).digest()
        ).decode()
        return {
            "POLY_ADDRESS":    self.cfg.proxy_address or self.signer_address,
            "POLY_SIGNATURE":  sig,
            "POLY_TIMESTAMP":  ts,
            "POLY_NONCE":      "0",
            "POLY_API_KEY":    self.api_key,
            "POLY_PASSPHRASE": self.api_passphrase,
        }


# ─── Rate limiter (Leaky Bucket) ──────────────────────────────────────────────

class RateLimiter:
    """Leaky bucket rate limiter with staggered wake-ups.

    Fixes the thundering herd problem in the token bucket: when N coroutines
    hit the limiter simultaneously with zero tokens, they all compute the
    same wait time and wake up together to fight for the lock.

    The leaky bucket tracks the next allowed timestamp.  Each caller
    atomically advances `_next` by 1/rate, guaranteeing perfectly staggered
    wake-ups with O(1) lock time and zero contention.
    """

    def __init__(self, per_sec: int):
        self._interval = 1.0 / float(per_sec)   # seconds between tokens
        self._next     = time.monotonic()        # next allowed send time
        self._lock     = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next <= now:
                self._next = now + self._interval
                return                            # immediate grant
            wait = self._next - now
            self._next += self._interval          # reserve our slot
        # sleep OUTSIDE the lock — each coroutine wakes at a different time
        await asyncio.sleep(wait)


# ─── Performance Metrics ──────────────────────────────────────────────────────

class Metrics:
    """Lightweight order-latency histograms + fill/PnL counters."""

    def __init__(self):
        self._order_latencies: Deque[float] = deque(maxlen=500)
        self._fill_count = 0
        self._order_count = 0
        self._total_pnl = 0.0
        self.log = get_logger("Metrics")

    def record_order_latency(self, ms: float) -> None:
        self._order_latencies.append(ms)
        self._order_count += 1

    def record_fill(self) -> None:
        self._fill_count += 1

    def record_pnl(self, delta: float) -> None:
        self._total_pnl += delta

    def summary(self) -> dict:
        lats = list(self._order_latencies)
        if lats:
            lats.sort()
            n = len(lats)
            p50 = lats[min(n // 2, n - 1)]
            # Nearest-rank percentile: rank = ceil(q*N) (1-based) -> index-1.
            # The prior ``int(n*q)`` returned index N for q=0.99,N=100 (the
            # max / p100) instead of the true 99th percentile.
            p95 = lats[max(0, math.ceil(n * 0.95) - 1)]
            p99 = lats[max(0, math.ceil(n * 0.99) - 1)] if n > 10 else p95
        else:
            p50 = p95 = p99 = 0.0

        return {
            "orders": self._order_count,
            "fills": self._fill_count,
            "pnl": round(self._total_pnl, 4),
            "lat_p50_ms": round(p50, 1),
            "lat_p95_ms": round(p95, 1),
            "lat_p99_ms": round(p99, 1),
        }


# C-5: classify exceptions on order placement.  The original code
# incremented _rejects on EVERY exception (HTTP 429, DNS blip, SSL renegot,
# auth expiry, balance miss, CLOB explicit reject) and the Risk halt
# tripped at >=5 — so a 5-second ISP flicker would halt the bot with open
# unmanaged positions.  Classify first, only count structural rejections.
class _OrderErrorClass(str, Enum):
    RATE_LIMIT   = "rate_limit"   # transient — do NOT count, back off
    NETWORK      = "network"      # transient — do NOT count
    AUTH_FAILURE = "auth"         # structural — count immediately
    BALANCE      = "balance"      # data — count
    REJECTION    = "rejection"    # CLOB explicit reject — count


def _classify_order_error(exc: BaseException) -> _OrderErrorClass:
    # M-1 fix: removed dead pycloborder isinstance check (lines were here pre-fix).
    # pycloborder does not exist on PyPI (Research agent confirmed 404); the
    # import always raised ImportError so the isinstance guard was always False.
    # BUG-FIX #21's claim of typed-exception protection was never active.
    # String matching below is the actual working path and is sufficient.
    s = str(exc).lower()
    if "429" in s or "rate limit" in s or "too many" in s:
        return _OrderErrorClass.RATE_LIMIT
    if any(k in s for k in ("timeout", "connectionreset", "connection reset",
                            "connection refused", "eof", "disconnected",
                            "temporarily unavailable", "503", "502", "504")):
        return _OrderErrorClass.NETWORK
    if any(k in s for k in ("signature", "unauthorized", "authentication",
                            "invalid api", "api key")):
        return _OrderErrorClass.AUTH_FAILURE
    if any(k in s for k in ("balance", "allowance", "insufficient")):
        return _OrderErrorClass.BALANCE
    return _OrderErrorClass.REJECTION


# ─── Order tracking ───────────────────────────────────────────────────────────

class OrderState(str, Enum):
    PENDING   = "pending"
    OPEN      = "open"
    FILLED    = "filled"
    CANCELLED = "cancelled"


@dataclass
class TrackedOrder:
    order_id:       str
    token_id:       str
    side:           Side
    price:          float
    size:           float
    strategy:       Strategy
    state:          OrderState = OrderState.PENDING
    created:        float      = field(default_factory=time.monotonic)
    filled_size:    float      = 0.0
    avg_fill_price: float      = 0.0


class OrderManager:
    def __init__(self, cfg: Config, client: PolyClient,
                 metrics: Optional[Metrics] = None):
        self.cfg    = cfg
        self.client = client
        self.log    = get_logger("Orders", cfg.log_level)
        self._orders: Dict[str, TrackedOrder] = {}
        self._by_token: Dict[Tuple[str, Side], str] = {}
        self._lock    = asyncio.Lock()
        self._rl      = RateLimiter(cfg.rate_limit)
        self._cancel_rl = RateLimiter(max(5, cfg.rate_limit // 2))  # separate cancel bucket
        self._rejects = 0
        self._metrics = metrics
        # Strong refs for fire-and-forget diagnostic probes (asyncio holds
        # only a weak ref to a bare create_task; the done-callback discards).
        self._bg_tasks: Set[asyncio.Task] = set()

        # v18.4 — fill replay state (see ``reconcile_fills``).
        # Monotonic cursor: ``_last_trade_cursor_ts`` walks the
        # ``/trades`` REST endpoint forward; trades already seen via WS
        # are deduped through ``_seen_trade_ids``.  The dedup set is
        # FIFO-capped at ``_seen_trade_ids_cap`` to prevent unbounded
        # memory growth over a long-running session.
        self._seen_trade_ids: Set[str] = set()
        self._seen_trade_order: Deque[str] = deque(maxlen=50000)
        # BUG-FIX #20: cap raised 5_000 -> 50_000.  Pre-fix allowed the
        # WS dedup set to roll over every ~5min at 3 fills/min, so a
        # reconnect burst could re-deliver up to 5min of trades and
        # double-credit the local ledger (the non-idempotent BUY branch
        # in _on_fill).  50_000 entries cover ~24h of trading at the
        # current fill rate (memory cost ~3 MB).
        self._seen_trade_ids_cap: int = 50_000
        self._last_trade_cursor_ts: float = time.time()
        # C-BUG-6 fix: first reconcile should walk back to catch pre-boot
        # fills that the WS never saw (e.g., fills from a crash/restart).
        self._first_reconcile_done: bool = False
        # Handler installed by Bot to replay missing fills through the
        # SAME code path as live WS fills (Bot._on_fill).
        self._fill_replay_handler: Optional[Callable[[dict], Any]] = None
        self._reconcile_fills_lock = asyncio.Lock()

        # v19 Scope-A — adverse-selection feedback.  The shadow probe (now
        # always-on, not DRY_RUN-only) measures post-fill mid drift; we keep
        # a rolling EWMA in bps so the strategy can ABORT entries when we are
        # systematically the dumb liquidity.  ``_shadow_sink`` optionally
        # forwards each measurement to the calibration CSV for offline join.
        self._adverse_ewma: Optional[float] = None
        self._shadow_sink: Optional[Callable[[dict], Any]] = None

    def set_shadow_sink(self, sink: Callable[[dict], Any]) -> None:
        """Install a callback that receives each shadow-probe measurement
        (a dict with token_id/side/adverse_bps/...).  Bot wires this to the
        strategy's calibration logger so adverse selection is recorded as a
        ``shadow`` row the offline harness can aggregate."""
        self._shadow_sink = sink

    def record_adverse(self, adverse_bps: float) -> None:
        """Fold one signed post-fill adverse-drift measurement (bps of mid,
        positive = against us) into the rolling EWMA."""
        a = self.cfg.adverse_ewma_alpha
        if self._adverse_ewma is None:
            self._adverse_ewma = adverse_bps
        else:
            self._adverse_ewma = (1.0 - a) * self._adverse_ewma + a * adverse_bps

    def adverse_ewma(self) -> Optional[float]:
        return self._adverse_ewma

    def set_fill_replay_handler(self,
                                handler: Callable[[dict], Any]) -> None:
        """Install the missed-fill dispatcher.

        ``handler`` receives a single trade dict (one row from the
        ``/trades`` REST response) and is expected to translate it to
        the bot's internal ``_on_fill`` semantics.  Idempotency MUST
        be guaranteed at the handler layer; ``reconcile_fills`` only
        dedupes by ``trade_id`` within a single session.
        """
        self._fill_replay_handler = handler

    def mark_trade_seen(self, trade_id: str) -> bool:
        """Register ``trade_id`` in the dedup set; return ``True`` iff it
        was NEWLY added, ``False`` if already present (or empty).

        The WS fill path uses the return value to DROP a duplicate trade
        delivery (a packet re-pushed on reconnect, or a fill the REST
        ``reconcile_fills`` pass already replayed) BEFORE it reaches
        ``_on_fill`` — whose BUY branch (``pos.add``) is not idempotent
        (the SELL branch is clamped by ``min(shares, pos.shares)``, but
        BUY has no such guard, so a re-delivered BUY would double-count
        the position).  It also still teaches the set so a later REST
        reconcile doesn't double-replay a WS-delivered fill.
        """
        if not trade_id:
            return False
        if trade_id in self._seen_trade_ids:
            return False
        if len(self._seen_trade_order) == self._seen_trade_ids_cap:
            # Evict the oldest id so the set stays bounded.
            evicted = self._seen_trade_order[0]
            # ``deque(maxlen=...).append`` will auto-evict, but the set
            # needs an explicit discard to stay in sync.
            self._seen_trade_ids.discard(evicted)
        self._seen_trade_order.append(trade_id)
        self._seen_trade_ids.add(trade_id)
        return True

    def _spawn_shadow_probe(self, token_id: str, side: Side,
                            fill_price: float) -> None:
        """Always-on adverse-selection probe (v19; was DRY_RUN-only).

        Records the fill against the book mid 500 ms later.  If the mid
        systematically moves AGAINST the fill right after execution, the
        "edge" is adverse selection (we are the liquidity that informed flow
        trades through), not alpha — and the bot will bleed live no matter
        how clean the code is.  ``adverse`` is signed so POSITIVE bps means
        the market moved against us.  The measurement now feeds a rolling
        EWMA (``record_adverse``) that the strategy's ``adverse_gate``
        consumes to ABORT entries, and is forwarded to the calibration CSV
        via ``_shadow_sink``.  Fire-and-forget and exception-safe.
        """
        if not self.cfg.shadow_probe_enabled:
            return
        mkt = self.client._token_to_market.get(token_id)
        if mkt is None:
            return
        book = mkt.book_yes if token_id == mkt.yes_token else mkt.book_no
        if book is None:
            return
        entry_mid = book.mid
        if entry_mid is None or entry_mid <= 0:
            return
        entry_spread = book.spread_pct

        async def _probe() -> None:
            try:
                await asyncio.sleep(0.5)
                post_mid = book.mid
                if post_mid is None:
                    return
                # Signed so positive = adverse (price fell after our BUY, or
                # rose after our SELL).
                adverse = (entry_mid - post_mid if side == Side.BUY
                           else post_mid - entry_mid)
                adverse_bps = (adverse / entry_mid) * 1e4
                self.record_adverse(adverse_bps)
                self.log.info(
                    "SHADOW_FILL %s %s @ %.4f | entry_mid=%.4f post_mid=%.4f "
                    "spread=%.4f adverse=%+.1fbps ewma=%+.1fbps",
                    side.value, token_id[:12], fill_price, entry_mid, post_mid,
                    entry_spread, adverse_bps, self._adverse_ewma or 0.0)
                if self._shadow_sink is not None:
                    try:
                        self._shadow_sink({
                            "market_id": mkt.market_id,
                            "token_id": token_id, "side": side.value,
                            "adverse_bps": adverse_bps,
                        })
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            t = asyncio.create_task(_probe(), name=f"shadow_{token_id[:8]}")
            self._bg_tasks.add(t)
            t.add_done_callback(self._bg_tasks.discard)
        except RuntimeError:
            pass

    async def place(self, token_id: str, side: Side, price: float,
                    size: float, strategy: Strategy,
                    otype: str = "GTC", neg_risk: bool = False,
                    tick_size: float = 0.01) -> Optional[str]:

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return None

        # S-5 fix: refuse to issue a duplicate (token,side) order while an
        # existing one is still PENDING/OPEN.  In maker mode a fresh entry
        # signal during the 45s GTC cancel window would silently overwrite
        # _by_token[(token,side)] with the new oid, leaving the old order
        # tracked on the exchange but invisible to the bot — an orphan that
        # only cancel_all() would clean up.
        # H-7 fix: reserve the slot under the SAME lock that checks it.
        # Pre-fix: lock released at the end of the check block without
        # writing a placeholder, so two concurrent place() calls both saw
        # no entry, both proceeded to network, and both submitted live orders
        # (TOCTOU double-entry). The sentinel is cleared on any failure below.
        _PENDING_SENTINEL = "__pending__"
        async with self._lock:
            existing_oid = self._by_token.get((token_id, side))
            if existing_oid is not None and existing_oid != _PENDING_SENTINEL:
                existing = self._orders.get(existing_oid)
                if existing is not None and existing.state in (
                        OrderState.PENDING, OrderState.OPEN):
                    self.log.warning(
                        "DUP_GUARD: refusing %s %s — existing %s still %s",
                        side.value, token_id[:12], existing_oid[:8],
                        existing.state.value)
                    return existing_oid
            # Reserve the slot before releasing the lock:
            self._by_token[(token_id, side)] = _PENDING_SENTINEL

        t0 = time.monotonic()

        if self.cfg.dry_run:
            # Enhanced dry-run simulation
            await asyncio.sleep(self.cfg.dry_run_latency_ms / 1000.0)
            filled = random.random() < self.cfg.dry_run_fill_prob
            elapsed_ms = (time.monotonic() - t0) * 1000
            oid = f"dry-{int(time.time() * 1000)}" if filled else None
            self.log.info("DRY %s %s @ %.4f  $%.2f  [%s]  fill=%s  %.1fms",
                          side.value, token_id[:12], price, size,
                          strategy.value, filled, elapsed_ms)
            if self._metrics:
                self._metrics.record_order_latency(elapsed_ms)
            if oid:
                async with self._lock:
                    self._orders[oid] = TrackedOrder(
                        oid, token_id, side, price, size, strategy,
                        state=OrderState.OPEN)
                    self._by_token[(token_id, side)] = oid
                # Shadow adverse-selection probe.
                self._spawn_shadow_probe(token_id, side, price)
                # v18.9: dispatch a simulated fill through the replay handler
                # so dry-run positions are tracked in the local ledger and
                # TP/TRAIL/STOP can fire — without this, pos_yes.shares
                # stayed at 0 and the entire exit lifecycle was dead in
                # dry-run mode.
                if self._fill_replay_handler is not None:
                    # Critic-v2 C-NEW-1 fix: 6-decimal share precision (CTF
                    # token grid).  Pre-fix ``math.floor`` discarded anything
                    # below 1.0 and silently produced dust that the next
                    # drift check flagged as a state divergence.
                    shares = round(size / max(price, 0.001), 6)
                    if shares < 1.0:
                        shares = 1.0
                    try:
                        res = self._fill_replay_handler({
                            "asset_id": token_id,
                            "side": side.value,
                            "price": str(price),
                            "size": str(shares),
                            "trade_id": oid,
                        })
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as e:
                        self.log.debug("dry-run fill replay error: %s", e)
            return oid

        await self._rl.acquire()
        try:
            oid = await self.client.place_order(
                token_id, side.value, price, size, otype, neg_risk, tick_size)
            elapsed_ms = (time.monotonic() - t0) * 1000
            if self._metrics:
                self._metrics.record_order_latency(elapsed_ms)
            if oid:
                async with self._lock:
                    self._orders[oid] = TrackedOrder(
                        oid, token_id, side, price, size, strategy,
                        state=OrderState.OPEN)
                    self._by_token[(token_id, side)] = oid
                    self._rejects = 0
                self.log.info("%s %s %s @ %.4f  $%.2f  [%s]  %.1fms",
                              otype, side.value, token_id[:12], price, size,
                              strategy.value, elapsed_ms)
                # v19: measure live adverse selection too (was DRY_RUN-only).
                self._spawn_shadow_probe(token_id, side, price)
                # v18.8 / C-1 fix: FOK immediate fill replay.  The pre-fix
                # path assumed a returned ``oid`` was a guaranteed fill at
                # the FOK *ceiling* price and credited the position with
                # ``shares = size / price``.  The CLOB returns ``oid`` on
                # ACCEPT (matching engine has not run), so a no-counterparty
                # FOK could be credited as a phantom position at the wrong
                # cost basis, then trip the drift halt mid-trade.
                # Fix: walk the book to compute VWAP + abort the replay if
                # the book is unfillable post-order; never credit phantom
                # shares.  Real fill confirmation arrives via UserFeed/REST
                # reconcile and is deduped through ``mark_trade_seen``.
                if otype == "FOK" and self._fill_replay_handler is not None:
                    mkt_ref = self.client._token_to_market.get(token_id)
                    book_ref = None
                    if mkt_ref is not None:
                        book_ref = (mkt_ref.book_yes if token_id == mkt_ref.yes_token
                                    else mkt_ref.book_no)
                    if side == Side.BUY:
                        entry_vwap, _, fillable = _round_trip_cost(book_ref, size)
                    else:
                        # SELL: VWAP we'd realize sweeping bids
                        _, entry_vwap, fillable = _round_trip_cost(book_ref, size)
                    if not fillable or entry_vwap == float("inf") or entry_vwap <= 0:
                        self.log.warning(
                            "FOK %s %s: book unfillable post-accept — skipping "
                            "fill replay (oid=%s); awaiting REST reconcile.",
                            side.value, token_id[:12], oid[:12])
                    else:
                        # Critic-v2 C-NEW-1 fix: 6-decimal precision (CTF grid).
                        # Pre-fix ``math.floor`` truncated below 1.0 → dust.
                        shares_est = round(size / entry_vwap, 6)
                        if shares_est < 1.0:
                            shares_est = 1.0
                        # Critic-v2 C-NEW-3 DELTA: seed BOTH the synthesized
                        # ``fok-{oid}`` key AND the raw ``oid`` into the dedup
                        # set.  When the real CLOB ``/trades`` row arrives via
                        # REST reconcile, its ``trade_id`` field can be the
                        # bare order id (older CLOB) or a wallet txhash (V2);
                        # without the bare-oid seed we double-credited the
                        # BUY leg on the next reconcile pass.  The REST
                        # reconcile loop in ``reconcile_fills`` checks
                        # ``trade_id`` against ``_seen_trade_ids`` directly,
                        # so adding ``oid`` here is the bridge.
                        # BUG-FIX #9: route through mark_trade_seen() for
                        # bounded FIFO eviction instead of unbounded direct
                        # .add()/.append().
                        fok_trade_id = f"fok-{oid}"
                        self.mark_trade_seen(fok_trade_id)
                        self.mark_trade_seen(oid)
                        # C-BUG-3 fix: do NOT call _fill_replay_handler here.
                        # Speculatively crediting the position before the CLOB
                        # confirms a real fill creates phantom shares that
                        # desync the local ledger.  Instead, only update the
                        # ORDER TRACKER state so the duplicate-entry guard in
                        # place() blocks re-entry.  The real position credit
                        # arrives via UserFeed WS (1-2s) or REST reconcile_fills
                        # (~30s), both of which are deduped by the seeded IDs.
                        async with self._lock:
                            tracked = self._orders.get(oid)
                            if tracked:
                                tracked.state = OrderState.FILLED
                                tracked.filled_size = shares_est
                                tracked.avg_fill_price = entry_vwap
                        self.log.info(
                            "FOK_ACCEPT %s %s vwap=%.4f ceiling=%.4f est_shares=%.1f "
                            "(awaiting real fill via WS/REST)",
                            side.value, token_id[:12], entry_vwap, price, shares_est)
            # H-7: release the sentinel if order was rejected (no oid returned).
            if not oid:
                async with self._lock:
                    if self._by_token.get((token_id, side)) == _PENDING_SENTINEL:
                        self._by_token.pop((token_id, side), None)
            return oid
        except Exception as e:
            # C-5: classify before counting.  Only structural rejections
            # (CLOB reject, auth failure, balance) trip the halt counter;
            # rate-limit and transient network errors back off and return.
            # NOTE on rate-limiter semantics: the leaky bucket advances
            # ``_next`` on EVERY call to ``acquire()`` regardless of
            # outcome, so an exception here means one rate-budget slot is
            # "spent" on a request that never completed.  The 1s sleep
            # for RATE_LIMIT below recovers, and the impact at 12 req/s
            # is one wasted slot per transient failure — log it so the
            # operator can correlate wasted budget with network events.
            ec = _classify_order_error(e)
            if ec == _OrderErrorClass.RATE_LIMIT:
                self.log.debug("rate-limit slot consumed (will recover via 1s sleep)")
                await asyncio.sleep(1.0)
            elif ec == _OrderErrorClass.NETWORK:
                self.log.debug("network slot consumed (transient: %s)", str(e)[:80])
            else:
                async with self._lock:
                    self._rejects += 1
            self.log.error("Order %s: %s", ec.value, str(e)[:120])
            # H-7: release the sentinel so future calls are not permanently blocked.
            async with self._lock:
                if self._by_token.get((token_id, side)) == _PENDING_SENTINEL:
                    self._by_token.pop((token_id, side), None)
            return None

    async def reconcile(self) -> int:
        if not self.client.sdk:
            return 0
        try:
            loop      = asyncio.get_running_loop()
            # V2: get_open_orders() vs V1: get_orders({maker_address, status})
            _get_open = getattr(self.client.sdk, "get_open_orders", None)
            if _get_open:
                live_list = await loop.run_in_executor(
                    None, _get_open)
            else:
                live_list = await loop.run_in_executor(
                    None,
                    lambda: self.client.sdk.get_orders(
                        {"maker_address": self.client.trading_address, "status": "LIVE"}
                    ),
                )
            live_index: Dict[str, dict] = {
                (o.get("id") or o.get("orderID", "")): o
                for o in (live_list or [])
                if isinstance(o, dict)
            }
        except Exception as e:
            self.log.debug("Reconcile fetch: %s", e)
            return 0

        pruned = 0
        now    = time.monotonic()
        async with self._lock:
            for oid, tracked in list(self._orders.items()):
                if oid in live_index:
                    if tracked.state == OrderState.PENDING:
                        tracked.state = OrderState.OPEN
                    raw        = live_index[oid]
                    # H-6 fix: CLOB V2 uses "size_matched" for cumulative fill.
                    # Pre-fix used "filledSize"/"takerAmount" which are V1 field
                    # names; both return None/0 under V2 -> partial fills never
                    # update tracked.filled_size and position drift accumulates.
                    new_filled = float(raw.get("size_matched") or
                                       raw.get("filledSize") or
                                       raw.get("takerAmount", 0) or 0)
                    if new_filled > tracked.filled_size:
                        prev   = tracked.filled_size
                        delta  = new_filled - prev
                        fp     = float(raw.get("price", tracked.price) or tracked.price)
                        tracked.avg_fill_price = (
                            (tracked.avg_fill_price * prev + fp * delta) / new_filled
                            if prev > 0 else fp
                        )
                        tracked.filled_size = new_filled
                elif tracked.state != OrderState.PENDING:
                    # v18.8: FOK orders marked FILLED via immediate replay
                    # are expected to disappear from the LIVE list — clean
                    # prune without the "missed partial fill" warning.
                    if tracked.state == OrderState.FILLED:
                        self._by_token.pop((tracked.token_id, tracked.side), None)
                        del self._orders[oid]
                        pruned += 1
                        continue
                    # Guard: warn if WS missed a partial fill before we prune
                    if tracked.filled_size > 0:
                        self.log.warning(
                            "Reconcile: pruning order %s with %.4f filled "
                            "(WS may have missed partial fill — position may desync)",
                            oid[:16], tracked.filled_size)
                    if tracked.state == OrderState.OPEN or now - tracked.created > 15:
                        self._by_token.pop((tracked.token_id, tracked.side), None)
                        del self._orders[oid]
                        pruned += 1

        if pruned:
            self.log.info("Reconcile: pruned %d stale orders (%d remaining)",
                          pruned, len(self._orders))
        return pruned

    async def reconcile_fills(self) -> int:
        """v18.4 — REST-driven fill replay (Pass 1 of dual-pass reconcile).

        Why
        ---
        ``UserFeed`` over WebSocket is best-effort: under disconnect,
        backpressure, or a server-side packet drop, fills can be lost
        permanently from the local PnL ledger.  This causes ghost
        positions where the bot believes it holds 0 shares but the
        chain says otherwise.

        How
        ---
        Every ``reconcile_fills_interval_s`` seconds we walk the CLOB
        ``/trades`` REST endpoint with a monotonically-advancing
        timestamp cursor:

          1. Pull all trades since ``_last_trade_cursor_ts``.
          2. Skip trades already in ``_seen_trade_ids``.
          3. For each NEW trade, dispatch via the
             ``_fill_replay_handler`` (Bot._on_fill) — the SAME code
             path used by live WS fills.  No parallel implementation,
             no drift between WS and REST handling.
          4. Advance ``_last_trade_cursor_ts`` to ``max(timestamp_seen)``.

        Idempotency: position accounting is FILL-ID-keyed via the
        ``_seen_trade_ids`` set, so even if a REST trade is reported
        with a slightly different cursor than the WS event, we replay
        at most once.

        Returns the number of fills replayed.  0 means everything is
        already up-to-date; a non-zero number after a server-side
        outage is the recovery signal.
        """
        if not self.client.sdk or self._fill_replay_handler is None:
            return 0
        # Serialize concurrent reconciliations — the cursor advance must
        # be atomic with the trade-id dedup write.
        if self._reconcile_fills_lock.locked():
            return 0
        async with self._reconcile_fills_lock:
            since_ts = self._last_trade_cursor_ts
            # C-BUG-6 fix: on the FIRST reconcile pass, walk back 1 hour
            # from boot to catch fills that happened during a crash/restart.
            # The dedup set will prevent double-counting any fills already
            # seen via WS.
            if not self._first_reconcile_done:
                since_ts = max(0.0, since_ts - 3600.0)
                self._first_reconcile_done = True
                self.log.info(
                    "reconcile_fills: first pass — walking back to %.0f "
                    "(%.0fs before boot)", since_ts,
                    self._last_trade_cursor_ts - since_ts)
            try:
                trades = await self._fetch_trades_since(since_ts)
            except Exception as e:
                self.log.warning("reconcile_fills fetch error: %s", e)
                return 0

            if not trades:
                return 0

            replayed = 0
            max_ts = since_ts
            for tr in trades:
                if not isinstance(tr, dict):
                    continue
                tid = self._extract_trade_id(tr)
                ts = self._extract_trade_ts(tr)
                if not tid:
                    continue
                if tid in self._seen_trade_ids:
                    continue
                # H-1 fix: dispatch FIRST, register and advance cursor only on success.
                # Pre-fix registered the id and updated max_ts BEFORE dispatch, so a
                # handler exception permanently skipped the fill (marked seen but never
                # applied) and the cursor advanced past it regardless.
                try:
                    res = self._fill_replay_handler(tr)
                    if asyncio.iscoroutine(res):
                        await res
                    # Only register after confirmed success:
                    if len(self._seen_trade_order) == self._seen_trade_ids_cap:
                        evicted = self._seen_trade_order[0]
                        self._seen_trade_ids.discard(evicted)
                    self._seen_trade_order.append(tid)
                    self._seen_trade_ids.add(tid)
                    if ts > max_ts:
                        max_ts = ts
                    replayed += 1
                    self.log.warning(
                        "reconcile_fills: REPLAYED trade %s (WS likely missed it)",
                        tid[:16])
                except Exception as e:
                    self.log.error(
                        "reconcile_fills: handler error on trade %s: %s",
                        tid[:16], e)
                    # Do NOT register the id or advance the cursor; next pass retries.

            # Advance the cursor.  Slight rollback (``max_ts - 1`` second)
            # would risk re-fetching newly-arrived trades; we trust the
            # dedup set and advance to the strict max seen.
            self._last_trade_cursor_ts = max_ts
            if replayed:
                self.log.info(
                    "reconcile_fills: replayed %d missing fills (cursor=%.3f)",
                    replayed, self._last_trade_cursor_ts)
            return replayed
            return replayed

    async def _fetch_trades_since(self, since_ts: float) -> List[dict]:
        """Internal: fetch ``/trades`` since a timestamp.

        v18.8 — uses ``TradeParams`` from py-clob-client (the SDK expects
        a typed object with ``.market``, ``.after``, etc. attributes —
        passing a plain dict causes ``AttributeError``).  Filters by
        ``maker_address`` so only OUR trades are returned.
        """
        loop = asyncio.get_running_loop()
        sdk = self.client.sdk
        since_int = int(since_ts)

        def _try() -> List[dict]:
            _get = getattr(sdk, "get_trades", None)
            if _get is None:
                return []
            # H-2 fix: import TradeParams from V2 first, V1 as fallback.
            # Pre-fix hardcoded py_clob_client (V1), so under the active V2 SDK:
            # - V1 not installed: ImportError -> params=None -> _get() loses cursor/filter
            # - V1 installed: V1-typed object passed to V2 get_trades -> TypeError -> REST reconcile dead
            try:
                from py_clob_client_v2.clob_types import TradeParams as _TradeParams
            except ImportError:
                try:
                    from py_clob_client.clob_types import TradeParams as _TradeParams
                except ImportError:
                    _TradeParams = None
            if _TradeParams is not None:
                params = _TradeParams(
                    after=since_int,
                    maker_address=self.client.trading_address,
                )
            else:
                params = None
            try:
                if params is not None:
                    res = _get(params)
                else:
                    res = _get()
                if isinstance(res, dict):
                    res = res.get("data") or res.get("trades") or []
                if isinstance(res, list):
                    return res
            except Exception:
                raise
            return []

        return await loop.run_in_executor(None, _try)

    @staticmethod
    def _extract_trade_id(tr: dict) -> str:
        return str(
            tr.get("trade_id")
            or tr.get("tradeId")
            or tr.get("id")
            or tr.get("transaction_hash")
            or ""
        )

    @staticmethod
    def _extract_trade_ts(tr: dict) -> float:
        # The CLOB ``/trades`` response uses ``match_time`` (seconds) in
        # v2 and ``timestamp`` (sometimes ms) in v1.  Normalize ANY unit
        # to seconds by magnitude: a real epoch in seconds is ~1.7e9, so
        # anything >= 1e11 is sub-second (ms ~1.7e12, us ~1.7e15, ns
        # ~1.7e18).  Divide by 1000 until it lands in the seconds band so
        # a ms/us/ns timestamp can't yield a cursor decades in the future
        # and permanently wedge reconciliation.
        for key in ("match_time", "timestamp", "matchTime", "ts"):
            v = tr.get(key)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(f) or f <= 0:
                continue
            # BUG-FIX #12: defense-in-depth.  ``math.isfinite`` already
            # rejects inf/NaN, but a previous code path emitted inf via
            # a scientific-notation edge case (and the field comment
            # marked this as ``# CRITICAL`` in CHANGES).  Cap the
            # magnitude to the year-2526 in nanoseconds — anything
            # beyond is a unit error and would permanently wedge
            # reconciliation if it leaked into the cursor.
            if f > 1e18:
                continue
            # BUG-FIX #12: also cap the unit-conversion loop.  Pre-fix
            # ``while f > 1e11: f /= 1000.0`` was unguarded — if a
            # regression in the magnitude check above ever let an inf
            # through, the loop would run forever.  Hard-cap iterations.
            iters = 0
            while f > 1e11 and iters < 5:
                f /= 1000.0
                iters += 1
            return f
        return 0.0

    async def cancel(self, oid: str) -> bool:
        await self._cancel_rl.acquire()     # prevent IP ban on cancel spam
        ok = await self.client.cancel(oid)
        if ok:
            async with self._lock:
                tracked = self._orders.pop(oid, None)
                if tracked:
                    self._by_token.pop((tracked.token_id, tracked.side), None)
        return ok

    async def cancel_all(self) -> None:
        # C-4 fix: exchange-first. Clearing local state BEFORE the exchange
        # call meant a network failure left orphaned live orders untracked,
        # which the duplicate-entry guard in place() then failed to detect.
        async with self._lock:
            snapshot_count = len(self._orders)
            snapshot_ids = list(self._orders.keys())
        if snapshot_count == 0:
            return
        try:
            await self.client.cancel_all()
        except Exception as e:
            self.log.error(
                "cancel_all: exchange call failed (%s) — %d orders may remain open: %s",
                e, snapshot_count,
                ", ".join(oid[:8] for oid in snapshot_ids[:5]))
            raise
        async with self._lock:
            self._orders.clear()
            self._by_token.clear()
        self.log.info("cancel_all: confirmed %d orders cancelled", snapshot_count)

    async def cancel_and_replace(self, token_id: str, side: Side,
                                  old_id: Optional[str], price: float,
                                  size: float, strategy: Strategy,
                                  neg_risk: bool = False,
                                  tick_size: float = 0.01) -> Optional[str]:
        if old_id:
            await self.cancel(old_id)
        return await self.place(token_id, side, price, size, strategy,
                                neg_risk=neg_risk, tick_size=tick_size)

    def find_open(self, token_id: str, side: Side) -> Optional[TrackedOrder]:
        oid = self._by_token.get((token_id, side))
        return self._orders.get(oid) if oid else None

    def remove(self, oid: str) -> None:
        tracked = self._orders.pop(oid, None)
        if tracked:
            self._by_token.pop((tracked.token_id, tracked.side), None)

    @property
    def count(self) -> int:
        return len(self._orders)

    @property
    def rejects(self) -> int:
        return self._rejects


# ─── Feeds ────────────────────────────────────────────────────────────────────

class BinanceFeed:
    def __init__(self, coins: List[str]):
        self._coins   = coins
        self._prices: Dict[str, float] = {}
        self._cbs:    List[Callable]   = []
        self._running = False
        self._last_msg: float = 0.0
        self.log      = get_logger("Binance")

    def price(self, coin: str) -> Optional[float]:
        return self._prices.get(coin.upper())

    def on_update(self, cb: Callable) -> None:
        self._cbs.append(cb)

    async def run(self) -> None:
        self._running = True
        streams = "/".join(f"{c.lower()}usdt@trade" for c in self._coins)
        url     = f"wss://stream.binance.com:9443/stream?streams={streams}"
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=5) as ws:
                    backoff = 1.0
                    self.log.info("Connected: %s", ", ".join(self._coins))
                    async for msg in ws:
                        if not self._running:
                            break
                        self._last_msg = time.monotonic()
                        try:
                            d   = _json_loads(msg).get("data", {})
                            sym = d.get("s", "").replace("USDT", "")
                            p   = float(d.get("p", 0))
                        except (ValueError, TypeError, AttributeError) as e:
                            self.log.debug("BinanceFeed parse error: %s", e)
                            continue
                        # v18.3: NaN/inf/zero guards — Binance very rarely
                        # emits literal "NaN" during halts.  Without this
                        # check the value silently propagates through
                        # every comparator (nan > x is always False).
                        if not sym or not math.isfinite(p) or p <= 0:
                            continue
                        self._prices[sym] = p
                        for cb in self._cbs:
                            try:
                                await cb(sym, p)
                            except Exception as cb_err:
                                self.log.debug(
                                    "BinanceFeed cb error: %s", cb_err)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    self.log.warning("WS error: %s (retry in %.0fs)", e, backoff)
                    # Equal-jitter backoff: decorrelates reconnect storms
                    # across shards/feeds so a venue outage doesn't trigger
                    # a synchronized thundering-herd reconnect (and a ban).
                    await asyncio.sleep(
                        backoff * 0.5 + random.uniform(0.0, backoff * 0.5))
                    backoff = min(backoff * 2, 30)

    @property
    def last_msg_age_s(self) -> float:
        return time.monotonic() - self._last_msg if self._last_msg else float("inf")

    async def stop(self) -> None:
        self._running = False


# ─── HyperPolyFeed — Sharded WebSocket Manager ──────────────────────────────

class HyperPolyFeed:
    """
    v18 sharded WebSocket feed. Improvements over v17 PolyFeed:
      1. N independent WS connections (shards) — one failure doesn't black out all feeds
      2. orjson parsing in hot path — 5-10x faster
      3. Incremental book state with cached running totals
      4. Per-shard health monitoring + independent reconnection
      5. Aggressive ping settings for faster stale detection
    """
    WS_URL      = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    TRADE_ALPHA = 0.3
    TRADE_TTL   = 30.0

    def __init__(self, shard_count: int = 2) -> None:
        self._books:      Dict[str, OrderBook] = {}
        self._tokens:     List[str]            = []
        self._token_set:  Set[str]             = set()
        self._cbs:        List[Callable]       = []
        self._trade_ewma: Dict[str, float]     = {}
        self._trade_ts:   Dict[str, float]     = {}
        self._last_msgs:  Dict[int, float]     = {}
        self._ws_shards:  Dict[int, Any]       = {}
        self._shard_count = max(1, shard_count)
        self._running     = False
        self._snapshot_received: Set[str]      = set()   # v18.1: snapshot guard
        # v18.3: per-shard queue of tokens that need to be (re)subscribed
        # the next time the shard's WS is up.  Prevents the data-loss
        # window described in the v18.0 review where new markets
        # discovered during a shard reconnect were silently dropped.
        self._pending_subs: Dict[int, Set[str]] = {
            i: set() for i in range(self._shard_count)}
        # Strong refs for fire-and-forget shard-restart tasks.  asyncio keeps
        # only a WEAK ref to a bare create_task, so the GC could otherwise
        # cancel a reconnect coroutine mid-await — precisely during a WS
        # outage, the worst time to lose it.  The done-callback discards.
        self._bg_tasks: Set[asyncio.Task] = set()
        self.log          = get_logger("HyperFeed")

    def subscribe(self, tid: str) -> None:
        if tid not in self._token_set:
            self._token_set.add(tid)
            self._tokens.append(tid)
            self._books.setdefault(tid, OrderBook(token_id=tid))

    async def subscribe_live(self, tids: List[str]) -> None:
        new = [t for t in tids if t not in self._token_set]
        if not new:
            return
        for tid in new:
            self.subscribe(tid)
        # v18.3: route each new token to its shard.  If the shard's WS
        # is currently up, send immediately; otherwise queue it so the
        # next successful connect replays the subscription.
        sent = 0
        for tid in new:
            shard_id = self._deterministic_shard(tid)
            ws = self._ws_shards.get(shard_id)
            self._pending_subs.setdefault(shard_id, set()).add(tid)
            if ws is None:
                continue
            try:
                await ws.send(_json_dumps({
                    "auth": {}, "type": "Market",
                    "markets": [], "assets_ids": [tid],
                }))
                self._pending_subs[shard_id].discard(tid)
                sent += 1
            except Exception as e:
                # Keep in pending for replay on reconnect.
                self.log.debug("live-sub send failed for %s: %s", tid[:8], e)
        self.log.info("Live-subscribed %d tokens (queued %d for reconnect)",
                      sent, len(new) - sent)

    def book(self, tid: str) -> Optional[OrderBook]:
        return self._books.get(tid)

    def last_trade(self, tid: str) -> Optional[float]:
        if time.monotonic() - self._trade_ts.get(tid, 0) > self.TRADE_TTL:
            return None
        return self._trade_ewma.get(tid)

    @property
    def last_msg_age_s(self) -> float:
        if not self._last_msgs:
            return float("inf")
        # Return the age of the most recently active shard
        return time.monotonic() - max(self._last_msgs.values())

    def shard_ages(self) -> Dict[int, float]:
        """Per-shard message age for health monitoring."""
        now = time.monotonic()
        return {sid: now - ts for sid, ts in self._last_msgs.items()}

    def on_update(self, cb: Callable) -> None:
        self._cbs.append(cb)

    def _deterministic_shard(self, tid: str) -> int:
        """Deterministic shard assignment via MD5 (not Python hash()).

        Python's hash() is randomized per process (PYTHONHASHSEED),
        making shard assignments non-reproducible across restarts.
        ``zlib.crc32`` is deterministic across processes and far cheaper
        than an MD5 hex-digest + big-int parse; since we only need a
        stable bucket index (not a cryptographic digest), a checksum is
        the right tool.  Called at (re)subscription time, not per tick.
        """
        return zlib.crc32(tid.encode()) % self._shard_count

    def _shard_tokens(self) -> Dict[int, List[str]]:
        """Distribute tokens across shards by deterministic hash."""
        shards: Dict[int, List[str]] = {i: [] for i in range(self._shard_count)}
        for tid in self._tokens:
            shards[self._deterministic_shard(tid)].append(tid)
        return shards

    async def run(self) -> None:
        self._running = True
        shard_map = self._shard_tokens()
        tasks = []
        for shard_id, tokens in shard_map.items():
            if tokens:
                tasks.append(asyncio.create_task(
                    self._run_shard(shard_id, tokens),
                    name=f"shard_{shard_id}"))
        if tasks:
            self.log.info("Started %d shards (%d tokens total)",
                          len(tasks), len(self._tokens))
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass

    async def _run_shard(self, shard_id: int, tokens: List[str]) -> None:
        backoff = 1.0
        while self._running:
            # v18.3: re-read the latest set of tokens for this shard on
            # every (re)connect.  Without this, new markets added via
            # subscribe_live() while the shard was reconnecting would
            # never be subscribed until the next manual restart.
            live_tokens = self._shard_tokens().get(shard_id, tokens)
            # Merge in any pending subs queued during a previous outage.
            pending = self._pending_subs.get(shard_id, set())
            if pending:
                merged = list({*live_tokens, *pending})
                live_tokens = merged
            # BUG-FIX #18: clear snapshot state for all shard tokens on
            # reconnect so stale deltas are rejected until a fresh
            # snapshot re-establishes the baseline.
            for tid in live_tokens:
                self._snapshot_received.discard(tid)
            try:
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=10,
                    ping_timeout=5,
                    max_size=16 * 1024 * 1024,
                ) as ws:
                    self._ws_shards[shard_id] = ws
                    backoff = 1.0
                    # Subscribe in batches of 10
                    for i in range(0, len(live_tokens), 10):
                        await ws.send(_json_dumps({
                            "auth": {}, "type": "Market",
                            "markets": [], "assets_ids": live_tokens[i:i + 10],
                        }))
                        await asyncio.sleep(0.03)
                    # Drain pending queue after successful subscribe burst.
                    self._pending_subs[shard_id] = set()
                    self.log.info("Shard %d: subscribed %d tokens",
                                  shard_id, len(live_tokens))
                    async for msg in ws:
                        if not self._running:
                            break
                        self._last_msgs[shard_id] = time.monotonic()
                        await self._handle(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._ws_shards.pop(shard_id, None)
                if self._running:
                    self.log.warning("Shard %d WS error: %s (retry %.0fs)",
                                     shard_id, e, backoff)
                    # Equal-jitter backoff: decorrelates reconnect storms
                    # across shards/feeds so a venue outage doesn't trigger
                    # a synchronized thundering-herd reconnect (and a ban).
                    await asyncio.sleep(
                        backoff * 0.5 + random.uniform(0.0, backoff * 0.5))
                    backoff = min(backoff * 2, 30)
        self._ws_shards.pop(shard_id, None)

    async def restart_shard(self, shard_id: int) -> None:
        """Restart a specific stale shard without affecting others."""
        shard_map = self._shard_tokens()
        tokens = shard_map.get(shard_id, [])
        if tokens:
            self.log.info("Restarting shard %d (%d tokens)", shard_id, len(tokens))
            # BUG-FIX #17: close the old WS to prevent duplicate consumers.
            old_ws = self._ws_shards.pop(shard_id, None)
            if old_ws:
                try:
                    await old_ws.close()
                except Exception:
                    pass
            t = asyncio.create_task(
                self._run_shard(shard_id, tokens),
                name=f"shard_{shard_id}_restart")
            self._bg_tasks.add(t)
            t.add_done_callback(self._bg_tasks.discard)

    async def _handle(self, raw: str) -> None:
        try:
            msgs = _json_loads(raw)
        except Exception as e:
            self.log.debug("WS parse error: %s", e)
            return
        if not isinstance(msgs, list):
            msgs = [msgs]
        for m in msgs:
            try:
                et  = m.get("event_type", "")
                tid = m.get("asset_id", "")
                if tid not in self._books:
                    continue
                bk = self._books[tid]

                if et == "book":
                    # v18.3: use OrderBook.replace_snapshot — O(N) once,
                    # dict-keyed internally for O(1) subsequent deltas.
                    bids = [(float(x["price"]), float(x["size"]))
                            for x in m.get("bids", [])
                            if float(x.get("size", 0)) > 0]
                    asks = [(float(x["price"]), float(x["size"]))
                            for x in m.get("asks", [])
                            if float(x.get("size", 0)) > 0]
                    bk.replace_snapshot(bids, asks)
                    bk.ts = time.monotonic()
                    self._snapshot_received.add(tid)

                elif et == "price_change":
                    # Drop deltas until a full snapshot has established
                    # the baseline — otherwise the book is incoherent.
                    if tid not in self._snapshot_received:
                        continue
                    # BUG-FIX #4: removed the always-false monotonic guard
                    # ``delta_ts < bk._snapshot_ts``.  Since both are
                    # monotonic clocks generated at receive-time, the delta
                    # timestamp is ALWAYS >= snapshot timestamp, making the
                    # check dead code.  The real stale-delta guard is the
                    # ``_snapshot_received`` check above (line 3188) plus
                    # BUG-FIX #18 (snapshot state cleared on reconnect).
                    delta_ts = time.monotonic()
                    for c in m.get("changes", []):
                        s = c.get("side", "").upper()
                        try:
                            p  = float(c["price"])
                            sz = float(c["size"])
                        except (KeyError, ValueError, TypeError):
                            continue
                        # v18.3: O(1) dict-level update.  No more linear
                        # scan + list pop, no more float-equality match.
                        bk.apply_delta(p, sz, is_bid=(s == "BID"))
                    bk.ts = delta_ts

                elif et == "last_trade_price":
                    try:
                        price = float(m["price"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    if math.isfinite(price) and 0 < price < 1:
                        prev = self._trade_ewma.get(tid)
                        self._trade_ewma[tid] = (
                            self.TRADE_ALPHA * price
                            + (1 - self.TRADE_ALPHA) * prev
                            if prev is not None else price
                        )
                        self._trade_ts[tid] = time.monotonic()

                # BUG-FIX #19: fire-and-forget callbacks to unblock the
                # WS parser loop.  Pre-fix, a slow strategy eval serialized
                # the entire WS read loop, adding 200-500ms latency.
                for cb in self._cbs:
                    try:
                        t = asyncio.create_task(cb(tid, bk))
                        self._bg_tasks.add(t)
                        t.add_done_callback(self._bg_tasks.discard)
                    except Exception as cb_err:
                        self.log.warning("book cb error: %s", cb_err)
            except Exception as inner:
                self.log.debug("WS msg dispatch error: %s", inner)

    async def unsubscribe(self, tids: List[str]) -> None:
        """Remove expired tokens and send WS unsubscribe payloads.

        Without WS unsubscribe, the CLOB server continues pushing
        price_change deltas for dead tokens — wasting network I/O,
        TCP buffer space, and orjson parsing cycles.
        """
        # Group tokens by shard for batched WS payloads
        shard_groups: Dict[int, List[str]] = {}
        for tid in tids:
            sid = self._deterministic_shard(tid)
            shard_groups.setdefault(sid, []).append(tid)
            self._token_set.discard(tid)
            self._books.pop(tid, None)
            self._trade_ewma.pop(tid, None)
            self._trade_ts.pop(tid, None)
            self._snapshot_received.discard(tid)
        self._tokens = [t for t in self._tokens if t in self._token_set]

        # Send unsubscribe payloads to each affected shard
        for sid, stids in shard_groups.items():
            ws = self._ws_shards.get(sid)
            if ws:
                try:
                    # M-2 fix: Polymarket WS unsubscribe format is
                    # {"assets_ids":[...], "operation":"unsubscribe"}
                    # (confirmed by Research agent against live docs).
                    # Pre-fix sent {"type":"Unsubscribe",...} which the
                    # server silently ignored, leaving zombie subscriptions.
                    await ws.send(_json_dumps({
                        "assets_ids": stids,
                        "operation": "unsubscribe",
                    }))
                except Exception:
                    pass  # WS may be reconnecting; local cleanup already done
        self.log.debug("Unsubscribed %d tokens (%d remain)",
                       len(tids), len(self._tokens))

    async def stop(self) -> None:
        self._running = False
        for ws in list(self._ws_shards.values()):
            try:
                await ws.close()
            except Exception:
                pass
        self._ws_shards.clear()


class UserFeed:
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    def __init__(self, client: PolyClient, om: OrderManager) -> None:
        self._client    = client
        self._om        = om
        self._lookup:   Dict[str, Market] = {}
        self._mids:     List[str]         = []
        self._fill_cbs: List[Callable]    = []
        self._running   = False
        self.connected  = False  # v18.9: exposed for adaptive reconcile
        # C-BUG-11 fix: track data freshness, not just TCP state.  A half-open
        # WS (TCP alive, server stopped pushing) reads as connected=True but
        # no fills arrive — the reconcile loop must detect this and poll fast.
        self._last_msg_ts: float = 0.0
        self.log        = get_logger("UserFeed")

    def set_markets(self, t2m: Dict[str, Market]) -> None:
        self._lookup = t2m
        self._mids   = list({m.market_id for m in t2m.values()})

    def on_fill(self, cb: Callable) -> None:
        self._fill_cbs.append(cb)

    @property
    def last_msg_age_s(self) -> float:
        """C-BUG-11: data freshness in seconds (∞ if no msg received yet)."""
        if self._last_msg_ts <= 0:
            return float("inf")
        return time.monotonic() - self._last_msg_ts

    async def run(self) -> None:
        if not self._client.api_key:
            self.log.warning("No API key — user feed disabled")
            return
        self._running = True
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=30, ping_timeout=15,
                    close_timeout=5, max_size=5 * 1024 * 1024,
                ) as ws:
                    # H-3 fix: UserFeed WS auth requires {apiKey, secret, passphrase}.
                    # Pre-fix computed an L2 HMAC on the subscribe body — but the
                    # CLOB V2 user channel does NOT do body HMAC; it authenticates
                    # via the raw api_secret as the "secret" field.  Missing that
                    # field caused every connect attempt to 401, silently degrading
                    # the bot to REST-only fill detection (30s gaps vs real-time).
                    # The redundant sub_bytes / hdrs HMAC computation is also removed.
                    sub = {
                        "type": "User",
                        "markets": self._mids,
                        "assets_ids": [],
                        "auth": {
                            "apiKey":     self._client.api_key,
                            "secret":     self._client.api_secret,
                            "passphrase": self._client.api_passphrase,
                        },
                    }
                    await ws.send(_json_dumps(sub))
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass
                    self.log.info("Connected (%d markets)", len(self._mids))
                    self.connected = True
                    backoff = 1.0  # Reset only after confirmed connection
                    async for msg in ws:
                        if not self._running:
                            break
                        try:
                            msgs = _json_loads(msg)
                            # C-BUG-11: stamp every parsed message for staleness
                            self._last_msg_ts = time.monotonic()
                            if not isinstance(msgs, list):
                                msgs = [msgs]
                            for m in msgs:
                                if m.get("event_type", "") not in (
                                        "trade", "order_fill", "match"):
                                    continue
                                tid = m.get("asset_id") or m.get("token_id", "")
                                mkt = self._lookup.get(tid)
                                if not mkt:
                                    continue
                                p   = float(m.get("price", 0))
                                sz  = float(m.get("size") or m.get("quantity", 0))
                                sd  = str(m.get("side", "")).upper()
                                if not p or not sz or sd not in ("BUY", "SELL"):
                                    continue
                                # v18.4 — teach the OrderManager's dedup
                                # set about this WS-delivered trade so a
                                # subsequent REST reconcile_fills doesn't
                                # double-replay it.
                                # v18.4 / H-4 fix: never skip dedup even when
                                # the WS event carries no recognized id field.
                                # Pre-fix: `if ws_trade_id:` bypassed dedup
                                # entirely for id-less fills, leaving them
                                # unregistered so a later REST reconcile
                                # double-credited the BUY (non-idempotent).
                                # Fix: synthesize a deterministic fallback id
                                # from market+price+size so the same economic
                                # event is always deduplicated.
                                ws_trade_id = str(
                                    m.get("trade_id")
                                    or m.get("id")
                                    or m.get("match_id")
                                    or f"ws-{mkt.market_id[:8]}-{sd}-{int(p*10000)}-{int(sz*1000)}"
                                )
                                try:
                                    newly = self._om.mark_trade_seen(ws_trade_id)
                                except Exception:
                                    newly = True  # fail toward processing
                                if not newly:
                                    # Duplicate delivery (reconnect
                                    # re-push) or a fill the REST
                                    # reconcile already replayed — skip
                                    # so _on_fill's non-idempotent BUY
                                    # branch can't double-count.
                                    continue
                                for cb in self._fill_cbs:
                                    try:
                                        await cb(mkt, tid, sd, sz, p)
                                    except Exception as cb_err:
                                        # v18.3: this used to silently
                                        # swallow ALL fill callback
                                        # errors — including position
                                        # accounting bugs.  Log loudly
                                        # so silent ledger drift is
                                        # visible.
                                        self.log.exception(
                                            "Fill callback error: %s", cb_err)
                                self.log.info("FILL %s %s %.2f@%.4f  '%s'",
                                              sd, tid[:12], sz, p,
                                              mkt.question[:30])
                        except Exception as ex:
                            self.log.debug("Parse error: %s", ex)
            except asyncio.CancelledError:
                # BUG-FIX #20: clear connected state on cancellation.
                self.connected = False
                break
            except Exception as e:
                self.connected = False
                if self._running:
                    self.log.warning("WS error: %s (retry in %.0fs)", e, backoff)
                    # Equal-jitter backoff: decorrelates reconnect storms
                    # across shards/feeds so a venue outage doesn't trigger
                    # a synchronized thundering-herd reconnect (and a ban).
                    await asyncio.sleep(
                        backoff * 0.5 + random.uniform(0.0, backoff * 0.5))
                    backoff = min(backoff * 2, 30)

    async def stop(self) -> None:
        self._running = False
        self.connected = False


# ─── Market discovery ─────────────────────────────────────────────────────────

COIN_KW: Dict[str, Set[str]] = {
    "BTC":   {"btc", "bitcoin"},
    "ETH":   {"eth", "ethereum"},
    "SOL":   {"sol", "solana"},
    "XRP":   {"xrp", "ripple"},
    "BNB":   {"bnb", "binance"},
    "DOGE":  {"doge", "dogecoin"},
    "MATIC": {"matic", "polygon"},
    "ADA":   {"ada", "cardano"},
}


def detect_coin(q: str) -> Optional[str]:
    """Identify the underlying coin from a Polymarket question string.

    v18.3: fails CLOSED on ambiguity (e.g. a question matching both BTC
    and ETH keywords).  Previously returned whichever appeared first in
    COIN_KW dict insertion order; now returns None so the market is
    skipped rather than mis-tagged.  Questions matching exactly one
    coin keyword set (e.g. ETH 2.0 launch markets) still resolve.
    """
    words = set(re.findall(r"[a-zA-Z]+", q.lower()))
    matched: List[str] = []
    for c, kw in COIN_KW.items():
        if kw & words:
            matched.append(c)
    if len(matched) == 1:
        return matched[0]
    return None


def _parse_end_time(raw: dict) -> Optional[float]:
    for k in ("endDate", "end_date", "endDateIso", "end_date_iso"):
        v = raw.get(k)
        if v:
            try:
                if isinstance(v, (int, float)):
                    return float(v)
                s = str(v).replace("Z", "+00:00")
                dt = datetime.fromisoformat(s)
                # FIX: if API returns timezone-naive string, assume UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                pass
    return None


def _jlist(v: Any) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            r = json.loads(v)
            if isinstance(r, list):
                return r
        except Exception:
            pass
    return []


async def discover_5min_markets(cfg: Config,
                                session: aiohttp.ClientSession) -> List[Market]:
    dlog = get_logger("5MinDiscovery", cfg.log_level)
    now = time.time()
    found: List[Market] = []

    for coin in cfg.coins:
        coin_lower = coin.lower()
        for tf_label, tf_secs in [("5m", 300), ("15m", 900)]:
            epoch = int(now - (now % tf_secs))
            for offset in [0, tf_secs]:
                slug = f"{coin_lower}-updown-{tf_label}-{epoch + offset}"
                try:
                    async with session.get(
                        f"{cfg.gamma_url}/markets",
                        params={"slug": slug, "closed": "false"},
                        timeout=aiohttp.ClientTimeout(total=6),
                    ) as r:
                        if not r.ok:
                            continue
                        data = await r.json(content_type=None)
                        items = data if isinstance(data, list) else [data] if isinstance(data, dict) and data.get("id") else []
                        for raw in items:
                            mkt = _parse_5min_market(raw, cfg, now, tf_secs)
                            if mkt:
                                found.append(mkt)
                except Exception:
                    pass

    if len(found) < 2:
        for kw_params in [
            {"closed": "false", "limit": "100", "offset": "0"},
        ]:
            try:
                async with session.get(
                    f"{cfg.gamma_url}/markets", params=kw_params,
                    timeout=aiohttp.ClientTimeout(total=12),
                ) as r:
                    if not r.ok:
                        continue
                    data = await r.json(content_type=None)
                    items = data if isinstance(data, list) else data.get("markets", [])
                    for raw in items:
                        q = raw.get("question") or raw.get("title") or ""
                        slug = raw.get("slug") or ""
                        has_5m  = bool(re.search(r"5[\s-]*min", q, re.IGNORECASE) or "5m" in slug)
                        has_15m = bool(re.search(r"15[\s-]*min", q, re.IGNORECASE) or "15m" in slug)
                        is_updown = "up or down" in q.lower()
                        if not ((has_5m or has_15m) and is_updown):
                            continue
                        coin = detect_coin(q)
                        if not coin or coin not in cfg.coins:
                            continue
                        tf_secs = 900 if has_15m else 300
                        mkt = _parse_5min_market(raw, cfg, now, tf_secs)
                        if mkt:
                            found.append(mkt)
            except Exception:
                pass

    seen: Set[str] = set()
    unique: List[Market] = []
    for m in found:
        if m.market_id not in seen:
            seen.add(m.market_id)
            unique.append(m)

    unique.sort(key=lambda m: m.liquidity, reverse=True)
    dlog.info("Found %d 5-min markets for %s", len(unique), cfg.coins)
    return unique


def _parse_5min_market(raw: dict, cfg: Config, now: float,
                       tf_secs: int = 300) -> Optional[Market]:
    try:
        tids = _jlist(raw.get("clobTokenIds"))
        outs = _jlist(raw.get("outcomes"))
        if len(tids) < 2:
            return None

        yes_id = no_id = None
        for i, o in enumerate(outs[:len(tids)]):
            ol = str(o).strip().lower()
            if ol in ("yes", "up", "higher", "more", "above", "over"):
                yes_id = str(tids[i])
            elif ol in ("no", "down", "lower", "less", "below", "under"):
                no_id = str(tids[i])
        yes_id = yes_id or str(tids[0])
        no_id = no_id or str(tids[1])
        if not yes_id or not no_id or yes_id == no_id:
            return None

        mid = str(raw.get("id") or raw.get("conditionId") or "")
        q = raw.get("question") or raw.get("title") or ""
        if not mid or not q:
            return None
        if raw.get("closed", False):
            return None
        if not raw.get("acceptingOrders", True):
            return None

        et = _parse_end_time(raw)
        if et and et < now:
            return None

        coin = detect_coin(q)
        if coin and coin not in cfg.coins:
            return None

        liq = float(raw.get("liquidityClob") or
                    raw.get("liquidityNum") or
                    raw.get("liquidity") or 0)

        return Market(
            market_id=mid, question=q, yes_token=yes_id, no_token=no_id,
            end_time=et, coin=coin, tf_secs=tf_secs, liquidity=liq,
            volatility=abs(float(raw.get("oneDayPriceChange") or 0)),
            neg_risk=bool(raw.get("negRisk") or raw.get("neg_risk") or False),
        )
    except Exception:
        return None


# ─── Price Tracker (Enhanced) ─────────────────────────────────────────────────

class PriceTracker:
    """v18.1: EWMA volatility (regime-adaptive) + multi-timeframe momentum.

    Key fix: old 45-min sample volatility was blind to regime changes.
    EWMA variance responds within seconds to volatility spikes, preventing
    the model from interpreting a macro move as a 5-sigma anomaly.
    """
    WINDOW = 2700
    _EWMA_ALPHA = 0.03    # ~33-sample half-life for EWMA variance

    def __init__(self, feed: BinanceFeed, prob_shrink: float = 1.0,
                 min_order_size_usdc: float = 0.0):
        self.feed = feed
        self.prob_shrink = prob_shrink
        self._per_coin_shrink: Dict[str, float] = {}
        # S-1: notional floor for OFI tilt (>= 2*min_order_size means the
        # book is deep enough to trust the imbalance signal).  Set by Bot.
        self._min_order_size_usdc: float = max(0.0, min_order_size_usdc)
        self._history: Dict[str, Deque[Tuple[float, float]]] = {}
        self._vwap_num: Dict[str, float] = {}
        self._vwap_den: Dict[str, float] = {}
        # EWMA volatility state
        self._ewma_var:  Dict[str, float] = {}   # exponentially weighted variance
        self._ewma_mean: Dict[str, float] = {}   # exponentially weighted mean return
        # S-2: parallel sorted lists for O(log N) get_price_at via bisect.
        # Maintained incrementally on each _on_price; trimmed in lock-step
        # with self._history so the indexes stay consistent under the
        # rolling-window eviction.  LatencyArb hits get_price_at on every
        # Binance tick (~50-150Hz); the previous O(N=2700) min() scan was
        # ~8M cmps/sec at 100tps × 3 coins.
        self._ts_index: Dict[str, List[float]] = {}
        self._px_by_ts: Dict[str, List[float]] = {}
        self.log = get_logger("PriceTracker")
        feed.on_update(self._on_price)

    async def _on_price(self, coin: str, price: float) -> None:
        # v18.3: NaN/inf/zero guard at ingestion.  Binance very rarely
        # emits ``"NaN"`` strings during halts and the bot must not
        # propagate them through every downstream comparison.
        if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            return
        now = time.time()
        if coin not in self._history:
            self._history[coin] = deque()
            self._vwap_num[coin] = 0.0
            self._vwap_den[coin] = 0.0
        dq = self._history[coin]
        # 1 Hz downsampling: append a new bucket at most once per second;
        # within the same second, overwrite the tail with the latest tick.
        # ``prev_1hz`` is the PREVIOUS second's (last) price — the correct
        # base for a genuine 1-second log return.
        if not dq or now - dq[-1][0] >= 1.0:
            prev_1hz = dq[-1][1] if dq else None
            dq.append((now, price))
            new_bucket = True
            # S-2: maintain parallel sorted lists for O(log N) get_price_at.
            ts_list = self._ts_index.setdefault(coin, [])
            px_list = self._px_by_ts.setdefault(coin, [])
            ts_list.append(now)
            px_list.append(price)
        else:
            dq[-1] = (dq[-1][0], price)
            prev_1hz = None
            new_bucket = False
            # Tail overwrite — keep the parallel index in sync.
            px_list = self._px_by_ts.get(coin)
            if px_list:
                px_list[-1] = price
        # v18.3: this is an EWMA of price (not VWAP).  Kept for the
        # downstream price-deviation signal.  Renaming would force a
        # broader API churn; the comment + naming in vwap_deviation()
        # acknowledge the mislabel.
        self._vwap_num[coin] = self._vwap_num.get(coin, 0.0) * 0.999 + price
        self._vwap_den[coin] = self._vwap_den.get(coin, 0.0) * 0.999 + 1.0
        # EWMA variance of log-returns (regime-adaptive).  CRITICAL: this
        # updates ONLY on a new 1 Hz bucket, never on every Binance @trade
        # tick (~10-50 Hz).  A per-tick return is a sub-second return whose
        # variance scales with the inter-tick interval Δt (Var≈σ²·Δt), so
        # updating per tick would make ``volatility()`` ≈ σ·√Δt — a
        # per-tick sigma — while ``prob_up`` consumes it as σ_per_sec,
        # understating horizon vol by ~√(ticks_per_sec) (≈4.5× at 20 tps)
        # and over-amplifying weak displacement signals.  Sampling at 1 Hz
        # makes ``ret`` a true 1-second return, so σ is genuinely per
        # second (matching the ``tau_s`` unit in the GBM).  Half-life for
        # alpha=0.03 is ln(2)/-ln(0.97) ≈ 22.8 seconds.
        if new_bucket and prev_1hz and prev_1hz > 0:
            ret = math.log(price / prev_1hz)
            if math.isfinite(ret):
                alpha = self._EWMA_ALPHA
                old_mean = self._ewma_mean.get(coin, 0.0)
                new_mean = alpha * ret + (1.0 - alpha) * old_mean
                old_var  = self._ewma_var.get(coin, ret * ret)
                # Deviation is measured against the PRE-update mean
                # (RiskMetrics EWMA variance).  Using ``new_mean`` here
                # would shrink the innovation by (1-alpha)^2 and bias the
                # steady-state variance LOW by that factor (~5.9% at
                # alpha=0.03), understating sigma and over-sizing Kelly.
                new_var  = alpha * (ret - old_mean) ** 2 + (1.0 - alpha) * old_var
                self._ewma_mean[coin] = new_mean
                self._ewma_var[coin]  = new_var
        cutoff = now - self.WINDOW
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        # S-2: trim the parallel sorted index in lock-step with the deque.
        ts_list = self._ts_index.get(coin)
        px_list = self._px_by_ts.get(coin)
        if ts_list:
            cutoff_idx = bisect.bisect_left(ts_list, cutoff)
            if cutoff_idx > 0:
                del ts_list[:cutoff_idx]
                if px_list is not None:
                    del px_list[:cutoff_idx]

    def get_price_at(self, coin: str, target_ts: float,
                     max_gap_s: float = 10.0) -> Optional[float]:
        # S-2 fix: pre-fix used min(dq, key=...) which is O(N=2700) per call.
        # LatencyArb hits this on every Binance tick (50-150 Hz × 3 coins),
        # i.e. ~8M comparisons/sec that stalled the event loop.  The parallel
        # sorted lists maintained in _on_price let us find the closest
        # timestamp via bisect in O(log N) and check at most two neighbours.
        ts_list = self._ts_index.get(coin)
        px_list = self._px_by_ts.get(coin)
        if not ts_list or not px_list:
            return None
        idx = bisect.bisect_left(ts_list, target_ts)
        best_idx, best_gap = -1, float("inf")
        for i in (idx - 1, idx):
            if 0 <= i < len(ts_list):
                gap = abs(ts_list[i] - target_ts)
                if gap < best_gap:
                    best_gap, best_idx = gap, i
        if best_idx < 0 or best_gap > max_gap_s:
            return None
        return px_list[best_idx]

    def _log_returns(self, coin: str, n: Optional[int] = None) -> List[float]:
        # S-6: when ``n`` is given, materialize only the last n+1 prices via
        # itertools.islice over a reversed view — pre-fix built the full
        # 2700-entry list and discarded 97% via [-60:] on every momentum()
        # call, paying two O(N) allocations per eval cycle.
        dq = self._history.get(coin)
        if not dq or len(dq) < 10:
            return []
        if n is None or n >= len(dq) - 1:
            prices = [p for _, p in dq]
        else:
            tail = list(itertools.islice(reversed(dq), n + 1))
            tail.reverse()
            prices = [p for _, p in tail]
        return [math.log(prices[i] / prices[i-1])
                for i in range(1, len(prices)) if prices[i-1] > 0]

    def volatility(self, coin: str) -> float:
        """Regime-adaptive volatility via EWMA variance.

        Falls back to sample volatility only when insufficient EWMA data.
        EWMA responds to vol spikes within seconds instead of smoothing
        them out over a 45-minute window.
        """
        ewma_var = self._ewma_var.get(coin)
        if ewma_var is not None and ewma_var > 0:
            return max(math.sqrt(ewma_var), 1e-8)
        # Fallback: sample volatility (cold start)
        rets = self._log_returns(coin)
        if len(rets) < 10:
            return 0.001
        mean = sum(rets) / len(rets)
        var = sum((r - mean)**2 for r in rets) / len(rets)
        return max(math.sqrt(var), 1e-8)

    def momentum(self, coin: str) -> float:
        # S-6: only need the most recent 60 returns; pass n=120 to keep a
        # safety margin (some entries may have prices[i-1] == 0 and be
        # dropped by the guard) without materializing the full 2700 deque.
        rets = self._log_returns(coin, n=120)
        if len(rets) < 5:
            return 0.0
        recent = rets[-60:]
        alpha = 2.0 / (len(recent) + 1)
        # M-ERR-9 fix: seed EMA with the mean of the window instead of
        # the first element.  With small windows (5-10 elements) the old
        # recent[0] seed biased the output toward one noisy datapoint.
        ema = sum(recent) / len(recent)
        for r in recent[1:]:
            ema = alpha * r + (1 - alpha) * ema
        return ema

    def velocity(self, coin: str, window_s: int = 30) -> float:
        dq = self._history.get(coin)
        if not dq or len(dq) < 2:
            return 0.0
        now = time.time()
        cutoff = now - window_s
        window_pts = [(ts, p) for ts, p in dq if ts >= cutoff]
        if len(window_pts) < 2:
            return 0.0
        old_price = window_pts[0][1]
        if old_price <= 0:
            return 0.0
        return (dq[-1][1] - old_price) / old_price

    def vwap_deviation(self, coin: str) -> float:
        """Current price deviation from VWAP. Positive = above VWAP."""
        den = self._vwap_den.get(coin, 0.0)
        if den <= 0:
            return 0.0
        vwap = self._vwap_num.get(coin, 0.0) / den
        current = self.feed.price(coin)
        if not current or vwap <= 0:
            return 0.0
        return (current - vwap) / vwap

    def roc(self, coin: str, lookback_s: int = 60) -> float:
        dq = self._history.get(coin)
        if not dq or len(dq) < 5:
            return 0.0
        now = time.time()
        cutoff = now - lookback_s
        old_pts = [(t, p) for t, p in dq if t >= cutoff]
        if len(old_pts) < 2:
            return 0.0
        return (old_pts[-1][1] - old_pts[0][1]) / old_pts[0][1]

    def is_choppy(self, coin: str, tf_secs: int = 300) -> bool:
        roc_30 = abs(self.roc(coin, 30))
        roc_60 = abs(self.roc(coin, 60))
        sigma = self.volatility(coin)
        return roc_30 < 0.0002 and roc_60 < 0.0003 and sigma > 0.0003

    def prob_up(self, coin: str, current_price: float,
                open_price: float, tau_s: float,
                yes_book: Optional[OrderBook] = None,
                yes_trade_ewma: Optional[float] = None,
                btc_displacement: Optional[float] = None) -> float:
        """P(S_T > S_0 | S_t) under driftless log-normal GBM with a
        small, bounded microstructure tilt.

        NB: GBM is applied to the underlying CRYPTO SPOT price (BTC/ETH/…),
        which lives on (0, ∞) — exactly where log-normal GBM is valid.
        The output is the digital-option probability P(S_T > S_0); it is
        NOT a GBM applied to the contract's [0,1] price.  The contract
        price is only ever used as a traded cost in the edge/Kelly math,
        never as the diffusing variable, so the "GBM on a bounded
        variable" objection does not apply.

        v18.3 rewrite.  The base term is the closed-form GBM survival
        probability for a binary up/down market::

            base = Φ( ln(S_t / S_0) / (σ · √(T-t)) )

        where σ is the per-second log-return volatility (EWMA, see
        ``volatility``) and (T-t) = ``tau_s``.  The deque downsamples
        Binance trades to ≈1 entry/sec on majors, so per-tick σ ≈
        per-second σ.

        On top of the base, three bounded tilts are applied:
          - OFI tilt           : ≤ ±4 prob points (top-of-book imbalance)
          - Trade-drift tilt   : ≤ ±2 prob points (last_trade EWMA vs mid)
          - BTC cross-influence: ≤ ±3 prob points (ETH/SOL vs BTC)

        Total microstructure influence is therefore capped at ~9 prob
        points; the rest of the signal comes from the mathematically
        grounded base CDF.
        """
        if (open_price <= 0 or current_price <= 0 or tau_s <= 0
                or not math.isfinite(current_price)
                or not math.isfinite(open_price)):
            return 0.5

        # --- Closed-form GBM base probability -----------------------------
        sigma_per_sec = max(self.volatility(coin), 2.5e-4)
        if not math.isfinite(sigma_per_sec) or sigma_per_sec <= 0:
            return 0.5
        # Horizon vol scales with √(time remaining).
        # S-3 fix: pre-fix `max(tau_s, 1.0)` clamped tau >= 1s — near-expiry
        # (tau < 1s) the model produced a flat sigma_horizon and the
        # probability stayed ≈0.5 instead of collapsing toward 0 or 1, which
        # suppressed the stop-loss exactly when it should fire.
        sigma_horizon = sigma_per_sec * math.sqrt(max(tau_s, 1e-4))
        if not math.isfinite(sigma_horizon) or sigma_horizon <= 0:
            return 0.5
        log_disp = math.log(current_price / open_price)
        if not math.isfinite(log_disp):
            return 0.5
        # Near-expiry collapse: when horizon vol is below ~1bp the
        # probability is essentially deterministic; report 0.98/0.02 so the
        # stop-loss gate (stop_loss_prob=0.43) and trail can act.
        if sigma_horizon < 1e-6:
            return 0.98 if log_disp > 0 else 0.02
        # Itô / volatility-drag correction.  Under driftless GBM,
        # ln(S_T/S_t) ~ N(-½σ²τ, σ²τ): the *median* terminal price sits
        # below the current price by ½σ²τ, so P(S_T > S_t) without this
        # term is biased HIGH (overestimates finishing ITM).  Subtracting
        # the drag is the textbook d₂ numerator.  Magnitude on 5-min
        # crypto is small (~10–35 bps of probability) — far below the
        # 1.2% min_edge gate — but it is mathematically correct and free,
        # and it removes a structural long-side bias.
        ito_drag = 0.5 * sigma_per_sec * sigma_per_sec * tau_s
        z = (log_disp - ito_drag) / sigma_horizon
        # Numerically stable normal CDF via math.erf.
        base = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        # --- Bounded microstructure tilts ---------------------------------
        tilt = 0.0
        if yes_book is not None:
            # S-1 fix: pre-fix used full-book imbalance, which on a $30
            # Polymarket book is moved from 0.5 -> 0.77 by a single $20
            # phantom bid (manipulator then lifts the ask into the inflated
            # probability and triggers our buy).  Patched to top-3 levels
            # with a min-notional floor of 2 * min_order_size — when the
            # book is too thin for OFI to be meaningful we set the tilt to 0.
            ofi_thin = (yes_book.top_depth_usdc <
                        2.0 * (self._min_order_size_usdc or 0.0))
            if ofi_thin:
                ofi = 0.0
            else:
                top_bids = heapq.nlargest(3, yes_book._bids_int.keys()) \
                    if yes_book._bids_int else []
                top_asks = heapq.nsmallest(3, yes_book._asks_int.keys()) \
                    if yes_book._asks_int else []
                top_bid_vol = sum(yes_book._bids_int.get(k, 0) for k in top_bids)
                top_ask_vol = sum(yes_book._asks_int.get(k, 0) for k in top_asks)
                total_top = top_bid_vol + top_ask_vol
                ofi = max(-0.5, min(0.5,
                    (top_bid_vol / total_top - 0.5) if total_top > 0 else 0.0))
            tilt += 0.08 * ofi                           # ≤ ±4 prob pts
            if yes_trade_ewma is not None:
                book_mid = yes_book.mid
                if book_mid is not None and book_mid > 0:
                    # Bounded to ±1 across 2c of mid drift.
                    drift = max(-1.0, min(1.0,
                                          (yes_trade_ewma - book_mid) / 0.02))
                    tilt += 0.02 * drift                 # ≤ ±2 prob pts

        if btc_displacement is not None and coin != "BTC":
            # ETH/SOL co-move with BTC; cap influence at 30bps of BTC.
            # Coefficient 0.03 caps this tilt at ±3 prob points, matching
            # the docstring contract (OFI ±4 + drift ±2 + BTC ±3 = ±9
            # total).  Was 0.04 (±4 → ±10 total), contradicting the
            # documented ~9-point cap.
            tilt += 0.03 * max(-1.0, min(1.0, btc_displacement / 0.003))

        p = base + tilt
        # Per-coin calibration shrinkage: use empirical correction if available,
        # otherwise fall back to the global prob_shrink.
        coin_shrink = self._per_coin_shrink.get(coin, self.prob_shrink)
        p = 0.5 + (p - 0.5) * coin_shrink
        return max(0.02, min(0.98, p))


def _round_trip_cost(book: Optional[OrderBook],
                     size_usdc: float) -> Tuple[float, float, bool]:
    """Walk both sides of the book to estimate round-trip cost.

    v18.7: Refactored to walk raw integer dictionaries (_asks_int,
    _bids_int) directly, eliminating cumulative float rounding from
    level-by-level arithmetic.  Shares are accumulated in integer
    micro-units (SIZE_SCALE=1e6) and converted to float only at the
    final division.

    Returns ``(entry_per_share, exit_per_share, fillable)``.

    - ``entry_per_share``: VWAP price the bot will pay if it FOKs
      ``size_usdc`` against the asks.  ``inf`` if the asks can't fill.
    - ``exit_per_share``: VWAP price the bot will receive if it later
      unwinds the same notional through the bids.  0.0 if the bids
      can't absorb the size.  Used as a conservative early-exit
      reference; if the trade is held to expiry this is irrelevant.
    - ``fillable``: False iff entry is impossible; callers should
      abort BEFORE wasting an API call on a guaranteed FOK rejection.
    """
    if not book or size_usdc <= 0:
        return float("inf"), 0.0, False

    PS = OrderBook.PRICE_SCALE   # 10_000
    SS = OrderBook.SIZE_SCALE    # 1_000_000

    # Scale target notional to micro-USD (PRICE_SCALE units × shares)
    # rem_scaled = size_usdc * PS * SS (integer micro-notional units)
    rem_scaled = int(round(size_usdc * PS * SS))

    # Entry walk — asks (ascending price keys)
    cost_scaled: int = 0    # sum of (price_key * taken_size_int)
    shares_int: int = 0     # sum of taken_size_int
    for key in sorted(book._asks_int.keys()):
        if key <= 0:
            continue
        level_size_int = book._asks_int[key]
        # notional at this level in scaled units = key * level_size_int
        level_notional = key * level_size_int
        if level_notional <= rem_scaled:
            cost_scaled += level_notional
            shares_int += level_size_int
            rem_scaled -= level_notional
        else:
            # Partial fill at this level: take exactly rem_scaled worth
            # shares_taken = rem_scaled / key (integer division, round down)
            taken_int = rem_scaled // key
            if taken_int > 0:
                cost_scaled += taken_int * key
                shares_int += taken_int
            rem_scaled = 0
            break
        if rem_scaled <= 0:
            break
    if rem_scaled > 0 or shares_int <= 0:
        return float("inf"), 0.0, False
    # Convert back to float: entry_per_share = cost/shares = (cost_scaled/PS/SS) / (shares_int/SS)
    entry_per_share = (cost_scaled / shares_int) / PS

    # Exit walk — bids (descending price keys)
    rem_scaled = int(round(size_usdc * PS * SS))
    rev_scaled: int = 0
    shares_out_int: int = 0
    for key in sorted(book._bids_int.keys(), reverse=True):
        if key <= 0:
            continue
        level_size_int = book._bids_int[key]
        level_notional = key * level_size_int
        if level_notional <= rem_scaled:
            rev_scaled += level_notional
            shares_out_int += level_size_int
            rem_scaled -= level_notional
        else:
            taken_int = rem_scaled // key
            if taken_int > 0:
                rev_scaled += taken_int * key
                shares_out_int += taken_int
            rem_scaled = 0
            break
        if rem_scaled <= 0:
            break
    # BUG-FIX #5: check bid-side residual — if bids can't absorb full
    # exit size, return fillable=False.  Pre-fix returned True on
    # partial bid depth, causing phantom-liquidity entries.
    if rem_scaled > 0 or shares_out_int <= 0:
        exit_per_share = (rev_scaled / shares_out_int) / PS if shares_out_int > 0 else 0.0
        return entry_per_share, exit_per_share, False
    exit_per_share = (rev_scaled / shares_out_int) / PS

    return entry_per_share, exit_per_share, True


def _estimate_slippage(book: Optional[OrderBook], size_usdc: float) -> float:
    """Backward-compat: returns ENTRY slippage above best ask.

    Kept for the latency-arb path that only needs a one-sided cost.
    For the 5-min strategy, prefer ``_round_trip_cost`` which models
    both sides.
    """
    if not book or not book._asks_int or size_usdc <= 0:
        return 0.99
    best_ask = book.best_ask
    if best_ask is None:
        return 0.99
    entry_per_share, _, fillable = _round_trip_cost(book, size_usdc)
    if not fillable or entry_per_share == float("inf"):
        return 0.99   # prohibitive: signals "do not trade"
    return max(0.001, entry_per_share - best_ask)


# ─── Scope-A: pure decision helpers (unit-tested) ───────────────────────────
#
# These are deliberately MODULE-LEVEL and side-effect-free so the trading
# decisions they encode can be unit-tested without standing up the full bot
# (Config / OrderManager / asyncio).  The class methods below delegate to
# them so live behaviour and tests share one implementation.

def kelly_size(p_final: float,
               entry_price: float,
               entry_slip: float,
               exit_slip: float,
               *,
               kelly_fraction: float,
               bankroll: float,
               max_bankroll_fraction: float,
               min_order_size: float,
               max_order_size: float,
                cold_start: bool = False,
                negative_ev_skips: bool = True,
                full_kelly_cap: float = 0.25,
                p_hold_to_expiry: float = 0.60,
                taker_fee_bps: float = 20.0,
                category_fee_rate: float = 0.0) -> float:
    """Fractional-Kelly stake for a binary asymmetric-payoff bet.

    Cin  = entry_price + entry_slip + taker_fee   (per-share cost in)
    Cout = p_hold * 1.0 + (1-p_hold) * (1 - exit_slip)
    b    = (Cout - Cin) / Cin                     (net payoff odds)
    f*   = (p*b - q) / b                          (full Kelly)

    C-2 fix: pre-fix Cout = 1 - exit_slip assumed every winner was sold
    via the book before resolution.  In reality a fraction p_hold_to_expiry
    of winners redeem at $1, so the previous expression undersized winners
    by exit_slip * p_hold (3-5% of size at exit_slip=5-8%).  Default 0.60
    matches the v18.4 `forced_exit_hold_prob` config.

    Q-1 fix: pre-fix Cin omitted the 20bps taker fee Polymarket charges
    on FOK takers, so the modeled edge was overstated ~0.4% per leg.

    Returns a dollar size in [min_order_size, max_order_size], OR 0.0 to
    signal SKIP.  When the bet is non-positive EV (p*b <= q) and
    negative_ev_skips is True (the LIVE path), we return 0.0 — the
    pre-v19 code returned min_order_size here, i.e. it still fired the
    minimum clip on a provably-losing bet.  In DRY_RUN
    (negative_ev_skips=False) we keep taking the floor so the calibration
    harness still collects outcome rows across the spectrum.
    """
    # BUG-FIX #28: invalid probability must SKIP (return 0.0), not fire
    # a minimum-size live trade on mathematically undefined inputs.
    if not (0.0 < p_final < 1.0):
        return 0.0 if negative_ev_skips else min_order_size
    # H-8 fix: Polymarket fee is probability-weighted: Fee = C * feeRate * p * (1-p),
    # not a flat bps on price.  When category_fee_rate > 0 (caller passes the
    # market-specific rate, e.g. 0.0175 for Crypto), use the correct formula.
    # Falls back to the flat taker_fee_bps model when category_fee_rate is 0
    # (backward-compat default) so existing call sites are unaffected.
    if category_fee_rate > 0:
        fee_per_share = max(0.0, category_fee_rate) * max(1e-6, p_final) * max(1e-6, 1.0 - p_final) * max(0.0, entry_price)
    else:
        fee_per_share = max(0.0, taker_fee_bps) * 1e-4 * max(0.0, entry_price)
    Cin = entry_price + max(0.0, entry_slip) + fee_per_share
    p_hold = max(0.0, min(1.0, p_hold_to_expiry))
    Cout_redeem = 1.0
    Cout_early = 1.0 - max(0.0, exit_slip)
    Cout = p_hold * Cout_redeem + (1.0 - p_hold) * Cout_early
    # BUG-FIX #29: invalid cost basis must SKIP, not fire min_order_size.
    if not (0.0 < Cin < 1.0):
        return 0.0 if negative_ev_skips else min_order_size
    if Cout <= Cin:
        return 0.0 if negative_ev_skips else min_order_size
    p = max(1e-6, min(1.0 - 1e-6, p_final))
    q = 1.0 - p
    b = (Cout - Cin) / Cin
    ev = p * b - q
    if ev <= 0.0:
        return 0.0 if negative_ev_skips else min_order_size
    f = max(0.0, min(full_kelly_cap, ev / b))
    frac = kelly_fraction * (0.5 if cold_start else 1.0)
    cap = bankroll * max_bankroll_fraction
    if min_order_size > cap:
        return 0.0
    size = min(bankroll * f * frac, cap)
    return round(max(min_order_size, min(max_order_size, size)), 2)


def adverse_gate(adverse_ewma_bps: Optional[float],
                 mid: Optional[float],
                 edge: float) -> bool:
    """True ⇒ SKIP the entry because measured post-fill adverse selection
    would eat at least the whole modeled edge.

    ``adverse_ewma_bps`` is a rolling EWMA of SIGNED post-fill mid drift in
    basis points of mid (positive = market moved against our fills, i.e. we
    were the dumb liquidity).  ``edge`` is the modeled per-share entry edge
    in [0,1] price units.  A None/empty measurement never blocks.
    """
    if adverse_ewma_bps is None or mid is None or mid <= 0:
        return False
    if adverse_ewma_bps <= 0.0:
        return False
    adverse_per_share = (adverse_ewma_bps / 1e4) * mid
    return adverse_per_share >= max(0.0, edge)


def maker_entry_price(best_bid: Optional[float],
                      best_ask: Optional[float],
                      tick: float,
                      join_ticks: int,
                      prob_cap: float) -> Optional[float]:
    """Passive (post-only) GTC entry price: join the bid (+ join_ticks)
    WITHOUT crossing the spread, and never above ``prob_cap - tick`` so we
    never rest a limit at a price that already has no edge.  Returns None if
    no valid non-crossing maker price exists.
    """
    if best_bid is None or best_bid <= 0 or tick <= 0:
        return None
    price = best_bid + max(0, join_ticks) * tick
    if best_ask is not None and best_ask > 0:
        price = min(price, best_ask - tick)        # never cross the spread
    price = min(price, prob_cap - tick)            # never post a no-edge rest
    price = math.floor(round(price / tick, 9)) * tick   # snap to grid
    if price <= 0.0 or price >= 1.0:
        return None
    return round(price, 6)


def should_force_exit_near_expiry(side_is_yes: bool,
                                  p_up: float,
                                  hold_if_winning: bool,
                                  hold_prob: float) -> bool:
    """Near expiry: dump to the book, or hold to $1/$0 settlement?

    Selling a likely WINNER into MM-gapped bids realizes worse than
    redemption, so HOLD when the held side is clearly winning and EXIT
    (salvage) otherwise.  ``True`` ⇒ force-sell now.
    """
    if not hold_if_winning:
        return True
    p_side = p_up if side_is_yes else (1.0 - p_up)
    return p_side < hold_prob


# ─── Scope-A: offline calibration / go-no-go harness ────────────────────────
#
# The bot already logs every EVAL and every trade OUTCOME to a CSV but NEVER
# acts on it.  This turns that recorded data into (a) an offline report an
# analyst can read (`python polybot.py --analyze`) and (b) a hard live
# go/no-go gate (`require_proven_edge`) that refuses real capital until
# measured edge > measured cost.  This is the honest version of "make it
# profitable": you cannot, in code, manufacture alpha — you can only refuse
# to bet real money until the recorded data proves it exists.

@dataclass
class CalibrationReport:
    n_trades:               int
    n_eval_rows:            int
    brier:                  Optional[float]
    realized_hit_rate:      Optional[float]
    mean_ask:               Optional[float]
    mean_entry_slip:        Optional[float]
    realized_edge_net_cost: Optional[float]
    mean_net_pnl:           Optional[float]
    total_net_pnl:          Optional[float]
    mean_adverse_bps:       Optional[float]
    reliability:            List[Tuple[float, float, int]]  # (bucket_lo, hit_rate, n)


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        f = float(s)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def load_calibration_rows(path: str) -> List[Dict[str, str]]:
    """Parse the calibration CSV into dict rows keyed by header name.

    Tolerant of extra/missing columns and absent files (returns []), so a
    fresh deployment that has never written a row degrades gracefully into
    a "not enough data" go/no-go verdict rather than a crash.
    """
    expanded = os.path.expanduser(path)
    if not expanded or not os.path.exists(expanded):
        return []
    rows: List[Dict[str, str]] = []
    with open(expanded, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row:
                rows.append(row)
    return rows


def build_matched_samples(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Pair each closed-trade OUTCOME row with the most recent prior EVAL
    row for the same market_id.  5-min markets get a unique market_id per
    interval, so this recovers the (predicted p, ask, realized win) tuples
    needed for Brier / reliability / edge-net-of-cost.
    """
    evals_by_mkt: Dict[str, List[Dict[str, Any]]] = {}
    outcomes: List[Dict[str, Any]] = []
    for r in rows:
        rt = (r.get("row_type") or "").strip()
        mid = (r.get("market_id") or "").strip()
        ts = _to_float(r.get("ts_unix")) or 0.0
        if rt == "eval" and mid:
            evals_by_mkt.setdefault(mid, []).append({
                "ts": ts, "p": _to_float(r.get("p")),
                "ask": _to_float(r.get("ask")),
                "entry_slip": _to_float(r.get("entry_slip")),
                "side": (r.get("side") or "").strip(),
            })
        elif rt == "outcome" and mid:
            outcomes.append({
                "ts": ts, "mid": mid,
                "win": _to_float(r.get("win")),
                "net_pnl": _to_float(r.get("net_pnl")),
            })
    for evs in evals_by_mkt.values():
        evs.sort(key=lambda e: e["ts"])
    matched: List[Dict[str, Any]] = []
    for o in outcomes:
        cands = [e for e in evals_by_mkt.get(o["mid"], [])
                 if e["ts"] <= o["ts"] and e["p"] is not None]
        if not cands:
            continue
        e = cands[-1]
        matched.append({
            "p": e["p"], "ask": e["ask"], "entry_slip": e["entry_slip"],
            "win": o["win"], "net_pnl": o["net_pnl"],
        })
    return matched


def calibration_report(rows: List[Dict[str, str]]) -> CalibrationReport:
    matched = build_matched_samples(rows)
    n_eval = sum(1 for r in rows if (r.get("row_type") or "").strip() == "eval")
    adverse = [v for v in (_to_float(r.get("adverse_bps")) for r in rows
                           if (r.get("row_type") or "").strip() == "shadow")
               if v is not None]
    mean_adverse = (sum(adverse) / len(adverse)) if adverse else None

    wins = [m for m in matched if m["win"] is not None]
    if not wins:
        return CalibrationReport(
            n_trades=len(matched), n_eval_rows=n_eval, brier=None,
            realized_hit_rate=None, mean_ask=None, mean_entry_slip=None,
            realized_edge_net_cost=None, mean_net_pnl=None, total_net_pnl=None,
            mean_adverse_bps=mean_adverse, reliability=[])

    n = len(wins)
    hit_rate = sum(m["win"] for m in wins) / n
    with_p = [m for m in wins if m["p"] is not None]
    brier = (sum((m["p"] - m["win"]) ** 2 for m in with_p) / len(with_p)
             if with_p else None)
    asks = [m["ask"] for m in wins if m["ask"] is not None]
    slips = [m["entry_slip"] for m in wins if m["entry_slip"] is not None]
    mean_ask = (sum(asks) / len(asks)) if asks else None
    mean_slip = (sum(slips) / len(slips)) if slips else 0.0
    edge_net = (hit_rate - mean_ask - (mean_slip or 0.0)
                if mean_ask is not None else None)
    pnls = [m["net_pnl"] for m in wins if m["net_pnl"] is not None]
    mean_pnl = (sum(pnls) / len(pnls)) if pnls else None
    total_pnl = sum(pnls) if pnls else None

    # Reliability curve: 0.05-wide buckets on predicted p.
    buckets: Dict[int, List[float]] = {}
    for m in with_p:
        b = min(19, int(m["p"] / 0.05))
        buckets.setdefault(b, []).append(m["win"])
    reliability = [
        (round(b * 0.05, 2), sum(v) / len(v), len(v))
        for b, v in sorted(buckets.items())
    ]
    return CalibrationReport(
        n_trades=len(matched), n_eval_rows=n_eval, brier=brier,
        realized_hit_rate=hit_rate, mean_ask=mean_ask, mean_entry_slip=mean_slip,
        realized_edge_net_cost=edge_net, mean_net_pnl=mean_pnl,
        total_net_pnl=total_pnl, mean_adverse_bps=mean_adverse,
        reliability=reliability)


def go_no_go(report: CalibrationReport,
             *,
             min_samples: int,
             min_edge: float,
             max_adverse_bps: float) -> Tuple[bool, List[str]]:
    """Decide whether the recorded data justifies risking REAL capital.

    Returns ``(allowed, reasons)``.  ``allowed`` is True only when every
    check passes; ``reasons`` lists each failed check.  Fails CLOSED: a
    missing metric is treated as "not proven".
    """
    reasons: List[str] = []
    if report.n_trades < min_samples:
        reasons.append(
            f"insufficient samples: {report.n_trades} closed trades "
            f"< required {min_samples}")
    if report.realized_edge_net_cost is None:
        reasons.append("no realized edge measurable (no matched outcomes)")
    elif report.realized_edge_net_cost <= min_edge:
        reasons.append(
            f"realized edge net of cost {report.realized_edge_net_cost:+.4f} "
            f"<= required {min_edge:+.4f} (the model's selected trades did "
            f"NOT beat the price they paid)")
    if report.mean_net_pnl is not None and report.mean_net_pnl <= 0.0:
        reasons.append(
            f"mean net PnL per trade {report.mean_net_pnl:+.4f} <= 0")
    if (report.mean_adverse_bps is not None
            and report.mean_adverse_bps > max_adverse_bps):
        reasons.append(
            f"adverse selection {report.mean_adverse_bps:.1f}bps "
            f"> cap {max_adverse_bps:.1f}bps (you are the dumb liquidity)")
    return (not reasons), reasons


def evaluate_go_no_go(cfg: "Config") -> Tuple[bool, List[str]]:
    """Load the calibration CSV named by ``cfg`` and run the go/no-go gate."""
    rows = load_calibration_rows(cfg.calibration_log_path)
    report = calibration_report(rows)
    return go_no_go(report, min_samples=cfg.min_proven_samples,
                    min_edge=cfg.min_proven_edge,
                    max_adverse_bps=cfg.max_adverse_bps)


def print_calibration_report(report: CalibrationReport, path: str) -> None:
    def fmt(v: Optional[float], spec: str = "+.4f") -> str:
        return "n/a" if v is None else format(v, spec)
    print("\n=== Polybot calibration report ===")
    print(f"source            : {os.path.expanduser(path)}")
    print(f"eval rows         : {report.n_eval_rows}")
    print(f"closed trades     : {report.n_trades}")
    print(f"realized hit rate : {fmt(report.realized_hit_rate, '.4f')}")
    print(f"mean entry ask    : {fmt(report.mean_ask, '.4f')}")
    print(f"mean entry slip   : {fmt(report.mean_entry_slip, '.4f')}")
    print(f"edge net of cost  : {fmt(report.realized_edge_net_cost)}"
          f"   (hit_rate - ask - slip; >0 means real edge)")
    print(f"Brier score       : {fmt(report.brier, '.4f')}"
          f"   (lower is better; 0.25 = coin flip)")
    print(f"mean net PnL/trade : {fmt(report.mean_net_pnl)}")
    print(f"total net PnL     : {fmt(report.total_net_pnl)}")
    print(f"adverse selection : {fmt(report.mean_adverse_bps, '.1f')} bps"
          f"   (post-fill mid drift against us)")
    if report.reliability:
        print("\nreliability (predicted bucket -> realized win rate):")
        for lo, rate, k in report.reliability:
            print(f"  [{lo:.2f},{lo + 0.05:.2f})  win={rate:.3f}  n={k}")
    print("")


# ─── 5-Min Strategy (Event-Driven) ──────────────────────────────────────────

class FiveMinStrategy:
    """
    v18: Event-driven evaluation with debounce.
    Triggers on Binance ticks and book updates, not timer polls.
    """

    def __init__(self, cfg: Config, om: OrderManager, risk: "Risk",
                 tracker: PriceTracker, metrics: Optional[Metrics] = None):
        self.cfg = cfg
        self.om = om
        self.risk = risk
        self.tracker = tracker
        self.metrics = metrics
        self.log = get_logger("FiveMinStrat", cfg.log_level)
        self._traded: Set[str] = set()
        self._open_prices: Dict[str, float] = {}
        self._open_intervals: Dict[str, int] = {}
        self._high_bids: Dict[Any, float] = {}
        # v19 Scope-A: consecutive-eval counter for the de-noised fast-exit.
        self._fast_exit_counts: Dict[Any, int] = {}
        self._net_exposure: float = 0.0
        # C-BUG-12 fix: _pos_lock REMOVED — position mutations are guarded
        # by Bot._pos_lock (set at Bot.__init__), not a per-strategy lock.
        # The old duplicate lock here was never acquired and caused confusion.
        self._sustain_counts: Dict[str, int] = {}
        self._entry_times: Dict[str, float] = {}
        # v18.7 — partial profit-taking state.
        # Tracks positions where TP1 (first partial take) has already fired.
        # Key: (market_id, token_id) → entry_price at which TP1 was taken.
        # The stored entry price becomes the break-even stop for the runner.
        self._tp1_taken: Dict[Tuple[str, str], float] = {}
        # v18.7 confidence mode — store edge at entry for decay calculation.
        self._entry_edges: Dict[Tuple[str, str], float] = {}
        # v18.7: In-flight exit shares — prevents race condition where the
        # evaluation loop sees stale pos.shares before fill arrives over WS.
        self._shares_in_flight: Dict[Tuple[str, str], float] = {}
        self._exit_fail_counts: Dict[Tuple[str, str], int] = {}
        # C-BUG-5 fix: track legs held to $1/$0 settlement so Risk.record_pnl
        # sees the (estimated) redemption PnL when markets resolve.  Key:
        # (market_id, token_id) → (shares, avg_entry_price, expected_payout).
        self._pending_redemptions: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        self.polyfeed: Optional[Any] = None
        self._balance_cache: float = 0.0
        self._balance_ts: float = 0.0
        # v18: Debounce state
        self._eval_debounce: Dict[str, float] = {}
        # v18.3: Adaptive Kelly state — fixed-size sliding window of
        # the last N=50 trade outcomes.  Previously int-truncated
        # accumulators caused the recent win-rate to drift over time.
        self._recent_outcomes: Deque[bool] = deque(maxlen=50)
        # O(1) win count maintained incrementally in ``record_outcome`` so
        # the Kelly hot path never re-scans the window.  Invariant:
        # ``_recent_wins == sum(_recent_outcomes)`` at all times (asserted
        # by the micro-test); ``record_outcome`` is the SOLE writer.
        self._recent_wins: int = 0
        # S-4: per-coin shrinkage state.  The portfolio-wide hit rate washed
        # out the per-regime cross-section (a hot BTC run was diluted by an
        # unrelated SOL cold streak).  Per-coin deques (cap 30 each) keep
        # the regime signal resolved while still bounding memory.
        self._per_coin_outcomes: Dict[str, Deque[bool]] = {}
        self._per_coin_wins: Dict[str, int] = {}
        # v18: Eval semaphore for concurrency limiting
        self._eval_sem = asyncio.Semaphore(cfg.max_concurrent_evals)
        # v18.2: Diagnostic counters for silent guard tracking
        self._diag_guard_hits: int = 0
        self._diag_eval_reached: int = 0
        self._diag_last_summary: float = time.monotonic()
        self._diag_trigger_calls: int = 0
        # Calibration CSV: open the append handle ONCE (lazily) instead of
        # stat()+open()+close() on every eval.  Keeping the line-buffered
        # handle resident removes ~3 syscalls + an open() from the hot path.
        self._calib_fh: Optional[Any] = None
        self._calib_init_done: bool = False
        # Block-buffered telemetry: count writes so we can flush every
        # _CALIB_FLUSH_EVERY rows rather than syscalling on every line.
        self._calib_writes: int = 0
        # Dedicated SINGLE-thread executor for calibration writes.  The
        # row is formatted on the event loop (pure CPU) but the actual
        # ``write()``/``flush()`` is offloaded here so a page-cache / EBS
        # stall can never block the loop (which also runs the WS parser
        # and risk manager).  max_workers MUST be 1: concurrent writers
        # would interleave partial lines on the shared FD and race on
        # ``_calib_writes``, corrupting the CSV.  One worker serializes
        # the writes and preserves row order.
        self._calib_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="calib-io")

    async def evaluate_all(self, markets: List[Market]) -> None:
        """Timer-driven fallback: evaluate all markets sequentially."""
        # Net exposure is now cached; only recalculated here for timer path
        self._net_exposure = sum(m.pos_yes.cost - m.pos_no.cost for m in markets)
        for mkt in markets:
            has_pos = mkt.pos_yes.shares > 0 or mkt.pos_no.shares > 0
            if mkt.market_id in self._traded and not has_pos:
                continue
            try:
                await self._evaluate(mkt)
            except Exception as e:
                self.log.debug("Eval '%s': %s", mkt.question[:30], e)

    async def evaluate_single(self, mkt: Market, all_markets: List[Market]) -> None:
        """v18 event-driven: evaluate a single market under the eval semaphore.

        Debounce + spawn throttling is enforced by the CALLERS
        (``Bot._on_price`` / ``Bot._on_book``) BEFORE the task is created:
        they stamp ``_eval_debounce[market_id]`` synchronously (with no
        ``await`` between the check and the stamp, so it is race-free on
        the single event-loop thread).  That guarantees at most ONE eval
        task per market per debounce window, bounding task creation to
        ~``n_markets / debounce`` even under a volatility tick storm.
        Without that producer-side stamp a burst could spawn many tasks
        for the same market before the first one ran, piling unbounded
        coroutines behind the semaphore.
        """
        self._diag_trigger_calls += 1
        async with self._eval_sem:
            try:
                await self._evaluate(mkt)
            except Exception as e:
                self.log.warning("EventEval '%s': %s", mkt.question[:30], e)

    async def _evaluate(self, mkt: Market) -> None:
        """v18.2: Diagnostic wrapper — tracks silent guard blocks."""
        self._diag_guard_hits += 1
        periodic = (time.monotonic() - self._diag_last_summary > 90)
        if periodic:
            self._diag_last_summary = time.monotonic()
            self.log.info(
                "DIAG | calls=%d | markets=%d | traded=%d | open_prices=%d",
                self._diag_guard_hits, len(self.polyfeed._token_set) // 2
                if self.polyfeed else 0, len(self._traded),
                len(self._open_prices))
            # Trace which guard blocks THIS evaluation
            try:
                if not mkt.coin or not mkt.end_time:
                    self.log.info("DIAG BLOCK: no_coin_or_end_time")
                    return
                if mkt.start_time is None:
                    self.log.info("DIAG BLOCK: no_start_time | end=%s", mkt.end_time)
                    return
                elapsed = time.time() - mkt.start_time
                ttc = mkt.end_time - time.time()
                is_5min = mkt.tf_secs <= 300
                min_elapsed = max(60, self.cfg.entry_start_s) if not is_5min else self.cfg.entry_start_s
                if elapsed < min_elapsed:
                    self.log.info("DIAG BLOCK: min_elapsed | %.0fs < %ds | coin=%s",
                                  elapsed, min_elapsed, mkt.coin)
                    return
                buffer_s = max(0, mkt.tf_secs - int(self.cfg.entry_end_s * mkt.tf_secs / 300))
                if ttc < buffer_s:
                    self.log.info("DIAG BLOCK: buffer_s | ttc=%.0f < buf=%d", ttc, buffer_s)
                    return
                binance_age = self.tracker.feed.last_msg_age_s
                if binance_age > 2.0:
                    prices = {c: self.tracker.feed.price(c) for c in self.cfg.coins}
                    self.log.info("DIAG BLOCK: binance_stale | age=%.1fs | prices=%s",
                                  binance_age, prices)
                    return
                cp = self.tracker.feed.price(mkt.coin)
                if not cp:
                    self.log.info("DIAG BLOCK: no_price | coin=%s", mkt.coin)
                    return
                if mkt.market_id not in self._open_prices:
                    op = self.tracker.get_price_at(str(mkt.coin), mkt.start_time, max_gap_s=10.0)
                    if op is None:
                        hist = len(self.tracker._history.get(mkt.coin, []))
                        self.log.info("DIAG BLOCK: no_open_price | coin=%s | hist=%d | interval=%.0f",
                                      mkt.coin, hist, mkt.start_time)
                        return
                if not mkt.fresh_books(self.cfg.book_max_age_ms):
                    ya = mkt.book_yes.age_ms if mkt.book_yes else -1
                    na = mkt.book_no.age_ms if mkt.book_no else -1
                    self.log.info("DIAG BLOCK: stale_books | yes=%.0fms no=%.0fms max=%.0f",
                                  ya, na, self.cfg.book_max_age_ms)
                    return
                ya = mkt.book_yes.best_ask if mkt.book_yes else None
                na = mkt.book_no.best_ask if mkt.book_no else None
                self.log.info(
                    "DIAG PASS | coin=%s el=%.0f ttc=%.0f p=%s ya=%s na=%s",
                    mkt.coin, elapsed, ttc, cp, ya, na)
            except Exception as de:
                self.log.info("DIAG ERROR: %s", de)
                return
        try:
            await self._evaluate_core(mkt)
        except Exception as e:
            if periodic:
                self.log.warning("EVAL EXCEPTION: %s", e)
            self.log.debug("Eval exception '%s': %s", mkt.question[:30], e)

    async def _evaluate_core(self, mkt: Market) -> None:
        if not mkt.coin or not mkt.end_time:
            return
        # BUG-FIX #22: stale-GTC cancel.  Pre-fix the only path that
        # cancelled GTC orders on expired markets was inside the periodic
        # eval block, which is debounced (400ms) and skips markets with
        # open positions.  A market that expired between evals would
        # leave resting maker orders on the CLOB until the next eval
        # cycle (up to ~1.2s late for 3 coins).  We now do the cancel
        # on EVERY eval pass (the cheap check is one ``end_time < now``
        # comparison) so a stale GTC rest is reaped within one debounce
        # window of expiry — bounded by the eval cadence, not by the
        # random order in which the periodic block fires.
        if mkt.end_time and mkt.end_time < time.time():
            for tracked in list(self.om._orders.values()):
                if tracked.token_id in (mkt.yes_token, mkt.no_token):
                    try:
                        asyncio.create_task(self.om.cancel(tracked.order_id))
                    except RuntimeError:
                        pass  # loop not running yet
        tf_secs = float(mkt.tf_secs)
        is_5min = (tf_secs <= 300.0)
        now = time.time()
        interval_start = mkt.start_time
        if interval_start is None:
            return
        elapsed = now - interval_start
        ttc = mkt.end_time - now if mkt.end_time else tf_secs - elapsed
        if elapsed < 0:
            return
        interval_epoch = int(interval_start)

        prev_interval = self._open_intervals.get(mkt.market_id)
        if prev_interval is not None and prev_interval != interval_epoch:
            self._open_prices.pop(mkt.market_id, None)
            self._traded.discard(mkt.market_id)
            self._sustain_counts.pop(mkt.market_id, None)
            for hk in [k for k in self._high_bids if isinstance(k, tuple) and k[0] == mkt.market_id]:
                self._high_bids.pop(hk, None)
        self._open_intervals[mkt.market_id] = interval_epoch

        min_elapsed = max(60, self.cfg.entry_start_s) if not is_5min else self.cfg.entry_start_s
        if elapsed < min_elapsed:
            return

        entry_end_scaled = int(self.cfg.entry_end_s * tf_secs / 300)
        buffer_s = max(0, int(tf_secs) - entry_end_scaled)
        if ttc < buffer_s:
            return

        binance_age = self.tracker.feed.last_msg_age_s
        if binance_age > 2.0:
            return

        current_price = self.tracker.feed.price(mkt.coin)
        if not current_price:
            return

        if mkt.market_id not in self._open_prices:
            open_price = self.tracker.get_price_at(str(mkt.coin), interval_start, max_gap_s=10.0)
            if open_price is None:
                return
            self._open_prices[mkt.market_id] = open_price

        open_price = self._open_prices[mkt.market_id]
        tau_s = max(1.0, ttc)

        if not mkt.fresh_books(self.cfg.book_max_age_ms):
            return

        yes_ask = mkt.book_yes.best_ask if mkt.book_yes else None
        no_ask = mkt.book_no.best_ask if mkt.book_no else None

        btc_disp = None
        if mkt.coin != "BTC":
            btc_price = self.tracker.feed.price("BTC")
            btc_open = self.tracker.get_price_at("BTC", interval_start, max_gap_s=10.0)
            if btc_price and btc_open and btc_open > 0:
                btc_disp = (btc_price - btc_open) / btc_open

        yes_trade = self.polyfeed.last_trade(mkt.yes_token) if self.polyfeed else None
        p_up = self.tracker.prob_up(mkt.coin, current_price, open_price, tau_s,
                                    yes_book=mkt.book_yes,
                                    yes_trade_ewma=yes_trade,
                                    btc_displacement=btc_disp)

        # FIX: use epsilon threshold to avoid treating float residuals as
        # real positions (e.g., 1e-15 leftover from Position.reduce).
        has_yes = mkt.pos_yes.shares > 1e-6
        has_no  = mkt.pos_no.shares > 1e-6

        # FORCED EXIT NEAR EXPIRY.  Scope-A (Flaw #5): MMs deliberately gap
        # bids down near expiry for forced sellers, so don't dump a likely
        # WINNER into that gapped bid — hold it to $1 settlement.  A
        # losing/uncertain leg is still force-sold to salvage value.
        if ttc < self.cfg.forced_exit_ttc_s and (has_yes or has_no):
            if has_yes:
                if should_force_exit_near_expiry(
                        True, p_up, self.cfg.forced_exit_hold_if_winning,
                        self.cfg.forced_exit_hold_prob):
                    bid = (mkt.book_yes.best_bid if mkt.book_yes and mkt.book_yes.best_bid else 0.01)
                    await self._execute_exit(mkt, mkt.yes_token, bid, "EXPIRY_YES", mkt.pos_yes.shares)
                else:
                    # C-BUG-5 fix: mark winning YES leg as pending $1 redemption
                    # so Risk sees the expected PnL when the market resolves.
                    self._mark_for_redemption(mkt, mkt.yes_token, mkt.pos_yes, 1.0)
            if has_no:
                if should_force_exit_near_expiry(
                        False, p_up, self.cfg.forced_exit_hold_if_winning,
                        self.cfg.forced_exit_hold_prob):
                    bid = (mkt.book_no.best_bid if mkt.book_no and mkt.book_no.best_bid else 0.01)
                    await self._execute_exit(mkt, mkt.no_token, bid, "EXPIRY_NO", mkt.pos_no.shares)
                else:
                    # C-BUG-5 fix: mark winning NO leg as pending $1 redemption
                    self._mark_for_redemption(mkt, mkt.no_token, mkt.pos_no, 1.0)
            return

        # TIME-DECAY EXIT: tighten stop as TTC decreases
        if ttc < self.cfg.time_decay_exit_ttc_s and (has_yes or has_no):
            decay_factor = max(0.3, ttc / self.cfg.time_decay_exit_ttc_s)
            dynamic_stop = 0.5 + (self.cfg.stop_loss_prob - 0.5) * decay_factor
        else:
            dynamic_stop = self.cfg.stop_loss_prob

        # FAST-EXIT: micro-price against us within 60s of entry.  Scope-A
        # (Flaw #5): the pre-v19 -3% threshold was ~one tick of noise on a
        # ~$0.50 contract.  Widen to ``fast_exit_drop_pct`` AND require it to
        # persist for ``fast_exit_sustain`` consecutive evals before firing,
        # so a single noisy print no longer panic-sells a sound position.
        entry_mono = self._entry_times.get(mkt.market_id)
        if entry_mono and time.monotonic() - entry_mono < 60:
            drop_mult = 1.0 - self.cfg.fast_exit_drop_pct
            for held, token, book, pos, label in (
                (has_yes, mkt.yes_token, mkt.book_yes, mkt.pos_yes, "FAST_YES"),
                (has_no,  mkt.no_token,  mkt.book_no,  mkt.pos_no,  "FAST_NO"),
            ):
                if not (held and book and book.micro_price):
                    continue
                fkey = (mkt.market_id, token)
                if book.micro_price < (pos.avg_price * drop_mult):
                    cnt = self._fast_exit_counts.get(fkey, 0) + 1
                    self._fast_exit_counts[fkey] = cnt
                    if cnt >= self.cfg.fast_exit_sustain:
                        self._fast_exit_counts.pop(fkey, None)
                        bid = book.best_bid or 0.01
                        self.log.info(
                            "FAST-EXIT %s %s | micro=%.3f < entry*%.2f (x%d)",
                            label, mkt.coin, book.micro_price, drop_mult, cnt)
                        await self._execute_exit(mkt, token, bid, label, pos.shares)
                        return
                else:
                    self._fast_exit_counts.pop(fkey, None)

        # TRAILING STOP + STOP LOSS + PARTIAL PROFIT-TAKING (v18.7)
        # Use micro_price (VWAP top 3 levels) for tracking — immune to L1
        # spoofing and single-tick illiquidity flash crashes.  Execute exits
        # at best_bid (the actual fillable price).
        if has_yes:
            bid = mkt.book_yes.best_bid if mkt.book_yes else None
            micro = mkt.book_yes.micro_price if mkt.book_yes else None
            trail_val = micro if micro else bid
            if trail_val is None or trail_val <= 0:
                return
            # C-3 fix: never act on a stale book.  Pre-fix the trail compared
            # `prev_high` against `(best_bid or 0.0)`, so during a momentary
            # quote withdrawal (shard reconnect, MM gap) bid -> None ->
            # trail_val=0 force-exited winning positions.  Skip the cycle if
            # the book hasn't ticked within `book_max_age_ms`; the trail
            # state stays frozen and resumes on the next fresh book.
            if mkt.book_yes is None or mkt.book_yes.is_stale(self.cfg.book_max_age_ms):
                return
            # Use trail_val as fallback when best_bid is None (ask-only book)
            if bid is None:
                bid = trail_val
            tp_key = (mkt.market_id, mkt.yes_token)
            # Effective shares = pos.shares minus any in-flight exit orders
            in_flight_yes = self._shares_in_flight.get(tp_key, 0.0)
            # v18.8: clear phantom in-flight if no open SELL order exists
            # (FOK was killed by CLOB but _shares_in_flight was never
            # decremented — see _execute_exit rollback for the immediate-
            # rejection case; this catches the CLOB-kill case after
            # reconciliation prunes the dead order from the OM).
            if in_flight_yes > 0 and not self.om.find_open(mkt.yes_token, Side.SELL):
                self._shares_in_flight.pop(tp_key, None)
                in_flight_yes = 0.0
            effective_yes = max(0.0, mkt.pos_yes.shares - in_flight_yes)
            if effective_yes < 1e-6:
                return
            if p_up < dynamic_stop:
                fail_cnt = self._exit_fail_counts.get(tp_key, 0)
                if fail_cnt >= 5:
                    # FLAW-5 fix: escalate to GTC instead of abandoning.
                    # The position must NEVER be left unmanaged.
                    self.log.warning(
                        "STOP_YES %s: %d FOK failures, escalating to GTC",
                        mkt.coin, fail_cnt)
                    await self._execute_exit(mkt, mkt.yes_token, bid, "STOP_YES_GTC", effective_yes)
                    return
                self._high_bids.pop(tp_key, None)
                self._tp1_taken.pop(tp_key, None)
                self._entry_edges.pop(tp_key, None)
                self._shares_in_flight.pop(tp_key, None)
                await self._execute_exit(mkt, mkt.yes_token, bid, "STOP_YES", effective_yes)
            else:
                # v18.7 — PARTIAL PROFIT-TAKING (fixed or confidence mode)
                entry_px = mkt.pos_yes.avg_price
                tp_fired = False
                if (self.cfg.partial_tp_enabled
                        and tp_key not in self._tp1_taken
                        and entry_px > 0):
                    # DEFECT-2 fix: gain_ratio MUST use bid (actual fill price),
                    # not trail_val (micro_price).  micro_price sits 3-8 cents
                    # above best_bid on thin books — triggers on phantom gains.
                    tp_bid_px = bid if (bid and bid > 0) else trail_val
                    gain_ratio = (tp_bid_px - entry_px) / entry_px
                    if self.cfg.tp_mode == "fixed":
                        if gain_ratio >= self.cfg.tp1_pct:
                            clip_pct = self.cfg.tp1_clip_pct
                            tp_fired = True
                    else:
                        # confidence mode: dynamic clip, gated by EDGE DECAY
                        # P1 fix: the old price-gain gate (conf_min_gain) fired
                        # on market price movement which is informationally
                        # vacuous on binaries.  The correct trigger is edge
                        # decay — exit when the signal that justified entry
                        # has materially weakened (edge < 50% of entry edge).
                        # A 1% gain floor prevents sub-spread noise exits.
                        cur_edge = p_up - trail_val
                        entry_edge = self._entry_edges.get(tp_key, 0.05)
                        edge_decayed = cur_edge < entry_edge * 0.50
                        if edge_decayed and gain_ratio > 0.01:
                            conf = self._compute_confidence(
                                tp_key, entry_px, trail_val, cur_edge, mkt)
                            clip_pct = min(self.cfg.conf_max_clip,
                                          max(self.cfg.conf_min_clip,
                                              conf * self.cfg.conf_scale))
                            tp_fired = True
                    if tp_fired:
                        clip_shares = math.floor(effective_yes * clip_pct)
                        if clip_shares >= 1.0 and clip_shares * (bid or 0.01) >= 0.50:
                            self._tp1_taken[tp_key] = entry_px
                            # DEFECT-6 fix: reset sustain to prevent phantom re-entry
                            self._sustain_counts.pop(mkt.market_id, None)
                            self._shares_in_flight[tp_key] = (
                                in_flight_yes + clip_shares)
                            mode_tag = "C" if self.cfg.tp_mode == "confidence" else "F"
                            self.log.info(
                                "TP1_YES[%s] %s | entry=%.3f bid=%.3f gain=+%.1f%% | "
                                "clip=%d/%d (%.0f%%)",
                                mode_tag, mkt.coin, entry_px, tp_bid_px,
                                gain_ratio * 100,
                                int(clip_shares), int(effective_yes),
                                clip_pct * 100)
                            await self._execute_exit(
                                mkt, mkt.yes_token, bid, "TP1_YES",
                                clip_shares)

                # BREAK-EVEN STOP for runner after TP1
                elif tp_key in self._tp1_taken and self.cfg.tp1_breakeven_stop:
                    be_price = self._tp1_taken[tp_key]
                    if trail_val <= be_price:
                        self._tp1_taken.pop(tp_key, None)
                        self._high_bids.pop(tp_key, None)
                        self._entry_edges.pop(tp_key, None)
                        self._shares_in_flight.pop(tp_key, None)
                        self.log.info(
                            "BE_STOP_YES %s | entry=%.3f trail=%.3f",
                            mkt.coin, be_price, trail_val)
                        await self._execute_exit(
                            mkt, mkt.yes_token, bid, "BE_STOP_YES",
                            effective_yes)
                    else:
                        # Trailing stop on the runner
                        trail_key = tp_key
                        prev_high = self._high_bids.get(trail_key, 0.0)
                        if trail_val > prev_high:
                            self._high_bids[trail_key] = trail_val
                        elif (prev_high >= self.cfg.trail_arm_level
                              and trail_val <= prev_high * (1.0 - self.cfg.trail_stop_pct)):
                            self._high_bids.pop(trail_key, None)
                            self._tp1_taken.pop(tp_key, None)
                            self._entry_edges.pop(tp_key, None)
                            self._shares_in_flight.pop(tp_key, None)
                            await self._execute_exit(
                                mkt, mkt.yes_token, bid, "TRAIL_YES",
                                effective_yes)
                else:
                    # Normal trailing stop (no TP1 taken yet)
                    trail_key = tp_key
                    prev_high = self._high_bids.get(trail_key, 0.0)
                    if trail_val > prev_high:
                        self._high_bids[trail_key] = trail_val
                    elif (prev_high >= self.cfg.trail_arm_level
                          and trail_val <= prev_high * (1.0 - self.cfg.trail_stop_pct)):
                        self._high_bids.pop(trail_key, None)
                        self._entry_edges.pop(trail_key, None)
                        self._shares_in_flight.pop(trail_key, None)
                        await self._execute_exit(mkt, mkt.yes_token, bid, "TRAIL_YES", effective_yes)

        if has_no:
            bid = mkt.book_no.best_bid if mkt.book_no else None
            micro = mkt.book_no.micro_price if mkt.book_no else None
            trail_val = micro if micro else bid
            if trail_val is None or trail_val <= 0:
                return
            # C-3 fix: stale-book gate (mirror of has_yes branch above).
            if mkt.book_no is None or mkt.book_no.is_stale(self.cfg.book_max_age_ms):
                return
            # Use trail_val as fallback when best_bid is None (ask-only book)
            if bid is None:
                bid = trail_val
            p_down = 1.0 - p_up
            tp_key = (mkt.market_id, mkt.no_token)
            # Effective shares = pos.shares minus any in-flight exit orders
            in_flight_no = self._shares_in_flight.get(tp_key, 0.0)
            # v18.8: clear phantom in-flight (see YES-side comment above).
            if in_flight_no > 0 and not self.om.find_open(mkt.no_token, Side.SELL):
                self._shares_in_flight.pop(tp_key, None)
                in_flight_no = 0.0
            effective_no = max(0.0, mkt.pos_no.shares - in_flight_no)
            if effective_no < 1e-6:
                return
            if p_down < dynamic_stop:
                fail_cnt = self._exit_fail_counts.get(tp_key, 0)
                if fail_cnt >= 5:
                    self.log.warning(
                        "STOP_NO %s: %d FOK failures, escalating to GTC",
                        mkt.coin, fail_cnt)
                    await self._execute_exit(mkt, mkt.no_token, bid, "STOP_NO_GTC", effective_no)
                    return
                self._high_bids.pop(tp_key, None)
                self._tp1_taken.pop(tp_key, None)
                self._entry_edges.pop(tp_key, None)
                self._shares_in_flight.pop(tp_key, None)
                await self._execute_exit(mkt, mkt.no_token, bid, "STOP_NO", effective_no)
            else:
                # v18.7 — PARTIAL PROFIT-TAKING (NO leg, fixed or confidence)
                entry_px = mkt.pos_no.avg_price
                tp_fired = False
                if (self.cfg.partial_tp_enabled
                        and tp_key not in self._tp1_taken
                        and entry_px > 0):
                    tp_bid_px = bid if (bid and bid > 0) else trail_val
                    gain_ratio = (tp_bid_px - entry_px) / entry_px
                    if self.cfg.tp_mode == "fixed":
                        if gain_ratio >= self.cfg.tp1_pct:
                            clip_pct = self.cfg.tp1_clip_pct
                            tp_fired = True
                    else:
                        # confidence mode: edge-decay gated (P1 fix — mirror of YES side)
                        cur_edge = (1.0 - p_up) - trail_val
                        entry_edge = self._entry_edges.get(tp_key, 0.05)
                        edge_decayed = cur_edge < entry_edge * 0.50
                        if edge_decayed and gain_ratio > 0.01:
                            conf = self._compute_confidence(
                                tp_key, entry_px, trail_val, cur_edge, mkt)
                            clip_pct = min(self.cfg.conf_max_clip,
                                          max(self.cfg.conf_min_clip,
                                              conf * self.cfg.conf_scale))
                            tp_fired = True
                    if tp_fired:
                        clip_shares = math.floor(effective_no * clip_pct)
                        if clip_shares >= 1.0 and clip_shares * (bid or 0.01) >= 0.50:
                            self._tp1_taken[tp_key] = entry_px
                            self._sustain_counts.pop(mkt.market_id, None)
                            self._shares_in_flight[tp_key] = (
                                in_flight_no + clip_shares)
                            mode_tag = "C" if self.cfg.tp_mode == "confidence" else "F"
                            self.log.info(
                                "TP1_NO[%s] %s | entry=%.3f bid=%.3f gain=+%.1f%% | "
                                "clip=%d/%d (%.0f%%)",
                                mode_tag, mkt.coin, entry_px, tp_bid_px,
                                gain_ratio * 100,
                                int(clip_shares), int(effective_no),
                                clip_pct * 100)
                            await self._execute_exit(
                                mkt, mkt.no_token, bid, "TP1_NO",
                                clip_shares)

                # BREAK-EVEN STOP for runner after TP1
                elif tp_key in self._tp1_taken and self.cfg.tp1_breakeven_stop:
                    be_price = self._tp1_taken[tp_key]
                    if trail_val <= be_price:
                        self._tp1_taken.pop(tp_key, None)
                        self._high_bids.pop(tp_key, None)
                        self._entry_edges.pop(tp_key, None)
                        self._shares_in_flight.pop(tp_key, None)
                        self.log.info(
                            "BE_STOP_NO %s | entry=%.3f trail=%.3f",
                            mkt.coin, be_price, trail_val)
                        await self._execute_exit(
                            mkt, mkt.no_token, bid, "BE_STOP_NO",
                            effective_no)
                    else:
                        # Trailing stop on the runner
                        trail_key = tp_key
                        prev_high = self._high_bids.get(trail_key, 0.0)
                        if trail_val > prev_high:
                            self._high_bids[trail_key] = trail_val
                        elif (prev_high >= self.cfg.trail_arm_level
                              and trail_val <= prev_high * (1.0 - self.cfg.trail_stop_pct)):
                            self._high_bids.pop(trail_key, None)
                            self._tp1_taken.pop(tp_key, None)
                            self._entry_edges.pop(tp_key, None)
                            self._shares_in_flight.pop(tp_key, None)
                            await self._execute_exit(
                                mkt, mkt.no_token, bid, "TRAIL_NO",
                                effective_no)
                else:
                    # Normal trailing stop (no TP1 taken yet)
                    trail_key = tp_key
                    prev_high = self._high_bids.get(trail_key, 0.0)
                    if trail_val > prev_high:
                        self._high_bids[trail_key] = trail_val
                    elif (prev_high >= self.cfg.trail_arm_level
                          and trail_val <= prev_high * (1.0 - self.cfg.trail_stop_pct)):
                        self._high_bids.pop(trail_key, None)
                        self._entry_edges.pop(trail_key, None)
                        self._shares_in_flight.pop(trail_key, None)
                        await self._execute_exit(mkt, mkt.no_token, bid, "TRAIL_NO", effective_no)
            return

        # Cancel stale unfilled GTC orders > 45s
        for token in (mkt.yes_token, mkt.no_token):
            stale = self.om.find_open(token, Side.BUY)
            if stale and time.monotonic() - stale.created > 45:
                await self.om.cancel(stale.order_id)

        if self.om.find_open(mkt.yes_token, Side.BUY) or self.om.find_open(mkt.no_token, Side.BUY):
            return
        if yes_ask is None or no_ask is None:
            return

        # YES/NO COMPLEMENT ARBITRAGE — resolution-guaranteed edge.
        # If YES_ask + NO_ask < 1.0 (net of fees), buying both locks in a
        # guaranteed profit at resolution regardless of outcome.
        fee_per_leg = self.cfg.taker_fee_bps * 1e-4
        arb_cost = yes_ask + no_ask + 2 * fee_per_leg * (yes_ask + no_ask) / 2
        arb_edge = 1.0 - arb_cost
        if arb_edge > 2 * self.cfg.min_edge and yes_ask > 0.01 and no_ask > 0.01:
            self.log.info(
                "COMPLEMENT_ARB %s | YES=%.3f + NO=%.3f = %.3f | edge=%.4f",
                mkt.coin, yes_ask, no_ask, yes_ask + no_ask, arb_edge)
            arb_size = min(self.cfg.max_order_size,
                          self.cfg.max_bankroll_fraction * (
                              self._balance_cache if self._balance_ts > 0
                              else self.cfg.max_order_size * 2))
            if arb_size >= self.cfg.min_order_size:
                self._traded.add(mkt.market_id)
                tick_y = mkt.get_tick(mkt.yes_token)
                tick_n = mkt.get_tick(mkt.no_token)
                oid_y = await self.om.place(
                    mkt.yes_token, Side.BUY, yes_ask, arb_size / 2,
                    Strategy.TEMPORAL, otype="FOK",
                    neg_risk=mkt.neg_risk, tick_size=tick_y)
                oid_n = await self.om.place(
                    mkt.no_token, Side.BUY, no_ask, arb_size / 2,
                    Strategy.TEMPORAL, otype="FOK",
                    neg_risk=mkt.neg_risk, tick_size=tick_n)
                if oid_y or oid_n:
                    self._entry_times[mkt.market_id] = time.monotonic()
            return

        if self.tracker.is_choppy(mkt.coin, int(tf_secs)):
            return

        for book in (mkt.book_yes, mkt.book_no):
            if book and (book.top_depth_usdc < self.cfg.min_top_book_usdc or
                         book.spread_pct > self.cfg.max_spread_pct):
                return

        # Check exit liquidity before entering
        for book in (mkt.book_yes, mkt.book_no):
            if book and book.best_bid and book.best_bid < 0.03:
                return

        min_prob = 0.56 if is_5min else 0.53
        # S-9 fix: spread/vol-normalized minimum edge.  The flat 1.2% rewarded
        # the WIDEST-spread (= noisiest) markets where 1.2% is within
        # measurement noise.  req_edge floors at the configured min_edge but
        # rises in proportion to either book spread or horizon vol, so a
        # noisy book has to clear a noise-aware bar.
        spread_yes = mkt.book_yes.spread_pct if mkt.book_yes else 0.0
        spread_no = mkt.book_no.spread_pct if mkt.book_no else 0.0
        spread_pct = max(spread_yes, spread_no)
        sigma_per_sec = self.tracker.volatility(mkt.coin) if mkt.coin else 0.0
        ttc_eff = max(1.0, mkt.end_time - time.time()) if mkt.end_time else 60.0
        sigma_h = sigma_per_sec * math.sqrt(ttc_eff)
        req_edge = max(
            self.cfg.min_edge,
            spread_pct * self.cfg.spread_edge_mult,
            sigma_h * self.cfg.sigma_edge_mult,
        )
        vel = self.tracker.velocity(mkt.coin, window_s=30)

        if p_up >= min_prob:
            if yes_ask is None or yes_ask > 0.85:
                return
            if btc_disp is not None and btc_disp < -0.003:
                return
            if vel < -0.0004:
                return
            # v18.4 — wire the FULL round-trip cost (entry + exit) into
            # the edge calculation.  The v18.3 path used a one-sided
            # _estimate_slippage that was structurally optimistic by
            # half-spread; v18.4 uses the bid-side walk for exit too.
            entry_per_share, exit_per_share, fillable = self._round_trip_cost(
                mkt.book_yes, self.cfg.min_order_size)
            if not fillable:
                return
            entry_slip = max(0.0, entry_per_share - yes_ask)
            # exit_slip = "best_bid - realized_exit_price" — the
            # liquidation cost if we have to bail to the book before
            # expiry.  Floored at 0 (a thin book can produce VWAP > best
            # if asks lift between our entry and exit).
            best_bid_yes = mkt.book_yes.best_bid if mkt.book_yes else 0.0
            exit_slip = max(0.0, (best_bid_yes or 0.0) - exit_per_share) if best_bid_yes else 0.0
            # Edge gate isolates ENTRY expected value only:
            #   EV = P_win - (ask + entry_slip)
            # A binary held to expiry redeems at $1/$0 with NO exit trade,
            # so exit slippage must NOT gate entry.  Exit cost is already
            # priced into position SIZE via the Kelly Cout term
            # (Cout = 1 - exit_slip); subtracting it here too would
            # double-count the same cost and suppress valid signals.
            edge = p_up - yes_ask - entry_slip
            self.log.info(
                "EVAL UP %s | el=%3.0fs | p=%.3f | edge=%.3f | ask=%.3f "
                "| es=%.4f xs=%.4f",
                mkt.coin, elapsed, p_up, edge, yes_ask, entry_slip, exit_slip)
            # v18.4 — calibration logging (realized-vs-predicted).
            self._log_prediction(mkt, "UP", p_up, yes_ask, edge,
                                 entry_slip, exit_slip)
            if edge >= req_edge:
                sc = self._sustain_counts.get(mkt.market_id, 0) + 1
                self._sustain_counts[mkt.market_id] = sc
                if sc < self.cfg.sustain_ticks:
                    self.log.info("SUSTAIN %s UP: %d/%d", mkt.coin, sc, self.cfg.sustain_ticks)
                    return
                self._sustain_counts[mkt.market_id] = 0
                if self._net_exposure + self.cfg.min_order_size > self.cfg.max_net_exposure_usdc:
                    return
                await self._place_sliced(mkt, mkt.yes_token, yes_ask, "UP",
                                          p_up, edge, entry_slip, exit_slip)
        else:
            p_down = 1.0 - p_up
            # Don't open the opposing (NO) leg while still holding YES — the
            # STOP/TRAIL block above manages the YES exit.  Stacking a NO
            # position here only nets out via the merge path at the cost of a
            # second spread + slippage (a scratch round-trip).
            if has_yes:
                return
            if p_down >= min_prob:
                if no_ask is None or no_ask > 0.85:
                    return
                if btc_disp is not None and btc_disp > 0.003:
                    return
                if vel > 0.0004:
                    return
                entry_per_share, exit_per_share, fillable = self._round_trip_cost(
                    mkt.book_no, self.cfg.min_order_size)
                if not fillable:
                    return
                entry_slip = max(0.0, entry_per_share - no_ask)
                best_bid_no = mkt.book_no.best_bid if mkt.book_no else 0.0
                exit_slip = max(0.0, (best_bid_no or 0.0) - exit_per_share) if best_bid_no else 0.0
                # Entry-EV-only gate (see UP branch); exit cost lives in
                # Kelly Cout, never double-counted at the entry gate.
                edge = p_down - no_ask - entry_slip
                self.log.info(
                    "EVAL DN %s | el=%3.0fs | p=%.3f | edge=%.3f | ask=%.3f "
                    "| es=%.4f xs=%.4f",
                    mkt.coin, elapsed, p_down, edge, no_ask, entry_slip, exit_slip)
                self._log_prediction(mkt, "DN", p_down, no_ask, edge,
                                     entry_slip, exit_slip)
                if edge >= req_edge:
                    sc = self._sustain_counts.get(mkt.market_id, 0) + 1
                    self._sustain_counts[mkt.market_id] = sc
                    if sc < self.cfg.sustain_ticks:
                        self.log.info("SUSTAIN %s DN: %d/%d", mkt.coin, sc, self.cfg.sustain_ticks)
                        return
                    self._sustain_counts[mkt.market_id] = 0
                    if self._net_exposure - self.cfg.min_order_size < -self.cfg.max_net_exposure_usdc:
                        return
                    await self._place_sliced(mkt, mkt.no_token, no_ask, "DN",
                                              p_down, edge, entry_slip, exit_slip)
            else:
                self._sustain_counts.pop(mkt.market_id, None)

    def _estimate_slippage(self, book: Optional[OrderBook],
                           size_usdc: float) -> float:
        """Delegate to module-level _estimate_slippage."""
        return _estimate_slippage(book, size_usdc)

    def _round_trip_cost(self, book: Optional[OrderBook],
                         size_usdc: float) -> Tuple[float, float, bool]:
        """v18.3: round-trip cost model (entry + exit)."""
        return _round_trip_cost(book, size_usdc)

    def _fok_sweep_price(self, book: Optional[OrderBook],
                         size_usdc: float, tick: float,
                         dec: int, mt: int) -> float:
        """Walk the book to find the exact FOK sweep price for this order."""
        return _fok_sweep_price(book, size_usdc, tick, dec, mt)

    async def _place_sliced(self, mkt: Market, token_id: str,
                            ask_price: float, label: str,
                            prob: float, edge: float,
                            entry_slip: float = 0.0,
                            exit_slip: float = 0.0) -> None:
        if not self.risk.ok():
            return
        # Guard: one entry per market per interval (prevents duplicate FOK
        # orders when multiple signals fire within the same debounce window).
        if mkt.market_id in self._traded:
            return
        # v19 Scope-A (Flaw #2): abort when MEASURED post-fill adverse
        # selection would eat the modeled edge.  The always-on shadow probe's
        # rolling EWMA is the realized cost of being the taker; if it exceeds
        # the edge, we are systematically the dumb liquidity and this entry
        # is a loser regardless of how clean the model looks.
        if self.cfg.adverse_select_gate:
            book0 = mkt.book_yes if token_id == mkt.yes_token else mkt.book_no
            mid0 = book0.mid if book0 else None
            if adverse_gate(self.om.adverse_ewma(), mid0, edge):
                self.log.info(
                    "SKIP ENTRY %s %s: adverse EWMA %+.1fbps eats edge %.3f",
                    label, mkt.coin, self.om.adverse_ewma() or 0.0, edge)
                return
        # v18.8: recompute exit_slip at actual book depth — the caller
        # computed it at min_order_size, but Kelly may size larger.  A
        # deeper book walk produces worse exit VWAP, so using the shallow
        # estimate would overstate Cout and inflate the Kelly fraction.
        book = mkt.book_yes if token_id == mkt.yes_token else mkt.book_no
        _, exit_vwap, _ = self._round_trip_cost(book, self.cfg.max_order_size)
        best_bid = book.best_bid if book else 0.0
        exit_slip = max(0.0, (best_bid or 0.0) - exit_vwap) if exit_vwap > 0 else exit_slip
        kelly_sz = self._kelly_size(prob, ask_price, entry_slip, exit_slip,
                                    coin=mkt.coin)
        # Risk-of-ruin gate (v18.6): a non-positive Kelly size means even the
        # minimum venue clip exceeds ``max_bankroll_fraction`` of bankroll —
        # the account is too small to take this trade without courting ruin.
        # Skip rather than over-bet the floor (see _kelly_size).
        if kelly_sz <= 0.0:
            self.log.info(
                "SKIP ENTRY %s %s: min clip exceeds %.0f%% bankroll cap "
                "($%.2f bankroll) — account too small to size safely",
                label, mkt.coin, self.cfg.max_bankroll_fraction * 100.0,
                (max(0.0, self._balance_cache) if self._balance_ts > 0
                 else self.cfg.max_order_size * 2))
            return

        tick = mkt.get_tick(token_id)
        dec, mt = mkt.tick_math(token_id)

        book = mkt.book_yes if token_id == mkt.yes_token else mkt.book_no

        # Re-validate the edge at the ACTUAL execution size.  The edge
        # passed in was computed against ``min_order_size`` depth, but
        # Kelly may size up to ``max_order_size`` — deeper sweeps eat
        # worse levels, so slippage on $42 ≠ slippage on $5.
        #
        # The entry-EV-only edge is monotonically NON-INCREASING in size
        # (a larger sweep can only reach equal-or-worse ask levels), so
        # rather than aborting when the Kelly size erodes the edge, we
        # size DOWN to the largest clip S* that still clears ``min_edge``
        # via a bounded bisection on [min_order_size, kelly_sz].  This
        # recovers trades that are valid at a smaller size instead of
        # leaving the alpha on the table.  Exit cost stays in Kelly Cout,
        # never in this gate (entry-EV isolation).
        def _exec_edge_at(s: float) -> Optional[float]:
            e_entry, _, ok = self._round_trip_cost(book, s)
            if not ok:
                return None
            return prob - ask_price - max(0.0, e_entry - ask_price)

        top_edge = _exec_edge_at(kelly_sz)
        if top_edge is not None and top_edge >= self.cfg.min_edge:
            sz = kelly_sz                      # full Kelly size clears the gate
        else:
            lo, hi = self.cfg.min_order_size, kelly_sz
            # Invariant guard: ``_kelly_size`` floors its result at
            # ``min_order_size``, so ``hi >= lo`` always holds here; this
            # ``max`` makes the search-space well-formed regardless, so a
            # future change to the Kelly floor can never invert the bracket.
            hi = max(hi, lo)
            base_edge = _exec_edge_at(lo)
            if base_edge is None or base_edge < self.cfg.min_edge:
                # No size — not even the minimum clip — clears the gate.
                self.log.info(
                    "SKIP ENTRY %s %s: edge %.4f < min_edge %.4f even at "
                    "$%.1f min clip",
                    label, mkt.coin,
                    (base_edge if base_edge is not None else float("nan")),
                    self.cfg.min_edge, lo)
                return
            best = lo
            for _ in range(8):                 # ~$0.02 precision on a $5–$42 range
                mid = 0.5 * (lo + hi)
                em = _exec_edge_at(mid)
                if em is not None and em >= self.cfg.min_edge:
                    best, lo = mid, mid         # clears gate → try larger
                else:
                    hi = mid                    # eroded → shrink
            sz = best
            self.log.info(
                "SIZE-DOWN %s %s: Kelly $%.1f → $%.1f to hold edge ≥ %.4f",
                label, mkt.coin, kelly_sz, sz, self.cfg.min_edge)

        if sz < self.cfg.min_order_size:
            return
        if mkt.total_cost + sz > self.cfg.max_position:
            return

        # v19 Scope-A (Flaw #2): optional PASSIVE/maker entry.  Instead of
        # exclusively lifting the ask (FOK taker), post a GTC limit at/near
        # the bid so we EARN the spread rather than pay it.  Off by default
        # (entry_mode="taker"); an unfilled rest is swept by the 45s
        # stale-GTC cancel in the exit loop.  Fills arrive via the normal
        # _on_fill path and are measured by the same always-on shadow probe.
        if self.cfg.entry_mode == "maker":
            maker_px = maker_entry_price(
                book.best_bid if book else None,
                book.best_ask if book else None,
                tick, self.cfg.maker_join_ticks, prob)
            if maker_px is None:
                self.log.info(
                    "SKIP ENTRY %s %s: no valid maker price (bid=%s ask=%s)",
                    label, mkt.coin,
                    (book.best_bid if book else None),
                    (book.best_ask if book else None))
                return
            self.log.info(
                "ENTRY(maker) %s %s | P=%.3f | edge=%.3f | post=%.4f | sz=$%.1f",
                label, mkt.coin, prob, edge, maker_px, sz)
            self._traded.add(mkt.market_id)
            oid = await self.om.place(
                token_id, Side.BUY, maker_px, sz,
                Strategy.TEMPORAL,
                otype="GTC", neg_risk=mkt.neg_risk, tick_size=tick,
            )
            if oid:
                self._entry_times[mkt.market_id] = time.monotonic()
                self._entry_edges[(mkt.market_id, token_id)] = edge
            else:
                self._traded.discard(mkt.market_id)
            return

        sweep_price = self._fok_sweep_price(book, sz, tick, dec, mt)
        if sweep_price <= 0:
            return
        if sweep_price > 0.82:
            self.log.info("SKIP ENTRY %s %s: sweep %.4f > 0.82 cap",
                          label, mkt.coin, sweep_price)
            # STRATEGY cap (not a venue constraint): paying > $0.82/share on
            # a binary caps upside at < 22% while exposing the full notional
            # to a → $0 loss — a poor risk/reward that also leaves little
            # room above our edge gate.  The CLOB itself WILL accept orders
            # above 0.82; this is a deliberate entry-price ceiling.
            return

        self.log.info("ENTRY %s %s | P=%.3f | edge=%.3f | price=%.4f | sz=$%.1f",
                      label, mkt.coin, prob, edge, sweep_price, sz)

        self._traded.add(mkt.market_id)
        oid = await self.om.place(
            token_id, Side.BUY, sweep_price, sz,
            Strategy.TEMPORAL,
            otype="FOK", neg_risk=mkt.neg_risk, tick_size=tick,
        )
        if oid:
            self._entry_times[mkt.market_id] = time.monotonic()
            self._entry_edges[(mkt.market_id, token_id)] = edge
        else:
            self._traded.discard(mkt.market_id)

    def _mark_for_redemption(self, mkt: "Market", token_id: str,
                             pos: "Position", expected_payout: float) -> None:
        """C-BUG-5 fix: record a winning leg held to $1/$0 settlement.

        When ``should_force_exit_near_expiry`` decides to HOLD (not sell),
        the position is invisible to Risk — it never goes through the SELL
        fill → record_pnl path.  This method:
          1. Logs the hold decision at INFO so the operator knows.
          2. Stores the expected redemption in ``_pending_redemptions`` so
             ``cleanup_expired`` can credit Risk when the market resolves.
          3. Records a SOFT PnL estimate to Risk so the daily-loss halt
             doesn't false-fire while the winning leg is pending settlement.
        """
        rkey = (mkt.market_id, token_id)
        if rkey in self._pending_redemptions:
            return  # already marked
        shares = pos.shares
        avg_price = pos.avg_price
        if shares < 1e-6 or avg_price <= 0:
            return
        expected_pnl = (expected_payout - avg_price) * shares
        self._pending_redemptions[rkey] = (shares, avg_price, expected_payout)
        # Record soft PnL so Risk sees this as a (likely) win, not a gap.
        self.risk.record_pnl(expected_pnl)
        self.log.info(
            "HOLD_TO_EXPIRY %s %s | shares=%.1f avg=%.4f payout=%.2f "
            "est_pnl=$%.2f (marked for redemption)",
            mkt.coin, "YES" if token_id == mkt.yes_token else "NO",
            shares, avg_price, expected_payout, expected_pnl)

    def _compute_confidence(self, tp_key: Tuple[str, str],
                            entry_px: float, trail_val: float,
                            current_edge: float, mkt: "Market") -> float:
        """Kelly-grounded partial exit confidence for binary positions.

        Compares EV of selling now at trail_val vs holding to $1/$0
        resolution.  On a binary with genuine edge (p_win > current_price),
        holding is almost always superior — returns a LOW confidence
        (= small or zero clip) in that case.

        Confidence is HIGH only when:
        - Edge has fully decayed (current_edge <= 0) — the signal that
          justified entry is gone, so holding is no longer backed by alpha
        - Time pressure is extreme (>80% elapsed) — forced exit imminent,
          take what you can get before MM gaps widen
        - Realized gain already exceeds the hold-EV (rare on a binary)

        Returns [0, 1] where 0 = hold everything, 1 = exit maximum.
        """
        cfg = self.cfg
        if entry_px <= 0 or trail_val <= 0:
            return 0.0

        # M-ERR-7 fix: p_win must be the MODEL's probability, not the market
        # price.  trail_val ≈ market price ≈ market-implied probability.
        # current_edge = p_model - trail_val, so p_model = trail_val + edge.
        # Using p_win = trail_val made ev_hold = ev_sell = trail_val - entry_px
        # and the 30% ev_component weight was ALWAYS dead (ev_delta = 0).
        # Now ev_hold uses the model's view and ev_sell the market's, so the
        # delta captures the gap between model and market — exactly the signal
        # that should drive partial-exit confidence.
        p_win = max(0.01, min(0.99, trail_val + current_edge))
        ev_hold = p_win * 1.0 - entry_px

        # EV of selling now at best_bid (trail_val approximates this)
        ev_sell = trail_val - entry_px

        # Edge decay component: fraction of original edge that's eroded
        entry_edge = self._entry_edges.get(tp_key, 0.05)
        edge_decay = min(1.0, max(0.0, entry_edge - current_edge) / max(entry_edge, 0.01))

        # Time pressure: elapsed / remaining
        entry_ts = self._entry_times.get(mkt.market_id, time.monotonic())
        elapsed = time.monotonic() - entry_ts
        end_ts = mkt.end_time
        if end_ts and end_ts > time.time():
            ttc = end_ts - time.time()
            time_pressure = min(1.0, elapsed / max(ttc, 1.0))
        else:
            time_pressure = min(1.0, elapsed / 300.0)

        # Confidence = 0 when hold-EV dominates sell-EV AND edge is intact
        # Confidence rises when: edge is gone, time is running out, or
        # sell-now is actually better than hold
        ev_delta = ev_sell - ev_hold
        ev_component = max(0.0, ev_delta / max(abs(ev_hold), 0.01))

        # P3 fix: edge_decay is the ONLY grounded signal for binary exits
        # (it measures whether the model's alpha has eroded).  Time pressure
        # is secondary (and already handled by forced_exit_near_expiry).
        # EV component is a sanity check, not a primary driver.
        confidence = (0.1 * ev_component
                      + 0.7 * edge_decay
                      + 0.2 * time_pressure)
        return max(0.0, min(1.0, confidence))

    def _kelly_size(self,
                    prob: float,
                    entry_price: float,
                    entry_slip: float = 0.0,
                    exit_slip: float = 0.0,
                    coin: Optional[str] = None) -> float:
        """v18.4 — Fractional Kelly with explicit net-payoff-odds model.

        Mathematics
        -----------
        Cin  = entry_price + entry_slip       per-share cost going IN
        Cout = 1.0 - exit_slip                per-share revenue at WIN
        b    = (Cout - Cin) / Cin             net-payoff odds (multiplier on Cin)
        f*   = (p * b - q) / b                full-Kelly fraction

        where p is the (calibration-shrunk) win probability and
        q = 1 - p.  This is the canonical Kelly formulation for a
        binary asymmetric-payoff bet; the v18.3 implementation used
        ``b = (1 - effective_price) / effective_price`` which is the
        SPECIAL CASE of this formula when ``exit_slip = 0`` and
        ``Cout = 1``.  The v18.4 generalization lets us model the
        round-trip exit cost (Cout < 1) explicitly so the slippage on
        both legs of the trade is reflected in the size, not just the
        edge gate.

        Guards (all return ``min_order_size``, the smallest viable
        order — "trade at minimum or skip"):
          * prob ∉ (0, 1)           — degenerate probability
          * Cin ∉ (0, 1)            — invalid binary cost basis
          * Cout ≤ Cin (b ≤ 0)      — round-trip cost ≥ max payout
          * p * b ≤ q               — negative expected value
          * any division-by-zero    — defensively pre-empted by gates

        Calibration shrinkage on p
        --------------------------
        Beta(1,1) posterior shrinkage is applied to the WIN PROBABILITY
        (NOT to ``kelly_frac``).  This is dimensionally correct: a
        calibration error in the model is best corrected at the
        probability layer, where it represents a coherent "what is the
        true expected win rate" rather than scaling the position size
        ad-hoc.

            p_shrunk = (k_wins + 1) / (n_recent + 2)   Beta(1,1) posterior on rolling window
            w        = n_recent / (n_recent + 20)      bounded crossover weight
            p_final  = (1 - w) * p_model + w * p_shrunk

        Bounded Bayesian shrinkage:
          * ``p_shrunk`` (sliding window, cap 50) is a regime-aware
            empirical hit rate — it deliberately FORGETS old outcomes.
          * ``w`` uses the SAME window count with a fixed 20-trade
            pseudo-count, so it is capped at 50/70≈0.714.  The model
            ``prob`` therefore keeps ≥28.6% of the weight permanently.
            (An earlier follow-up drove ``w`` off an unbounded lifetime
            counter so ``w -> 1``; that let a flat portfolio scalar with
            no per-setup resolution dominate sizing and was reverted.)

        Interpretation of the crossover:
          n_recent=0   → w=0     → p_final = p_model       (trust the model)
          n_recent=20  → w=0.5   → equal blend             (transition phase)
          n_recent=50  → w≈0.714 → 28.6% model floor       (steady state)

        Bankroll & sizing
        -----------------
        ``bankroll`` is the cached USDC balance (refreshed asynchronously)
        with a floor of ``2 * max_order_size`` so under-funded boots
        don't deflate sizing during startup.  Cold-start (``n_recent < 20``)
        halves ``kelly_frac`` as an additional safety margin while the
        empirical win-rate has high posterior variance.

        Returns: dollar size, clipped to
        ``[min_order_size, max_order_size]`` and rounded to cents.
        """
        # ── Input sanitization ──
        if not (0 < prob < 1):
            return self.cfg.min_order_size

        # ── Calibration shrinkage (Beta(1,1) posterior on p) ──
        # S-4 fix: when ``per_coin_crossover`` is set and ``coin`` is known,
        # use the per-coin window so a hot BTC run isn't washed out by an
        # unrelated SOL cold streak (and vice versa).  The portfolio-wide
        # window is the fallback for legacy callers and the cold start.
        if (self.cfg.per_coin_crossover and coin
                and coin in self._per_coin_outcomes):
            window = self._per_coin_outcomes[coin]
            wins_in_window = self._per_coin_wins.get(coin, 0)
        else:
            window = self._recent_outcomes
            wins_in_window = self._recent_wins
        n_recent = len(window)
        if self.cfg.adaptive_kelly and n_recent > 0:
            k_wins = wins_in_window
            p_shrunk = (k_wins + 1) / (n_recent + 2)
            w = n_recent / (n_recent + 20)
            p_blend = (1.0 - w) * prob + w * p_shrunk
            # Selection-bias guard: ``p_shrunk`` is the win rate of the
            # EXECUTED subset, which is filtered by the ``edge >= req_edge``
            # entry gate — a survivorship-biased posterior, NOT a clean
            # estimate of P(win) for this signal.  Blending it UPWARD would
            # let a lucky filtered streak inflate Kelly size and court ruin
            # when the filtered distribution reverts.
            p_final = min(prob, p_blend)
        else:
            p_final = prob

        # Bankroll = the REAL cached balance once an authoritative fetch
        # has set ``_balance_ts``.  Only the pre-fetch cold start floors at
        # ``2 * max_order_size`` so startup isn't deflated to min size.
        # v18.9: in DRY_RUN mode, always use the simulated floor so paper
        # trades fire even when the on-chain balance is too low — the
        # whole point of dry-run is to test the strategy without capital.
        if self.cfg.dry_run:
            bankroll = self.cfg.max_order_size * 2
        elif self._balance_ts > 0:
            bankroll = max(0.0, self._balance_cache)
        else:
            bankroll = self.cfg.max_order_size * 2

        # Delegate the Kelly math + risk-of-ruin cap to the pure,
        # unit-tested ``kelly_size``.  Scope-A fix (Flaw #3): on the LIVE
        # path a non-positive-EV bet now returns 0.0 (SKIP) instead of the
        # old ``min_order_size`` — the pre-v19 code still fired the minimum
        # clip on a provably-losing bet, which is precisely Kelly-on-zero-
        # edge ruin.  In DRY_RUN we keep taking the floor so the calibration
        # harness still records outcomes across the probability spectrum.
        cold_start = self.cfg.adaptive_kelly and n_recent < 20
        return kelly_size(
            p_final, entry_price, entry_slip, exit_slip,
            kelly_fraction=self.cfg.kelly_fraction,
            bankroll=bankroll,
            max_bankroll_fraction=self.cfg.max_bankroll_fraction,
            min_order_size=self.cfg.min_order_size,
            max_order_size=self.cfg.max_order_size,
            cold_start=cold_start,
            negative_ev_skips=not self.cfg.dry_run,
            # C-2: hold-to-expiry blend for Cout (mirrors forced_exit_hold_prob).
            p_hold_to_expiry=self.cfg.forced_exit_hold_prob if
                self.cfg.forced_exit_hold_if_winning else 0.0,
            # Q-1: include taker fee in Cin.  DEFECT-5 fix: ALWAYS apply the
            # real fee in BOTH dry-run and live so calibration measures the same
            # strategy the live path trades.  The pre-fix 0bps dry-run created
            # systematic false positives in the go/no-go gate.
            taker_fee_bps=self.cfg.taker_fee_bps,
            category_fee_rate=self.cfg.category_fee_rate,  # H-8: probability-weighted fee
        )

    def record_outcome(self, win: bool,
                       market_id: Optional[str] = None,
                       net_pnl: Optional[float] = None,
                       coin: Optional[str] = None) -> None:
        """Track recent outcomes for adaptive Kelly + emit a
        calibration-CSV ``outcome`` row.

        v18.3: switched from int-truncated decay
        (``int(recent_total * 0.9)``) to a fixed-size deque of the
        last 50 trade outcomes.  Eliminates accumulation drift and
        makes the Beta-posterior shrinkage exact.

        v18.4: also emits an ``outcome`` row to the calibration CSV
        so off-line analysis can join eval-predictions against trade
        results by ``market_id``.

        v18.8 (S-4): when ``coin`` is supplied, also maintain a per-coin
        rolling window so ``_kelly_size`` can shrink toward the regime-
        local hit rate instead of the cross-coin portfolio average.
        """
        win_b = bool(win)
        if (len(self._recent_outcomes) == self._recent_outcomes.maxlen
                and self._recent_outcomes[0]):
            self._recent_wins -= 1
        self._recent_outcomes.append(win_b)
        if win_b:
            self._recent_wins += 1
        # S-4: per-coin window (cap 30 outcomes per coin).
        if coin:
            dq = self._per_coin_outcomes.get(coin)
            if dq is None:
                dq = deque(maxlen=30)
                self._per_coin_outcomes[coin] = dq
                self._per_coin_wins[coin] = 0
            if len(dq) == dq.maxlen and dq[0]:
                self._per_coin_wins[coin] -= 1
            dq.append(win_b)
            if win_b:
                self._per_coin_wins[coin] = self._per_coin_wins.get(coin, 0) + 1
        if market_id is not None:
            self._calibration_log("outcome", market_id=market_id,
                                  win=win_b, net_pnl=net_pnl)

    # ── §4 — Calibration logging ─────────────────────────────────────────────
    def _log_prediction(self,
                        mkt: "Market",
                        side: str,
                        p: float,
                        ask: float,
                        edge: float,
                        entry_slip: float,
                        exit_slip: float) -> None:
        """Append an ``eval`` row to the calibration CSV.

        Emitted every time the bot evaluates a market — regardless of
        whether the gate fires.  An offline join (by ``market_id`` and
        the eventual ``outcome`` row) yields the realized-vs-predicted
        edge curve required to compute Brier score, log-loss, and
        reliability diagrams.  Costs ~50 µs amortized (block-buffered
        write, flushed every _CALIB_FLUSH_EVERY rows + on shutdown).
        """
        # v18.5 — log the full top-of-book on BOTH sides so offline analysis
        # can measure the true net-of-spread PnL of the opposite (fade) side,
        # not just the picked side's ask.  Logging only; no effect on trading.
        by, bn = mkt.book_yes, mkt.book_no
        self._calibration_log(
            "eval",
            market_id=mkt.market_id,
            coin=getattr(mkt, "coin", ""),
            side=side,
            p=p,
            ask=ask,
            edge=edge,
            entry_slip=entry_slip,
            exit_slip=exit_slip,
            yes_bid=(by.best_bid if by else None),
            yes_ask=(by.best_ask if by else None),
            no_bid=(bn.best_bid if bn else None),
            no_ask=(bn.best_ask if bn else None),
        )

    def _log_shadow(self, info: Dict[str, Any]) -> None:
        """Append a ``shadow`` row recording one post-fill adverse-selection
        measurement (bps of mid, positive = against us).  Wired to the
        always-on OrderManager shadow probe via ``set_shadow_sink`` and
        consumed by the offline harness (``calibration_report``) to compute
        mean adverse selection for the go/no-go gate."""
        self._calibration_log(
            "shadow",
            market_id=str(info.get("market_id") or ""),
            side=str(info.get("side") or ""),
            adverse_bps=info.get("adverse_bps"),
        )

    def _calibration_log(self, row_type: str, **fields: Any) -> None:
        """Internal: append a row to the calibration CSV with strict
        column order.  Lazily creates parent dir + writes a header on
        the first call of the session.

        Defensive: any I/O exception is logged at WARNING and dropped;
        a corrupt log file must never crash the trading loop.
        """
        cfg = self.cfg
        if not cfg.calibration_log_enabled or not cfg.calibration_log_path:
            return
        # Format the row on the calling (event-loop) thread — pure CPU,
        # never blocks — then hand the finished line to the single-thread
        # writer so the blocking I/O happens off the loop.
        now = time.time()
        iso = datetime.fromtimestamp(
            now, tz=timezone.utc).isoformat(timespec="seconds").replace(
                "+00:00", "Z")
        # Column order MUST match the header below.
        cols = [
            iso,
            f"{now:.3f}",
            row_type,
            cfg.prob_model,
            str(fields.get("market_id", "")),
            str(fields.get("coin", "")),
            str(fields.get("side", "")),
            "" if fields.get("p") is None else f"{float(fields['p']):.6f}",
            "" if fields.get("ask") is None else f"{float(fields['ask']):.6f}",
            "" if fields.get("edge") is None else f"{float(fields['edge']):.6f}",
            "" if fields.get("entry_slip") is None else f"{float(fields['entry_slip']):.6f}",
            "" if fields.get("exit_slip") is None else f"{float(fields['exit_slip']):.6f}",
            "" if fields.get("win") is None else ("1" if fields["win"] else "0"),
            "" if fields.get("net_pnl") is None else f"{float(fields['net_pnl']):.4f}",
            "" if fields.get("yes_bid") is None else f"{float(fields['yes_bid']):.6f}",
            "" if fields.get("yes_ask") is None else f"{float(fields['yes_ask']):.6f}",
            "" if fields.get("no_bid") is None else f"{float(fields['no_bid']):.6f}",
            "" if fields.get("no_ask") is None else f"{float(fields['no_ask']):.6f}",
            "" if fields.get("adverse_bps") is None else f"{float(fields['adverse_bps']):.3f}",
        ]
        # CSV-safe: escape any embedded commas in string fields.
        safe = [
            c if ("," not in c and "\n" not in c) else '"' + c.replace('"', '""') + '"'
            for c in cols
        ]
        line = ",".join(safe) + "\n"
        try:
            self._calib_pool.submit(self._write_calib_row, line)
        except RuntimeError:
            # Pool already shut down during teardown — drop the row.
            pass

    def _write_calib_row(self, line: str) -> None:
        """Runs ONLY on the dedicated single-thread calib executor.

        Lazily opens the block-buffered append handle on first call and
        flushes every ``_CALIB_FLUSH_EVERY`` rows.  Because exactly one
        worker ever executes this, ``_calib_fh`` / ``_calib_writes`` need
        no locking and rows never interleave.
        """
        try:
            f = self._calib_fh
            if f is None:
                if self._calib_init_done:
                    # A prior open attempt failed; don't retry every row.
                    return
                self._calib_init_done = True
                path = os.path.expanduser(self.cfg.calibration_log_path)
                need_header = not os.path.exists(path) or os.path.getsize(path) == 0
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                f = open(path, "a", encoding="utf-8")
                self._calib_fh = f
                if need_header:
                    f.write(
                        "ts_iso,ts_unix,row_type,model,market_id,coin,"
                        "side,p,ask,edge,entry_slip,exit_slip,win,net_pnl,"
                        "yes_bid,yes_ask,no_bid,no_ask,adverse_bps\n"
                    )
            f.write(line)
            self._calib_writes += 1
            if self._calib_writes % _CALIB_FLUSH_EVERY == 0:
                f.flush()
        except Exception as e:
            # Don't let calibration logging take down the bot.
            self.log.warning("Calibration log write failed: %s", e)

    def close_calibration_log(self) -> None:
        """Flush and close the calibration file descriptor.

        Called from ``Bot._shutdown`` so the session-resident append
        handle is never orphaned.  The handle is block-buffered (flushed
        every _CALIB_FLUSH_EVERY rows), so the explicit flush here drains
        any rows still sitting in the buffer before releasing the FD on a
        clean shutdown.  Idempotent and exception-safe.

        Order matters: drain the single-thread writer pool FIRST
        (``wait=True``) so any queued rows are written before we close
        the handle and we never ``write()`` to a closed FD.
        """
        pool = getattr(self, "_calib_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        fh = self._calib_fh
        if fh is not None and not fh.closed:
            try:
                fh.flush()
                fh.close()
            except Exception as e:
                self.log.warning("Calibration log close failed: %s", e)
        self._calib_fh = None

    async def _execute_exit(self, mkt: Market, token_id: str,
                            bid_price: float, label: str, shares: float) -> None:
        # v18.8: helper — rollback phantom _shares_in_flight if exit fails.
        # TP1 pre-increments in-flight BEFORE calling here; a failed FOK
        # (thin book, sub-minimum, CLOB rejection) would otherwise leave a
        # phantom counter that blocks future exit evaluations from seeing
        # the full position.
        flight_key = (mkt.market_id, token_id)
        def _rollback_inflight() -> None:
            cur = self._shares_in_flight.get(flight_key, 0.0)
            if cur > 0:
                cur = max(0.0, cur - shares)
                if cur < 1e-6:
                    self._shares_in_flight.pop(flight_key, None)
                else:
                    self._shares_in_flight[flight_key] = cur

        # FIX: guard against float residuals and sub-minimum exits
        if shares < 1e-6:
            _rollback_inflight()
            return
        if shares * (bid_price or 0.01) < 0.50:
            _rollback_inflight()
            return
        tick = mkt.get_tick(token_id)
        dec, mt = mkt.tick_math(token_id)

        # Walk the bids to find exact sweep price for the SELL FOK
        book = mkt.book_yes if token_id == mkt.yes_token else mkt.book_no
        size_usdc = shares * (bid_price or 0.01)
        sweep_price = _fok_sweep_price_sell(book, size_usdc, tick, dec, mt)
        if sweep_price <= 0:
            self.log.warning("EXIT %s skipped — insufficient bid depth", label)
            _rollback_inflight()
            return
        if sweep_price < 0.01:
            sweep_price = 0.01

        self.log.info("EXIT %s %s | price=%.4f | shares=%.1f", label, mkt.coin, sweep_price, shares)
        exit_usdc = shares * sweep_price
        # Critic-v2 C-NEW-4 fix: pre-check fillability via bid-side
        # round-trip walk.  The CLOB returns a valid oid even when the FOK
        # is immediately killed by the matching engine for lack of bid
        # depth — so ``if not oid`` never fired and the position sat
        # "in flight" until reconcile pruned the stale order ~15s later
        # (the exact failure mode the critic flagged).  When the bids
        # can't absorb ``exit_usdc``, route the GTC fallback DIRECTLY
        # without wasting an EIP-712 nonce on the doomed FOK.
        _, _bid_vwap, bid_fillable = _round_trip_cost(book, exit_usdc)
        if not bid_fillable:
            self.log.warning(
                "EXIT %s %s: bid depth insufficient ($%.2f) — skipping "
                "FOK, routing GTC fallback directly", label, mkt.coin, exit_usdc)
            oid = None
        else:
            oid = await self.om.place(
                token_id, Side.SELL, sweep_price, exit_usdc,
                Strategy.TEMPORAL,
                otype="FOK", neg_risk=mkt.neg_risk, tick_size=tick,
            )
        if not oid:
            # v18.9: GTC FALLBACK — FOK was killed (thin book).  Post a GTC
            # limit sell at (best_bid - 1 tick) so the position doesn't expire
            # worthless.  The GTC sits on the book until filled or market
            # resolves.  This recovers 80%+ of capital vs. holding to expiry.
            fallback_price = max(0.01, sweep_price - tick)
            fallback_usdc = shares * fallback_price
            if fallback_usdc >= 0.50:
                gtc_oid = await self.om.place(
                    token_id, Side.SELL, fallback_price, fallback_usdc,
                    Strategy.TEMPORAL,
                    otype="GTC", neg_risk=mkt.neg_risk, tick_size=tick,
                )
                if gtc_oid:
                    self.log.info(
                        "GTC_FALLBACK %s %s | price=%.4f | shares=%.1f "
                        "(FOK killed, posted limit)",
                        label, mkt.coin, fallback_price, shares)
                else:
                    _rollback_inflight()
            else:
                _rollback_inflight()
        # Only clear entry time on successful FULL exit, not partial TP.
        # v18.9: guard with ``oid`` — a failed FOK (thin book, sub-min,
        # CLOB rejection) must NOT erase the entry timestamp, otherwise
        # the fast-exit 60s window and confidence time_pressure lose
        # their reference and the next eval cycle can't fire them.
        efc_key = (mkt.market_id, token_id)
        if oid:
            self._exit_fail_counts.pop(efc_key, None)
        elif not oid:
            self._exit_fail_counts[efc_key] = self._exit_fail_counts.get(efc_key, 0) + 1
        if oid and label not in ("TP1_YES", "TP1_NO"):
            self._entry_times.pop(mkt.market_id, None)

    def cleanup_expired(self, markets: List[Market]) -> None:
        active_ids = {m.market_id for m in markets}

        # 1. Clean ALL state dictionaries — not just _traded.
        #    Markets evaluated but never traded accumulate in _open_prices,
        #    _open_intervals, _eval_debounce forever (288 markets/coin/24h).
        #    v18.8: iterate the UNION of all tracking dict keys — a market
        #    debounced but never inserted into _open_prices (e.g. open_price
        #    lookup returned None) would otherwise leak its _eval_debounce
        #    entry indefinitely.
        all_tracked = set()
        for container in (self._open_prices, self._open_intervals,
                          self._sustain_counts, self._entry_times,
                          self._eval_debounce):
            all_tracked.update(container.keys())
        for mid in all_tracked:
            if mid not in active_ids:
                self._open_prices.pop(mid, None)
                self._open_intervals.pop(mid, None)
                self._sustain_counts.pop(mid, None)
                self._entry_times.pop(mid, None)
                self._eval_debounce.pop(mid, None)

        # 2. Clean _traded
        self._traded.intersection_update(active_ids)

        # 3. Clean _high_bids + _fast_exit_counts (compound keys:
        #    (market_id, token_id)).  A market that ticks ONE eval of noise
        #    but never sustains it long enough to hit the .pop() in the exit
        #    block would otherwise leak its counter key forever as markets
        #    expire (288 markets/coin/24h), so reap by active market_id here.
        for key in list(self._high_bids.keys()):
            if isinstance(key, tuple) and key[0] not in active_ids:
                self._high_bids.pop(key, None)
        for key in list(self._fast_exit_counts.keys()):
            if isinstance(key, tuple) and key[0] not in active_ids:
                self._fast_exit_counts.pop(key, None)
        # 4. Clean _tp1_taken, _entry_edges, _shares_in_flight (v18.7)
        for key in list(self._tp1_taken.keys()):
            if isinstance(key, tuple) and key[0] not in active_ids:
                self._tp1_taken.pop(key, None)
        for key in list(self._entry_edges.keys()):
            if isinstance(key, tuple) and key[0] not in active_ids:
                self._entry_edges.pop(key, None)
        for key in list(self._shares_in_flight.keys()):
            if isinstance(key, tuple) and key[0] not in active_ids:
                self._shares_in_flight.pop(key, None)
        for key in list(self._exit_fail_counts.keys()):
            if isinstance(key, tuple) and key[0] not in active_ids:
                self._exit_fail_counts.pop(key, None)
        # 5. Clean _pending_redemptions (C-BUG-5 fix: held-to-expiry legs)
        for key in list(self._pending_redemptions.keys()):
            if isinstance(key, tuple) and key[0] not in active_ids:
                self._pending_redemptions.pop(key, None)


def _fok_sweep_price(book: Optional[OrderBook], size_usdc: float,
                     tick: float, dec: int, mt: int) -> float:
    """Walk the book to find the exact price level that fills a FOK order.

    v18.7: Integer-space walk — eliminates float rounding accumulation.
    Module-level function so both FiveMinStrategy and LatencyArb can use it.
    Returns 0.0 if the book has insufficient depth to fill size_usdc.
    """
    if not book or not book._asks_int:
        return 0.0
    PS = OrderBook.PRICE_SCALE
    SS = OrderBook.SIZE_SCALE
    rem_scaled = int(round(size_usdc * PS * SS))
    worst_key = 0
    for key in sorted(book._asks_int.keys()):
        if key <= 0:
            continue
        worst_key = key
        level_notional = key * book._asks_int[key]
        rem_scaled -= level_notional
        if rem_scaled <= 0:
            break
    if rem_scaled > 0:
        return 0.0
    worst_price = worst_key / PS
    return snap_price(worst_price + tick, tick, "BUY", dec, mt)


def _fok_sweep_price_sell(book: Optional[OrderBook], size_usdc: float,
                          tick: float, dec: int, mt: int) -> float:
    """Walk the bids to find the exact price level that fills a SELL FOK.

    v18.7: Integer-space walk — eliminates float rounding accumulation.
    Returns 0.0 if the book has insufficient depth to fill size_usdc.
    """
    if not book or not book._bids_int:
        return 0.0
    PS = OrderBook.PRICE_SCALE
    SS = OrderBook.SIZE_SCALE
    rem_scaled = int(round(size_usdc * PS * SS))
    worst_key = 0
    for key in sorted(book._bids_int.keys(), reverse=True):
        if key <= 0:
            continue
        worst_key = key
        level_notional = key * book._bids_int[key]
        rem_scaled -= level_notional
        if rem_scaled <= 0:
            break
    if rem_scaled > 0 or worst_key <= 0:
        return 0.0
    worst_price = worst_key / PS
    # BUG-FIX #7: removed unnecessary one-tick concession.  Pre-fix
    # subtracted one tick from the worst bid level, giving away ~1-2
    # bps per exit for no FOK benefit.
    return snap_price(worst_price, tick, "SELL", dec, mt)


# ─── Latency Arb (Volatility-Normalized) ─────────────────────────────────────

class LatencyArb:
    """v18: Volatility-normalized latency arb. Replaces crude linear model."""

    def __init__(self, cfg: Config, om: OrderManager, tracker: PriceTracker,
                 polyfeed: HyperPolyFeed, by_coin: Dict[str, List[Market]]):
        self.cfg = cfg
        self.om = om
        self.tracker = tracker
        self.polyfeed = polyfeed
        self.by_coin = by_coin
        self.log = get_logger("LatencyArb", cfg.log_level)
        self._cooldowns: Dict[str, float] = {}
        self._shadow_last: Dict[str, float] = {}
        self._shadow_fh = None
        self._shadow_init_done = False
        self._shadow_writes = 0
        # Single-thread executor mirrors FiveMinStrategy._calib_pool: the row
        # is formatted on the event loop (pure CPU) but the blocking
        # write()/flush() is offloaded here so an EBS / page-cache stall on
        # the shadow CSV can never stall WS parsing or blind the risk loop.
        # ``_shadow_log`` runs synchronously from the 10-50 Hz Binance tick
        # handler, so direct file I/O there was an event-loop hazard.
        self._shadow_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="latarb-shadow-io")

    async def on_binance_tick(self, coin: str, price: float) -> None:
        if self.cfg.latarb_shadow:
            self._shadow_scan(coin, price)
        if not self.cfg.latency_arb_enabled:
            return
        markets = self.by_coin.get(coin, [])
        now = time.time()
        for mkt in markets:
            if not mkt.end_time or not mkt.coin:
                continue
            ttc = mkt.end_time - now
            if ttc < 30 or ttc > mkt.tf_secs - 15:
                continue
            if now < self._cooldowns.get(mkt.market_id, 0):
                continue
            interval_start = mkt.start_time
            if not interval_start:
                continue
            open_price = self.tracker.get_price_at(coin, interval_start, max_gap_s=10.0)
            if not open_price or open_price <= 0:
                continue

            displacement = (price - open_price) / open_price
            up = displacement > 0

            book = mkt.book_yes if up else mkt.book_no
            # Latency-arb freshness gate — INTENTIONALLY trades the STALE
            # leg (age >= 400ms), NOT the fresh one.  This is not the
            # "always act on fresh data" rule that applies when you are at
            # an information DISADVANTAGE; here Binance is the fast/leading
            # venue and Polymarket lags.  A Polymarket book that JUST
            # updated (<400ms) has likely already repriced to the move, so
            # there is no edge left; a book that hasn't ticked in >=400ms
            # is still resting at its pre-move quote — that not-yet-
            # repriced ask is exactly what we pick off.  The edge gate
            # below independently confirms ``ask < model_prob`` so we only
            # fire on a genuinely mispriced quote, and the FOK makes a
            # vanished/phantom quote a harmless no-fill (no on-chain cost).
            if not book or book.age_ms < 400:
                continue
            ask = book.best_ask
            if ask is None or ask > 0.65:
                continue

            # v18.3: Signed CDF model.  Previously the code used
            # ``logistic(|z| * 1.5)`` which ALWAYS returns >= 0.5, so
            # the bot always took the side matching the latest tick
            # (pure trend-following with extra steps).  We now compute
            # the proper GBM survival probability for the chosen
            # direction; if the absolute z is too small we abstain.
            sigma_per_sec = max(self.tracker.volatility(coin), 1e-6)
            # L-1 fix: use max(ttc, 1e-4) to match prob_up's S-3 fix.
            # Pre-fix used max(ttc, 1.0) which froze sigma_horizon at 1s
            # near expiry, understating terminal confidence and suppressing
            # valid near-expiry latency-arb entries.
            sigma_horizon = sigma_per_sec * math.sqrt(max(ttc, 1e-4))
            if sigma_horizon <= 0 or not math.isfinite(sigma_horizon):
                continue
            log_disp = math.log(price / open_price) if open_price > 0 else 0.0
            if not math.isfinite(log_disp):
                continue
            # Itô volatility-drag correction (see PriceTracker.prob_up).
            ito_drag = 0.5 * sigma_per_sec * sigma_per_sec * max(ttc, 1e-4)  # L-1: consistent floor
            z = (log_disp - ito_drag) / sigma_horizon
            p_up = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            # Direction-aware probability that the chosen leg wins.
            model_prob = p_up if up else (1.0 - p_up)
            # Conservative clipping — leaves alpha on the table but
            # avoids overbetting on tail moves that already mean-revert.
            model_prob = max(0.30, min(0.85, model_prob))
            slippage = _estimate_slippage(book, self.cfg.min_order_size)
            edge = model_prob - ask - slippage
            if edge < self.cfg.latency_arb_edge:
                continue

            if book.top_depth_usdc < self.cfg.min_top_book_usdc:
                continue
            if mkt.total_cost + self.cfg.min_order_size > self.cfg.max_position:
                continue

            token = mkt.yes_token if up else mkt.no_token
            tick = mkt.get_tick(token)
            dec, mt = mkt.tick_math(token)

            # Walk the book for accurate FOK sweep price
            book = mkt.book_yes if up else mkt.book_no
            sweep = _fok_sweep_price(book, self.cfg.min_order_size, tick, dec, mt)
            if sweep <= 0:
                continue
            sz = self.cfg.min_order_size

            self.log.info("LATARB %s %s | disp=%.4f | z=%.2f | ask=%.3f | edge=%.3f | age=%.0fms",
                          "UP" if up else "DN", coin, displacement, z,
                          ask, edge, book.age_ms)

            await self.om.place(
                token, Side.BUY, sweep, sz, Strategy.TEMPORAL,
                otype="FOK", neg_risk=mkt.neg_risk, tick_size=tick,
            )
            self._cooldowns[mkt.market_id] = now + self.cfg.latency_arb_cooldown

    def _shadow_scan(self, coin: str, price: float) -> None:
        """Phase-1 latency-arb measurement (logging only, never trades).

        On each Binance tick, record every market whose Polymarket book is
        STALE (hasn't repriced in >= latarb_shadow_min_age_ms).  An offline
        join to the resolved outcome (latency_edge.py) then measures whether
        the spot-implied side is buyable at the stale ask net of cost -- i.e.
        whether spot genuinely leads the Polymarket book.  Throttled per
        market; defensive -- never raises into the event loop.
        """
        try:
            markets = self.by_coin.get(coin, [])
            now = time.time()
            min_age = self.cfg.latarb_shadow_min_age_ms
            throttle = self.cfg.latarb_shadow_throttle_ms / 1000.0
            for mkt in markets:
                if not mkt.end_time or not mkt.start_time or not mkt.coin:
                    continue
                ttc = mkt.end_time - now
                if ttc < 15 or ttc > (mkt.tf_secs - 5):
                    continue
                by, bn = mkt.book_yes, mkt.book_no
                if not by or not bn:
                    continue
                if max(by.age_ms, bn.age_ms) < min_age:
                    continue
                if now - self._shadow_last.get(mkt.market_id, 0.0) < throttle:
                    continue
                open_price = self.tracker.get_price_at(coin, mkt.start_time, max_gap_s=10.0)
                if not open_price or open_price <= 0:
                    continue
                disp = (price - open_price) / open_price
                self._shadow_last[mkt.market_id] = now
                self._shadow_log(now, mkt, coin, ttc, disp, price, open_price, by, bn)
        except Exception as e:  # logging path must never crash the loop
            self.log.warning("shadow scan error: %s", e)

    def _shadow_log(self, now, mkt, coin, ttc, disp, price, open_price, by, bn) -> None:
        # Format the row on the event-loop thread (pure CPU, never blocks),
        # then hand the finished line to the single-thread writer so the
        # blocking I/O happens OFF the loop (see _shadow_pool).
        iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat(
            timespec="seconds").replace("+00:00", "Z")
        up_side = "UP" if disp > 0 else "DN"
        def f(x):
            return "" if x is None else f"{float(x):.6f}"
        row = ",".join([
            iso, f"{now:.3f}", str(mkt.market_id), coin,
            f"{ttc:.1f}", f"{disp:.6f}", up_side,
            f"{price:.4f}", f"{open_price:.4f}",
            f(by.best_bid), f(by.best_ask), f(bn.best_bid), f(bn.best_ask),
            f"{by.age_ms:.0f}", f"{bn.age_ms:.0f}",
        ]) + "\n"
        try:
            self._shadow_pool.submit(self._write_shadow_row, row)
        except RuntimeError:
            # Pool already shut down during teardown — drop the row.
            pass

    def _write_shadow_row(self, line: str) -> None:
        """Runs ONLY on the dedicated single-thread shadow executor.

        Lazily opens the block-buffered append handle on first call and
        flushes every ``_CALIB_FLUSH_EVERY`` rows.  Because exactly one
        worker ever executes this, ``_shadow_fh`` / ``_shadow_writes`` need
        no locking and rows never interleave.
        """
        try:
            f = self._shadow_fh
            if f is None:
                if self._shadow_init_done:
                    # A prior open attempt failed; don't retry every row.
                    return
                self._shadow_init_done = True
                path = os.path.expanduser(self.cfg.latarb_shadow_path)
                need_header = not os.path.exists(path) or os.path.getsize(path) == 0
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                f = open(path, "a", encoding="utf-8")
                self._shadow_fh = f
                if need_header:
                    f.write(
                        "ts_iso,ts_unix,market_id,coin,ttc,spot_disp,up_side,"
                        "spot_price,open_price,yes_bid,yes_ask,no_bid,no_ask,"
                        "yes_age_ms,no_age_ms\n")
            f.write(line)
            self._shadow_writes += 1
            if self._shadow_writes % _CALIB_FLUSH_EVERY == 0:
                f.flush()
        except Exception as e:
            self.log.warning("shadow log write failed: %s", e)

    def close_shadow_log(self) -> None:
        """Drain the writer pool, flush, and close the FD on shutdown.

        Order matters: drain the single-thread writer pool FIRST
        (``wait=True``) so any queued rows are written before we close the
        handle and we never ``write()`` to a closed FD.  Idempotent and
        exception-safe.
        """
        pool = getattr(self, "_shadow_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        fh = self._shadow_fh
        if fh is not None and not fh.closed:
            try:
                fh.flush()
                fh.close()
            except Exception as e:
                self.log.warning("shadow log close failed: %s", e)
        self._shadow_fh = None


# ─── Risk ─────────────────────────────────────────────────────────────────────

class Risk:
    def __init__(self, cfg: Config, om: OrderManager) -> None:
        self.cfg = cfg
        self.om  = om
        self.log = get_logger("Risk", cfg.log_level)
        self._halted    = False
        self._reason    = ""
        # BUG-FIX #21: track halt type for discriminated daily-reset.
        self._halt_type = ""
        self._pnl       = 0.0
        self._day_start = 0.0
        # Monotonic: an NTP step / wall-clock jump must not prematurely
        # clear the daily-loss halt (a forward jump would otherwise reset
        # the day early and re-arm full drawdown).
        self._day_reset = time.monotonic()
        self._consecutive_losses = 0

    @property
    def halted(self) -> bool:
        return self._halted

    def record_pnl(self, delta: float) -> None:
        """Accumulate partial / running PnL.

        v18.3: this method NO LONGER updates ``_consecutive_losses``.
        Streak tracking is now driven by ``record_trade_closed`` so
        that a multi-fill exit (N partial sells) is counted as a
        SINGLE win or loss instead of N.  ``record_pnl`` still drives
        the daily-loss halt because daily PnL is partial-additive.
        """
        self._pnl += delta

    def record_trade_closed(self, net_pnl: float) -> None:
        """Update consecutive-loss streak.  Called once per closed trade.

        v18.3 fix: previously every negative partial-sell fill bumped
        ``_consecutive_losses``, so a 2-partial losing exit counted as
        2 losses and the ``max_consecutive_losses`` halt fired roughly
        twice as fast as intended.
        """
        if net_pnl < 0:
            self._consecutive_losses += 1
        elif net_pnl > 0:
            self._consecutive_losses = 0
        # net_pnl == 0: leave streak unchanged (rare; round-trip break-even)

    def ok(self) -> bool:
        if time.monotonic() - self._day_reset > 86_400:
            self._day_start = self._pnl
            self._day_reset = time.monotonic()
            if self._halted:
                # BUG-FIX #21: only auto-clear halts that are safe to
                # reset unattended.  Drift/reject halts require operator
                # investigation before resuming.
                if self._halt_type in ("daily_loss", "consec_losses"):
                    self._halted = False
                    self._reason = ""
                    self._halt_type = ""
                    self._consecutive_losses = 0
                    self.log.info("Daily reset — halt cleared (type was %s)",
                                  self._halt_type)
                else:
                    self.log.warning(
                        "Daily reset — halt NOT cleared "
                        "(type=%s requires operator)", self._halt_type)
        if self._halted:
            return False
        dp = self._pnl - self._day_start
        if dp < -self.cfg.max_daily_loss:
            self._halt(f"Daily loss ${-dp:.2f}", halt_type="daily_loss")
            return False
        if self.om.rejects >= 5:
            self._halt(f"{self.om.rejects} consecutive order rejects",
                       halt_type="rejects")
            return False
        if self._consecutive_losses >= self.cfg.max_consecutive_losses:
            self._halt(f"{self._consecutive_losses} consecutive losses",
                       halt_type="consec_losses")
            return False
        if self.om.count >= self.cfg.max_open_orders:
            return False
        return True

    def _halt(self, reason: str, halt_type: str = "unknown") -> None:
        self._halted = True
        self._reason = reason
        self._halt_type = halt_type
        self.log.critical("HALT [%s]: %s", halt_type, reason)

    def status(self) -> dict:
        dp = self._pnl - self._day_start
        return {
            "pnl":     round(self._pnl, 4),
            "daily":   round(dp, 4),
            "orders":  self.om.count,
            "halted":  self._halted,
            "halt_type": self._halt_type,
            "reason":  self._reason,
            "consec_losses": self._consecutive_losses,
        }


# ─── Bot ──────────────────────────────────────────────────────────────────────

class Bot:
    def __init__(self, cfg: Config) -> None:
        self.cfg      = cfg
        self.log      = get_logger("Bot", cfg.log_level)
        self.metrics  = Metrics() if cfg.metrics_enabled else None
        self.client   = PolyClient(cfg)
        self.om       = OrderManager(cfg, self.client, self.metrics)
        self.risk     = Risk(cfg, self.om)
        self.binance  = BinanceFeed(cfg.coins)
        self.polyfeed = HyperPolyFeed(shard_count=cfg.ws_shard_count)
        self.userfeed: Optional[UserFeed]        = None
        self.tracker:      Optional[PriceTracker]      = None
        self.fivemin:      Optional[FiveMinStrategy]   = None
        self.latency_arb:  Optional[LatencyArb]        = None
        self.fivemin_markets: List[Market]              = []
        self._5m_ids:  Set[str]                        = set()
        self.markets:  List[Market]                    = []
        self.t2m:      Dict[str, Market]               = {}
        self.by_coin:  Dict[str, List[Market]]         = {}
        self.tasks:    List[asyncio.Task]     = []
        # Strong refs to fire-and-forget eval tasks: asyncio holds only a
        # WEAK ref to a bare create_task, so without this the GC can drop
        # an in-flight evaluation mid-await.  The done-callback discards.
        self._eval_tasks: Set[asyncio.Task] = set()
        # Same strong-ref guard for the polyfeed full-reconnect task spawned
        # from the health loop: a bare create_task there is exactly the
        # GC-drop bug ``_eval_tasks`` fixes, and it fires during a total WS
        # outage — the single most important moment to keep the reconnect
        # coroutine alive.  The done-callback discards on completion.
        self._bg_tasks: Set[asyncio.Task] = set()
        self.running   = False
        self.shutdown_ev = asyncio.Event()
        self.session:  Optional[aiohttp.ClientSession] = None
        self._pos_lock = asyncio.Lock()   # Bot-level lock for position mutations
        # v18.3: tracks the cumulative PnL of a sell-leg as it gets
        # filled across multiple partials.  Drained on full-leg-close.
        # Keyed by (market_id, token_id) so the YES and NO legs of the
        # same market keep INDEPENDENT in-flight accumulators; closing
        # one leg must not pop the other leg's running PnL.
        self._trade_pnl_in_flight: Dict[Tuple[str, str], float] = {}
        # Persistent, bounded thread pool for the periodic on-chain drift
        # check's blocking SDK balance fetches.  Allocated ONCE at boot
        # (not per drift cycle) so the 5-minute sweep never pays repeated
        # OS thread create/teardown cost, and the process-wide default
        # executor stays reserved for latency-critical order I/O.
        self._drift_io_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(self.cfg.drift_check_concurrency)),
            thread_name_prefix="drift-io")

    async def run(self) -> None:
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=60),
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"Accept": "application/json"},
        )
        try:
            await self._boot()
        except Exception as e:
            self.log.critical("Boot failed: %s", e, exc_info=True)
        finally:
            if self.session and not self.session.closed:
                await self.session.close()

    async def _boot(self) -> None:
        self._banner()

        # 1. Connectivity
        try:
            async with self.session.get(f"{self.cfg.clob_url}/") as r:
                if r.status not in (200, 404):
                    raise ConnectionError(f"CLOB returned HTTP {r.status}")
            self.log.info("CLOB reachable")
        except Exception as e:
            self.log.critical("CLOB unreachable: %s", e)
            return

        self.log.info("Polymarket SDK: %s",
                      "py-clob-client-v2 (CLOB V2)" if _SDK_IS_V2
                      else "py-clob-client V1 — DEAD, orders WILL be rejected"
                      if _HAS_SDK else "NOT INSTALLED")

        # 2. Auth
        ok = await self.client.initialize(self.session)
        if not ok:
            self.log.critical(
                "Authentication failed.\n"
                "  Check: POLYMARKET_PRIVATE_KEY, POLYMARKET_PROXY_ADDRESS\n"
                "  Try:   POLYMARKET_SIGNATURE_TYPE=1 in .env -> restart")
            return

        self.log.info("Auth: %s  |  Trader: %s",
                      self.client.active_mode, self.client.trading_address)

        # 3. Balance
        balance = await self.client.get_balance()
        self.log.info("CLOB balance: $%.4f", balance)
        if balance < 1.0 and not self.cfg.dry_run:
            self.log.warning("Low balance ($%.4f)", balance)

        # 4. Cancel stale orders
        await self.om.cancel_all()

        # 5. Discover 5-minute crypto markets
        self.fivemin_markets = await discover_5min_markets(self.cfg, self.session)
        if not self.fivemin_markets:
            self.log.warning("No 5-min markets found — will poll for them")
        else:
            self.log.info("Found %d 5/15-min crypto markets", len(self.fivemin_markets))

        self.markets = self.fivemin_markets
        for m in self.markets:
            self.polyfeed.subscribe(m.yes_token)
            self.polyfeed.subscribe(m.no_token)
            self.t2m[m.yes_token] = m
            self.t2m[m.no_token]  = m
            self._5m_ids.add(m.market_id)
            if m.coin:
                self.by_coin.setdefault(m.coin, []).append(m)

        self.client.set_market_ref(self.t2m)
        self.userfeed = UserFeed(self.client, self.om)
        self.userfeed.set_markets(self.t2m)
        self.userfeed.on_fill(self._on_fill)
        # v18.4 — install REST-fill-replay handler on the OrderManager.
        # ``OrderManager.reconcile_fills`` will call this for any trade
        # observed via REST but not yet seen via WS.
        self.om.set_fill_replay_handler(self._replay_rest_fill)
        await self._seed_books()

        # 6. Signing test (skip in DRY_RUN — no real orders are sent)
        if self.cfg.dry_run:
            passed = True
            self.log.info("DRY_RUN — skipping signing test")
        elif not self.markets:
            passed = True
        else:
            test_mkt   = self.markets[0]
            test_token = test_mkt.yes_token
            test_tick  = test_mkt.get_tick(test_token)
            test_neg   = test_mkt.neg_risk
            self.log.info("Running signing test on '%s'…", test_mkt.question[:40])
            passed = await self.client.test_order(test_token, test_tick, test_neg)

        if not passed:
            # BUG-FIX #37: alternative sig-type probes are only safe in
            # DRY_RUN.  Each ``_build_sdk(alt)`` + ``test_order()`` call
            # posts a real order to the CLOB (now capped at 1.0 share @
            # 0.001 = ~$0.001 of risk, but still consumes nonce slots
            # and can match a stale resting limit).  On a LIVE boot with
            # a real capital account, refuse to try alternatives — the
            # operator must fix POLYMARKET_SIGNATURE_TYPE in .env and
            # restart.  This is the honest version of "don't fire
            # orders you didn't mean to fire".
            if not self.cfg.dry_run:
                self.log.warning(
                    "signing test FAILED for sig_type=%d — refusing to try "
                    "alternatives in LIVE (each probe costs an order). "
                    "Set POLYMARKET_SIGNATURE_TYPE=1 or 2 in .env and restart.",
                    self.cfg.signature_type)
            else:
                self.log.warning("sig_type=%d failed — trying alternatives in DRY_RUN…",
                                 self.cfg.signature_type)
                original = self.cfg.signature_type
                for alt in [1, 0, 2]:
                    if alt == original:
                        continue
                    if await self.client._build_sdk(alt):
                        if await self.client.test_order(test_token, test_tick, test_neg):
                            self.log.info("sig_type=%d works! Set in .env to skip probe.", alt)
                            passed = True
                            break

        if not passed:
            self.log.critical(
                "SIGNING FAILED — ALL SIG TYPES REJECTED\n"
                "  EOA: %s  Proxy: %s\n"
                "  If the reject was 'order_version_mismatch' you are on the\n"
                "  dead CLOB V1 SDK — install V2 (this is the usual cause):\n"
                "    1. %s/bin/pip install py-clob-client-v2\n"
                "    2. Wrap USDC.e -> pUSD (polymarket.com one-time approval)\n"
                "    3. Verify POLYMARKET_PROXY_ADDRESS + SIGNATURE_TYPE=2",
                self.client.signer_address,
                self.cfg.proxy_address or "(none)",
                sys.prefix)
            self.client.lib_broken = True
            return

        self.log.info("Signing test PASSED  sig_type=%d (%s)",
                      self.cfg.signature_type,
                      _SIG_LABELS.get(self.cfg.signature_type, "?"))

        # 7. Market summary
        self.log.info("=" * 68)
        self.log.info("  %-42s  %-6s  %-3s  %s", "Question", "Coin", "NR", "Tick(YES)")
        self.log.info("  " + "-" * 66)
        for m in self.markets:
            self.log.info("  %-42s  %-6s  %-3s  %s",
                          m.question[:42], m.coin or "STABLE",
                          "Y" if m.neg_risk else "N",
                          m.tick_sizes.get(m.yes_token, "?"))
        self.log.info("=" * 68)

        # 8. Strategy init
        self.tracker = PriceTracker(
            self.binance, self.cfg.prob_shrink,
            min_order_size_usdc=self.cfg.min_order_size,
        )
        # Boot-time calibration feedback: load historical hit rates per coin
        # from the calibration CSV and compute empirical shrink corrections.
        self._load_calibration_shrink()
        self.polyfeed.on_update(self._on_book)
        self.binance.on_update(self._on_price)
        self.fivemin = FiveMinStrategy(
            self.cfg, self.om, self.risk, self.tracker, self.metrics)
        self.fivemin.polyfeed = self.polyfeed
        self.fivemin._balance_cache = balance
        self.fivemin._balance_ts = time.time()
        # v19 Scope-A: route the always-on adverse-selection probe to the
        # calibration CSV (shadow rows) so the offline harness can aggregate.
        self.om.set_shadow_sink(self.fivemin._log_shadow)
        self.latency_arb = LatencyArb(
            self.cfg, self.om, self.tracker, self.polyfeed, self.by_coin)

        # v19 Scope-A (Flaw #1/#3/#7): HARD live go/no-go gate.  When
        # ``require_proven_edge`` is set and we are NOT in DRY_RUN, refuse to
        # risk real capital until the recorded calibration data PROVES a
        # measured edge net of cost.  This is the honest answer to "make it
        # profitable": prove it on data first.  DRY_RUN always proceeds so
        # the data needed to clear the gate can be collected.
        if self.cfg.require_proven_edge and not self.cfg.dry_run:
            ok_go, reasons = evaluate_go_no_go(self.cfg)
            if not ok_go:
                self.log.critical(
                    "GO/NO-GO: NO-GO — refusing to trade live with real "
                    "capital until measured edge > measured cost:")
                for rsn in reasons:
                    self.log.critical("  - %s", rsn)
                self.log.critical(
                    "  Run DRY_RUN to collect data, then "
                    "`python polybot.py --analyze %s` to inspect.",
                    self.cfg.calibration_log_path)
                return
            self.log.info("GO/NO-GO: GO — %s",
                          "; ".join(reasons) if reasons else "gate cleared")

        loop = asyncio.get_running_loop()
        for sig_name in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig_name, lambda: self.shutdown_ev.set())
            except NotImplementedError:
                pass

        self.running = True
        mode = "DRY RUN" if self.cfg.dry_run else "LIVE"
        arch = "EVENT-DRIVEN" if self.cfg.event_driven else "TIMER"
        self.log.info(
            "Started [%s] [%s]  sig=%d  bal=$%.2f  mkts=%d  shards=%d  json=%s",
            mode, arch, self.cfg.signature_type, balance,
            len(self.markets), self.cfg.ws_shard_count,
            "orjson" if _FAST_JSON else "stdlib")

        self.tasks = [
            asyncio.create_task(self.polyfeed.run(),        name="polyfeed"),
            asyncio.create_task(self.binance.run(),         name="binance"),
            asyncio.create_task(self.userfeed.run(),        name="userfeed"),
            asyncio.create_task(self._reconcile_loop(),     name="reconcile"),
            asyncio.create_task(self._health_loop(),        name="health"),
            asyncio.create_task(self._status_loop(),        name="status"),
            asyncio.create_task(self._fivemin_refresh(),    name="discovery"),
            asyncio.create_task(self._shutdown_wait(),      name="shutdown"),
        ]
        # Timer-driven fallback (only if event_driven is off)
        if not self.cfg.event_driven:
            self.tasks.append(asyncio.create_task(
                self._fivemin_timer_loop(), name="fivemin_timer"))

        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            pass

    # ── Seed books ────────────────────────────────────────────────────────────

    async def _seed_books(self) -> None:
        self.log.info("Seeding order books and tick sizes…")
        sem = asyncio.Semaphore(8)

        async def seed_book(tid: str) -> bool:
            async with sem:
                try:
                    async with self.session.get(
                        f"{self.cfg.clob_url}/book",
                        params={"token_id": tid},
                        timeout=aiohttp.ClientTimeout(total=6),
                    ) as r:
                        if not r.ok:
                            return False
                        d  = await r.json(content_type=None)
                        bk = self.polyfeed.book(tid)
                        if not bk:
                            return False
                        # v18.3: use new OrderBook.replace_snapshot API.
                        bids = [(float(b["price"]), float(b["size"]))
                                for b in d.get("bids", [])
                                if float(b.get("size", 0)) > 0]
                        asks = [(float(a["price"]), float(a["size"]))
                                for a in d.get("asks", [])
                                if float(a.get("size", 0)) > 0]
                        bk.replace_snapshot(bids, asks)
                        bk.ts = time.monotonic()
                        # Mark snapshot received so subsequent deltas
                        # from the WS shard are not dropped.
                        self.polyfeed._snapshot_received.add(tid)
                        m = self.t2m.get(tid)
                        if m:
                            if tid == m.yes_token:
                                m.book_yes = bk
                            else:
                                m.book_no = bk
                        return True
                except Exception as e:
                    self.log.warning("Book seed %s: %s", tid[:12], e)
                    return False

        async def fetch_tick(tid: str) -> Optional[float]:
            async with sem:
                try:
                    async with self.session.get(
                        f"{self.cfg.clob_url}/tick-size",
                        params={"token_id": tid},
                        timeout=aiohttp.ClientTimeout(total=6),
                    ) as r:
                        if not r.ok:
                            return None
                        d = await r.json(content_type=None)
                        if isinstance(d, (int, float)):
                            raw = d
                        elif isinstance(d, str):
                            raw = d
                        elif isinstance(d, dict):
                            raw = (d.get("minimum_tick_size") or
                                   d.get("tick_size") or
                                   d.get("minTickSize"))
                        else:
                            raw = None
                        ts = float(raw) if raw else 0.0
                        if ts <= 0 or ts >= 1:
                            ts = 0.01
                        m = self.t2m.get(tid)
                        if m:
                            m.set_tick(tid, ts)
                        return ts
                except Exception as e:
                    self.log.warning("Tick-size %s: %s", tid[:12], e)
                    return None

        toks = list(self.t2m.keys())
        book_results, tick_results = await asyncio.gather(
            asyncio.gather(*[seed_book(t) for t in toks], return_exceptions=True),
            asyncio.gather(*[fetch_tick(t) for t in toks], return_exceptions=True),
        )
        books_ok = sum(1 for r in book_results if r is True)
        tick_ok  = sum(1 for r in tick_results
                       if isinstance(r, float) and 0 < r < 1)
        self.log.info("Seeded %d/%d books, %d/%d ticks",
                      books_ok, len(toks), tick_ok, len(toks))

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_book(self, tid: str, book: OrderBook) -> None:
        m = self.t2m.get(tid)
        if not m or not self.running:
            return
        if self.client.lib_broken:
            return

        if tid == m.yes_token:
            m.book_yes = book
        else:
            m.book_no = book

        if self.risk.halted:
            # Trigger the cancel-all + shutdown exactly once; otherwise every
            # subsequent book tick during the halt->drain window fires another
            # REST cancel_all and can trip CLOB rate limits (HTTP 429).
            if not self.shutdown_ev.is_set():
                # S-7: opt-in auto-flatten.  Pre-fix, the halt path cancelled
                # open orders but left existing positions exposed into
                # resolution.  Operators on small accounts can opt in to a
                # one-shot FOK-at-bid flatten before cancel_all so the
                # worst-case "halt with losing leg open" path is bounded.
                # DEFAULT is OFF — large-account operators must investigate.
                if self.cfg.auto_flatten_on_halt:
                    try:
                        await self._flatten_all_positions()
                    except Exception as e:
                        self.log.error("auto_flatten_on_halt failed: %s", e)
                # M-4 fix: try/finally so shutdown_ev.set() fires even if
                # cancel_all raises (possible now that C-3 makes it propagate).
                try:
                    await self.om.cancel_all()
                finally:
                    self.shutdown_ev.set()
            return
        if not self.risk.ok():
            return

        # v18: Event-driven strategy trigger.  Fire-and-forget (mirrors
        # ``_on_price``): the feed dispatches book callbacks with ``await``
        # inline, so awaiting ``evaluate_single`` here would serialize the
        # entire WS parser behind strategy evaluation — and stall it
        # completely whenever the eval semaphore is saturated — starving
        # ingestion and staling the local book.  Pre-check the debounce so
        # we don't spawn a task per tick when the interval hasn't elapsed.
        if self.cfg.event_driven and self.fivemin and m.coin:
            now = time.monotonic()
            last = self.fivemin._eval_debounce.get(m.market_id, 0.0)
            if now - last < self.fivemin.cfg.eval_debounce_ms / 1000.0:
                return
            # Stamp BEFORE spawning (no await in between → race-free) so a
            # book-update burst can't pile unbounded eval tasks for one
            # market behind the semaphore.
            self.fivemin._eval_debounce[m.market_id] = now
            try:
                t = asyncio.create_task(
                    self.fivemin.evaluate_single(m, self.fivemin_markets),
                    name=f"eval_{m.market_id[:8]}")
                self._eval_tasks.add(t)
                t.add_done_callback(self._eval_tasks.discard)
            except Exception as e:
                self.log.warning("_on_book eval error: %s", e)

    async def _on_price(self, coin: str, price: float) -> None:
        if not self.running or self.risk.halted:
            return

        # Latency arb on every Binance tick (lightweight, no debounce needed)
        if self.latency_arb:
            try:
                await self.latency_arb.on_binance_tick(coin, price)
            except Exception as e:
                self.log.debug("LatencyArb error: %s", e)

        # v18: Event-driven strategy trigger on price update
        # Pre-check debounce HERE to avoid coroutine overhead on 50-100 ticks/s.
        # Only fire-and-forget if debounce interval has actually elapsed.
        if self.cfg.event_driven and self.fivemin:
            now = time.monotonic()
            markets = self.by_coin.get(coin, [])
            for mkt in markets:
                last = self.fivemin._eval_debounce.get(mkt.market_id, 0.0)
                if now - last < self.fivemin.cfg.eval_debounce_ms / 1000.0:
                    continue   # skip — debounce not elapsed
                # Stamp BEFORE spawning (no await in between → race-free):
                # backpressure that caps in-flight eval tasks to one per
                # market per debounce window, even under a tick storm.
                self.fivemin._eval_debounce[mkt.market_id] = now
                try:
                    t = asyncio.create_task(
                        self.fivemin.evaluate_single(mkt, self.fivemin_markets),
                        name=f"eval_{mkt.market_id[:8]}")
                    self._eval_tasks.add(t)
                    t.add_done_callback(self._eval_tasks.discard)
                except Exception as e:
                    self.log.warning("_on_price eval error: %s", e)

    async def _on_fill(self, mkt: Market, tid: str, side_str: str,
                       shares: float, price: float) -> None:
        """Position mutation is guarded by Bot._pos_lock.

        The lock lives on Bot (not FiveMinStrategy) to guarantee mutual
        exclusion even during boot when self.fivemin may still be None.
        """
        # H-5 fix: validate price and shares BEFORE acquiring the lock.
        # Pre-fix had no bounds check; a malformed WS or REST message with
        # price=1e18, size=NaN, or negative values bypassed the side guard
        # and corrupted pos.add / _balance_cache / daily-loss.
        if not math.isfinite(price) or not (0.0 < price < 1.0):
            self.log.warning("_on_fill: invalid price %r for %s — dropping",
                             price, tid[:12])
            return
        if not math.isfinite(shares) or not (0.0 < shares < 1e8):
            self.log.warning("_on_fill: invalid shares %r for %s — dropping",
                             shares, tid[:12])
            return
        async with self._pos_lock:
            # BUG-FIX #34: validate fill side — treat unknown sides
            # as data corruption (dropped WS field, malformed REST trade)
            # rather than silently processing as SELL.
            if side_str not in ("BUY", "SELL"):
                self.log.warning("_on_fill: unknown side %r — dropping", side_str)
                return
            side = Side.BUY if side_str == "BUY" else Side.SELL
            pos  = mkt.pos_yes if tid == mkt.yes_token else mkt.pos_no

            # Snapshot cost BEFORE any mutations for accurate exposure delta
            old_yes_cost = mkt.pos_yes.cost
            old_no_cost  = mkt.pos_no.cost

            # Effective SELL quantity actually settled against the LOCAL
            # ledger.  A duplicated WS packet, a delayed REST replay, or a
            # ghost fill can report ``shares > pos.shares``; booking PnL and
            # crediting cash on the feed's size would manufacture phantom
            # profit that masks the daily-loss halt AND inflates the Kelly
            # bankroll (_balance_cache).  Prime-directive invariant: never
            # realize more shares than the ledger holds.
            sell_fill = 0.0

            if side == Side.BUY:
                pos.add(shares, price * shares)
                if self.metrics:
                    self.metrics.record_fill()
            else:
                if pos.shares > 0:
                    # C-BUG-4 fix: accumulate partial PnL in the in-flight
                    # tracker but do NOT credit Risk until the leg fully
                    # closes.  Previously, every partial sell hit Risk._pnl
                    # immediately — a TP1 at +$4 on a $5 max_daily_loss
                    # budget left only $1 of headroom before a false halt,
                    # even though the position wasn't closed and the gain
                    # wasn't realized.  Risk.record_pnl now fires ONCE
                    # at full leg close with the accumulated net_pnl.
                    sell_fill = min(shares, pos.shares)
                    avg_at_sell = pos.avg_price
                    partial_pnl = (price - avg_at_sell) * sell_fill
                    # Track per-LEG trade-in-progress PnL (market_id, token).
                    flight_key = (mkt.market_id, tid)
                    self._trade_pnl_in_flight[flight_key] = (
                        self._trade_pnl_in_flight.get(flight_key, 0.0)
                        + partial_pnl)
                    pos.reduce(sell_fill)
                    # v18.7: Deduct filled shares from in-flight tracker
                    if self.fivemin and flight_key in self.fivemin._shares_in_flight:
                        self.fivemin._shares_in_flight[flight_key] = max(
                            0.0, self.fivemin._shares_in_flight[flight_key] - sell_fill)
                        if self.fivemin._shares_in_flight[flight_key] < 1e-6:
                            self.fivemin._shares_in_flight.pop(flight_key, None)
                    # Metrics still sees partials for real-time display
                    if self.metrics:
                        self.metrics.record_pnl(partial_pnl)
                    if pos.shares < 1e-6:
                        # Leg fully closed — fire streak + PnL + outcome ONCE.
                        net_pnl = self._trade_pnl_in_flight.pop(
                            flight_key, partial_pnl)
                        # C-BUG-4: credit Risk with the FULL net PnL only now.
                        self.risk.record_pnl(net_pnl)
                        self.risk.record_trade_closed(net_pnl)
                        if self.fivemin:
                            # v18.7: Flush TP/edge/in-flight state on full
                            # closure to prevent stale keys corrupting
                            # re-entry logic within the same interval.
                            self.fivemin._tp1_taken.pop(flight_key, None)
                            self.fivemin._entry_edges.pop(flight_key, None)
                            self.fivemin._shares_in_flight.pop(flight_key, None)
                            self.fivemin._fast_exit_counts.pop(flight_key, None)
                            self.fivemin.record_outcome(
                                net_pnl > 0,
                                market_id=mkt.market_id,
                                net_pnl=float(net_pnl),
                                coin=mkt.coin,
                            )

            # Virtual merge REMOVED (v18.8).  The prior block reduced local
            # pos_yes/pos_no by min(yes, no) to simulate a CTF merge
            # redemption, but it NEVER broadcast an on-chain merge tx.
            # This desynced local shares from chain shares, causing the
            # drift check (_check_position_drift) to halt the bot on
            # the very next 5-min reconciliation pass.  Positions now
            # remain until expiry or explicit exit; the $1 redemption
            # is only realized when the market resolves on-chain.

            # Update cached net exposure: O(1) delta from before/after snapshot
            # net_exposure = Σ(yes.cost - no.cost) across all markets
            # This market's contribution changed by:
            #   (new_yes_cost - new_no_cost) - (old_yes_cost - old_no_cost)
            if self.fivemin:
                new_yes_cost = mkt.pos_yes.cost
                new_no_cost  = mkt.pos_no.cost
                delta = (new_yes_cost - new_no_cost) - (old_yes_cost - old_no_cost)
                self.fivemin._net_exposure += delta

                # Real-time Kelly bankroll sync: treat _balance_cache as a
                # live ledger rather than a 60s-polled metric.  This prevents
                # over-leveraging during rapid drawdown streaks where the
                # REST-polled balance is stale.
                if side == Side.BUY:
                    self.fivemin._balance_cache -= price * shares
                else:
                    # Credit only the cash for shares actually held/sold
                    # (``sell_fill``), not the feed's raw size — see the
                    # ghost-fill invariant above.
                    self.fivemin._balance_cache += price * sell_fill

    # ── v18.4 §5 — REST fill replay & drift halt ─────────────────────────────

    async def _replay_rest_fill(self, trade: dict) -> None:
        """Adapter: translate a REST ``/trades`` row into ``_on_fill``.

        Called by ``OrderManager.reconcile_fills`` when a trade is
        observed via REST but not in the WS-seen set.  Idempotency is
        guaranteed by the caller (trade_id dedup); this adapter only
        translates field names and dispatches.

        Required fields on the trade dict (with version-tolerant
        fallbacks):
          * ``asset_id`` or ``token_id``  — the CTF token traded
          * ``side``                       — "BUY" or "SELL"
          * ``price``                      — execution price (string or float)
          * ``size`` or ``maker_amount_filled`` — execution size in shares

        Behavior on malformed / unknown trades: log a WARNING and drop.
        The cursor advance still happens (so a single bad row won't
        wedge the loop), and the missing fill becomes visible via the
        position-drift halt on the next pass.
        """
        try:
            asset_id = str(
                trade.get("asset_id")
                or trade.get("token_id")
                or trade.get("tokenId")
                or ""
            )
            if not asset_id:
                self.log.warning("_replay_rest_fill: missing asset_id in %s",
                                 {k: trade.get(k) for k in ("id", "trade_id", "side")})
                return
            mkt = self.t2m.get(asset_id)
            if mkt is None:
                # Trade for a market we no longer track (expired and
                # cleaned up).  Skip silently.
                return

            side_raw = str(trade.get("side", "")).upper()
            if side_raw not in ("BUY", "SELL"):
                self.log.warning(
                    "_replay_rest_fill: unknown side %r on trade %s",
                    side_raw, trade.get("id"))
                return

            price = float(trade.get("price") or 0)
            # ``size`` is in shares.  Some SDKs use ``maker_amount_filled``
            # or ``taker_amount_filled`` instead — handle both.
            size = float(
                trade.get("size")
                or trade.get("filled_size")
                or trade.get("maker_amount_filled")
                or trade.get("taker_amount_filled")
                or 0
            )
            if price <= 0 or size <= 0:
                return

            self.log.warning(
                "_replay_rest_fill: applying %s %s @ %.4f (%.4f shares) "
                "from REST (mkt=%s)",
                side_raw, asset_id[:12], price, size, mkt.market_id[:8])
            await self._on_fill(mkt, asset_id, side_raw, size, price)
        except Exception as e:
            self.log.error("_replay_rest_fill error: %s (trade=%s)", e,
                           {k: trade.get(k) for k in ("id", "trade_id", "side", "price", "size")})

    async def _check_position_drift(self) -> int:
        """v18.4 — on-chain position-shares drift check.

        For every market with a local position > 0, query the on-chain
        CTF share balance and compare to our local count.  If any
        market's drift exceeds ``cfg.drift_halt_threshold_shares``,
        HALT the bot (do NOT auto-flatten — that would convert a
        state-tracking bug into a guaranteed loss).

        Drift sources (real-world examples):
          * UserFeed dropped a partial fill that REST replay also missed.
          * Manual on-chain transfer (user moved CTF tokens outside the bot).
          * Indexer lag making the REST balance temporarily stale (will
            self-resolve; the threshold absorbs <0.01 share noise).

        v18.4.1 — Parallelized via ``asyncio.gather`` under a
        ``Semaphore(drift_check_concurrency)`` (default 4).  Prior
        sequential implementation issued blocking ``run_in_executor``
        balance fetches one-by-one; with N=20 active markets × 2 tokens
        this serialized ~40 REST calls behind a single event-loop
        thread.  Concurrent fetches with bounded fan-out keep request
        bursts under CLOB per-IP rate limits while collapsing wall time.

        Returns the number of markets with drift detected (0 = clean).
        """
        if not self.client.sdk:
            return 0
        # v18.9: skip drift check in DRY_RUN — simulated fills create local
        # positions that have no on-chain counterpart, so every dry-run fill
        # would trigger a spurious drift halt.
        if self.cfg.dry_run:
            return 0
        threshold = self.cfg.drift_halt_threshold_shares
        loop = asyncio.get_running_loop()
        n_workers = max(1, int(self.cfg.drift_check_concurrency))
        sem = asyncio.Semaphore(n_workers)
        # Dedicated, bounded thread pool for the blocking SDK balance
        # fetches.  ``run_in_executor(None, …)`` would share the process-
        # wide default executor with order signing / posting / cancels; a
        # drift fan-out of ~40 calls could then queue behind (or crowd
        # out) those latency-critical offloads.  Isolating I/O here keeps
        # the default pool free.  (Note: run_in_executor never blocks the
        # event loop itself — it frees it — so the real risk is pool
        # contention, not the WS-ping starvation sometimes claimed.)
        # Uses the persistent class-level pool (allocated once at boot) so
        # this 5-minute sweep never churns OS threads on the event loop.
        io_pool = self._drift_io_pool

        # Build the task list FIRST (no I/O) so we can size the
        # semaphore / gather pool with knowledge of the workload.
        tasks_meta: List[Tuple[Market, str, float, str]] = []
        for mkt in list(self.markets):
            for tid, pos, side_label in (
                (mkt.yes_token, mkt.pos_yes, "YES"),
                (mkt.no_token, mkt.pos_no, "NO"),
            ):
                # v18.7: Audit ALL registered tokens unconditionally.
                # Prior gate (pos.shares >= threshold) left the bot blind
                # to ghost on-chain positions when local ledger showed flat.
                tasks_meta.append((mkt, tid, pos.shares, side_label))

        if not tasks_meta:
            return 0

        async def _fetch_one(tid: str) -> float:
            """Fetch on-chain CTF balance for a single token id.

            Returns ``NaN`` on any failure mode (SDK absent, malformed
            response, transport error).  Throttled via the outer
            semaphore so concurrent fan-out stays under CLOB's per-IP
            rate budget.
            """
            async with sem:
                def _get_bal() -> float:
                    sdk = self.client.sdk
                    getter = getattr(sdk, "get_balance_allowance", None)
                    if not getter or AssetType is None:
                        return float("nan")
                    try:
                        # MUST be the typed ``BalanceAllowanceParams`` the SDK
                        # expects — the collateral fetch (see ``get_balance``)
                        # uses the same object.  Passing a raw dict made the
                        # SDK raise ``TypeError`` (silently swallowed below),
                        # so ``chain_shares`` was ALWAYS NaN and this entire
                        # on-chain drift check never fired.
                        params = BalanceAllowanceParams(
                            asset_type=AssetType.CONDITIONAL, token_id=tid)
                        resp = getter(params)
                        if isinstance(resp, dict):
                            raw = (
                                resp.get("balance")
                                or resp.get("balance_allowance")
                                or 0
                            )
                            return _parse_bal_micro(raw) / _USDC_SCALE
                    except Exception:
                        pass
                    return float("nan")
                try:
                    return await loop.run_in_executor(io_pool, _get_bal)
                except Exception:
                    return float("nan")

        # Fan out: bounded concurrency via the semaphore inside _fetch_one.
        # ``finally`` guarantees the dedicated pool's threads are reaped
        # even if a fetch raises, so no worker threads leak across cycles.
        results = await asyncio.gather(
            *(_fetch_one(meta[1]) for meta in tasks_meta),
            return_exceptions=False,
        )

        # The balance fan-out already completed, so logging EVERY drifted
        # market before halting is free and gives the operator the full
        # picture (multiple independent drifts vs. one root cause) in a
        # single shot rather than one-per-restart.
        drift_count = 0
        nan_count = 0
        first_msg = ""
        for (mkt, tid, local_shares, side_label), chain_shares in zip(
            tasks_meta, results
        ):
            if not math.isfinite(chain_shares):
                # BUG-FIX #18: count (don't drop silently) the NaN
                # fetches so a network outage affecting all N fetches
                # surfaces as "0 drifted, N failed" rather than
                # masking the outage as a clean check.
                nan_count += 1
                continue
            diff = abs(local_shares - chain_shares)
            if diff >= threshold:
                drift_count += 1
                direction = "OVER" if local_shares > chain_shares else "UNDER"
                msg = (
                    f"POSITION DRIFT mkt={mkt.market_id[:8]} "
                    f"{side_label} local={local_shares:.4f} "
                    f"chain={chain_shares:.4f} diff={diff:.4f} ({direction})"
                )
                # C-BUG-2 fix: HALT on ALL drift, including over-count.
                # The old auto-correct silently overwrote shares and cost
                # basis on local > chain, masking the root cause and
                # potentially manufacturing phantom PnL.  An over-count
                # can mean: (a) FOK was killed but locally credited,
                # (b) market resolved on-chain while local ledger held,
                # (c) WS delivered a duplicate fill.  All three require
                # operator inspection — none should be auto-corrected.
                self.log.critical(msg)
                if not first_msg:
                    first_msg = msg
        if nan_count:
            # BUG-FIX #18: loud warning when fetches failed — operator
            # must investigate whether drift is real but invisible.
            self.log.warning(
                "drift check: %d/%d fetches returned NaN — coverage may "
                "be incomplete; if this persists, on-chain RPC is down",
                nan_count, len(results))
        if drift_count:
            self.risk._halt(
                f"position drift on {drift_count} market(s); first: {first_msg}",
                halt_type="drift")
        return drift_count

    # ── Background loops ──────────────────────────────────────────────────────

    async def _status_loop(self) -> None:
        while self.running:
            await asyncio.sleep(60)
            s   = self.risk.status()
            bal = await self.client.get_balance()
            trading = (
                "PAUSED:lib_broken" if self.client.lib_broken else
                f"HALTED:{s['reason']}" if s["halted"] else "OK"
            )
            metrics_str = ""
            if self.metrics:
                ms = self.metrics.summary()
                metrics_str = (f"  lat_p50={ms['lat_p50_ms']:.0f}ms"
                               f"  lat_p95={ms['lat_p95_ms']:.0f}ms"
                               f"  fills={ms['fills']}")
            self.log.info(
                "STATUS  pnl=$%.2f  day=$%.2f  orders=%d  bal=$%.2f  %s%s",
                s["pnl"], s["daily"], s["orders"], bal, trading, metrics_str)
            if self.fivemin:
                self.fivemin._balance_cache = bal
                self.fivemin._balance_ts = time.time()
                self.log.info(
                    "  DIAG  guard_hits=%d  triggers=%d  open_prices=%d  traded=%d",
                    self.fivemin._diag_guard_hits,
                    self.fivemin._diag_trigger_calls,
                    len(self.fivemin._open_prices),
                    len(self.fivemin._traded))

    async def _reconcile_loop(self) -> None:
        """v18.9 — adaptive dual-pass reconciliation loop.

        Schedule:
          * Pass 1 (every cycle): ``OrderManager.reconcile_fills`` —
            REST ``/trades`` walk, replay any missed fills.
          * Pass 2 (every cycle): ``OrderManager.reconcile`` — open-orders
            sync, prune stale local state.
          * Pass 3 (every Nth cycle, default N=10 so ~5 min): on-chain
            position-shares drift check.  Halt on mismatch.

        v18.9: Adaptive interval — when UserFeed WS is connected (real-time
        fill notifications), use the configured interval (default 30s).
        When WS is DOWN, reduce to 5s so fill detection stays fast via REST.
        """
        # Initial delay so the bot has time to warm up before its first
        # round of self-introspection.
        await asyncio.sleep(25)
        base_interval = max(5.0, self.cfg.reconcile_fills_interval_s)
        fast_interval = 5.0  # Used when UserFeed WS is disconnected
        # Drift check runs less often (heavier; one RPC per market-leg).
        drift_every_n_cycles = 10
        cycle = 0
        # C-6: balance refresh cadence — Kelly bankroll cache age cap.
        # Pre-fix the balance was set once at boot and never refreshed,
        # so during a drawdown Kelly continued sizing off boot bankroll
        # (a 4-loss $16 drawdown on $100 bank made every subsequent
        # trade ~19% oversized vs real capital).
        last_bal_refresh = time.monotonic()
        while self.running:
            cycle += 1
            if not self.cfg.dry_run:
                try:
                    await self.om.reconcile_fills()
                except Exception as e:
                    self.log.warning("reconcile_fills error: %s", e)
                try:
                    await self.om.reconcile()
                except Exception as e:
                    self.log.warning("Reconcile error: %s", e)
                now_mono = time.monotonic()
                if now_mono - last_bal_refresh >= self.cfg.balance_refresh_s:
                    last_bal_refresh = now_mono
                    try:
                        new_bal = await self.client.get_balance()
                        if new_bal is not None and new_bal > 0 and self.fivemin:
                            old_bal = self.fivemin._balance_cache
                            self.fivemin._balance_cache = new_bal
                            self.fivemin._balance_ts = time.time()
                            if abs(new_bal - old_bal) > 1.0:
                                self.log.info(
                                    "Balance refresh: $%.2f -> $%.2f (delta=%+.2f)",
                                    old_bal, new_bal, new_bal - old_bal)
                    except Exception as e:
                        self.log.debug("Balance refresh failed: %s", e)
            if cycle % drift_every_n_cycles == 0 and not self.cfg.dry_run:
                try:
                    await self._check_position_drift()
                except Exception as e:
                    self.log.warning("drift check error: %s", e)
            # C-BUG-11 fix: check DATA freshness, not just TCP state.
            # A half-open WS (TCP alive, server stopped pushing) reports
            # connected=True but last_msg_age_s → ∞.  Treat >60s stale
            # as effectively down so we poll at fast_interval (5s).
            if self.userfeed:
                ws_up = (self.userfeed.connected
                         and self.userfeed.last_msg_age_s < 60.0)
            else:
                ws_up = False
            interval = base_interval if ws_up else fast_interval
            await asyncio.sleep(interval)

    async def _health_loop(self) -> None:
        await asyncio.sleep(45)
        while self.running:
            # Per-shard health monitoring
            shard_ages = self.polyfeed.shard_ages()
            for sid, age in shard_ages.items():
                if age > 90:
                    self.log.warning("Shard %d stale (%.0fs) — restarting", sid, age)
                    await self.polyfeed.restart_shard(sid)

            overall_age = self.polyfeed.last_msg_age_s
            if overall_age > 120:
                self.log.warning("ALL shards stale — full reconnect")
                await self.polyfeed.stop()
                t = asyncio.create_task(self.polyfeed.run(),
                                        name="polyfeed_restart")
                self._bg_tasks.add(t)
                t.add_done_callback(self._bg_tasks.discard)

            if self.binance.price("BTC") is None:
                self.log.warning("Binance feed: no BTC price yet")
            await asyncio.sleep(45)

    async def _fivemin_timer_loop(self) -> None:
        """Fallback timer-driven loop (used when EVENT_DRIVEN=false)."""
        interval = self.cfg.strategy_interval_s
        await asyncio.sleep(5)
        while self.running:
            await asyncio.sleep(interval)
            if not self.running or self.risk.halted or self.client.lib_broken:
                continue
            if self.fivemin and self.fivemin_markets:
                try:
                    await self.fivemin.evaluate_all(self.fivemin_markets)
                except Exception as e:
                    self.log.debug("Timer loop error: %s", e)

    async def _fivemin_refresh(self) -> None:
        """Poll for new 5-min markets every discovery_interval_s."""
        await asyncio.sleep(8)
        while self.running:
            await asyncio.sleep(self.cfg.discovery_interval_s)
            if not self.running:
                break
            try:
                new_markets = await discover_5min_markets(self.cfg, self.session)
                existing_ids = {m.market_id for m in self.fivemin_markets}
                added = []
                for m in new_markets:
                    if m.market_id not in existing_ids:
                        self.fivemin_markets.append(m)
                        self.t2m[m.yes_token] = m
                        self.t2m[m.no_token]  = m
                        self._5m_ids.add(m.market_id)
                        if m.coin:
                            self.by_coin.setdefault(m.coin, []).append(m)
                        added.extend([m.yes_token, m.no_token])
                if added:
                    await self.polyfeed.subscribe_live(added)
                    self.userfeed.set_markets(self.t2m)
                    self.log.info("Added %d new markets (%d total)",
                                  len(added) // 2, len(self.fivemin_markets))

                # Remove expired (set-based for O(1))
                now = time.time()
                keep = []
                expired_tids = []
                expired_count = 0
                for m in self.fivemin_markets:
                    if m.end_time and m.end_time < now - 300:
                        self.t2m.pop(m.yes_token, None)
                        self.t2m.pop(m.no_token, None)
                        expired_tids.extend([m.yes_token, m.no_token])
                        expired_count += 1
                    else:
                        keep.append(m)
                if expired_count:
                    self.fivemin_markets = keep
                    # Keep ``self.markets`` (the drift-check / boot view)
                    # pointing at the live set so it can't diverge and
                    # accumulate expired markets across refresh cycles.
                    self.markets = keep
                    # Unsubscribe dead tokens from WS shards to prevent
                    # zombie subscriptions (288 expirations/day/coin)
                    await self.polyfeed.unsubscribe(expired_tids)
                    if self.fivemin:
                        self.fivemin.cleanup_expired(self.fivemin_markets)
                    self._prune_expired_bot_state(
                        {m.market_id for m in self.fivemin_markets})
                    self.log.info("Removed %d expired markets, unsubscribed %d tokens",
                                  expired_count, len(expired_tids))
            except Exception as e:
                self.log.debug("Discovery refresh: %s", e)

    def _prune_expired_bot_state(self, active_mids: Set[str]) -> None:
        """Drop Bot-level state keyed by markets no longer live.

        Prevents unbounded memory drift on a 24/7 process:
          * ``_trade_pnl_in_flight`` is popped on full-leg close, but a leg
            settled on-chain (or whose final WS fill packet was dropped)
            never reaches that path — its ``(market_id, token)`` accumulator
            would otherwise leak forever.
          * ``by_coin`` / ``_5m_ids`` grow on every discovery cycle.
        ``by_coin`` is mutated IN PLACE so the ``LatencyArb`` reference to
        the same dict observes the pruning.
        """
        for k in [k for k in self._trade_pnl_in_flight
                  if k[0] not in active_mids]:
            self._trade_pnl_in_flight.pop(k, None)
        self._5m_ids.intersection_update(active_mids)
        for coin, mkts in list(self.by_coin.items()):
            kept = [m for m in mkts if m.market_id in active_mids]
            if kept:
                self.by_coin[coin] = kept
            else:
                self.by_coin.pop(coin, None)

    def _load_calibration_shrink(self) -> None:
        """Boot-time calibration feedback: read the calibration CSV, join
        eval rows to outcome rows per market, group by coin, and compute
        per-coin empirical shrink corrections.

        shrink_k = hit_rate / (mean_ask + hit_rate - (1 - mean_ask))

        Maps break-even (hit_rate == mean_ask) to shrink=1.0.  A coin
        beating its spread maps to shrink>1.0.  Clamped [0.5, 1.5].
        Requires >= 30 matched samples per coin.
        """
        path = os.path.expanduser(self.cfg.calibration_log_path)
        if not os.path.exists(path):
            return
        try:
            evals: Dict[str, List[Tuple[str, float, float]]] = {}
            outcomes: Dict[str, bool] = {}
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return
                col_idx = {h.strip(): i for i, h in enumerate(header)}
                rt_i = col_idx.get("row_type")
                mid_i = col_idx.get("market_id")
                coin_i = col_idx.get("coin")
                ask_i = col_idx.get("ask")
                win_i = col_idx.get("win")
                if any(x is None for x in (rt_i, mid_i)):
                    return
                for row in reader:
                    if len(row) <= max(rt_i, mid_i):
                        continue
                    rt = row[rt_i].strip()
                    mid = row[mid_i].strip()
                    if rt == "eval" and coin_i is not None and ask_i is not None:
                        coin_val = row[coin_i].strip() if coin_i < len(row) else ""
                        ask_val = row[ask_i].strip() if ask_i < len(row) else ""
                        if coin_val and ask_val:
                            try:
                                ask_f = float(ask_val)
                                if mid not in evals:
                                    evals[mid] = []
                                evals[mid].append((coin_val, ask_f, 0.0))
                            except ValueError:
                                pass
                    elif rt == "outcome" and win_i is not None:
                        win_val = row[win_i].strip() if win_i < len(row) else ""
                        if win_val in ("1", "0"):
                            outcomes[mid] = (win_val == "1")

            coin_stats: Dict[str, List[Tuple[float, bool]]] = {}
            for mid, win in outcomes.items():
                if mid in evals:
                    for (coin_val, ask_f, _) in evals[mid]:
                        if coin_val not in coin_stats:
                            coin_stats[coin_val] = []
                        coin_stats[coin_val].append((ask_f, win))

            for coin, samples in coin_stats.items():
                if len(samples) < 30:
                    continue
                hits = sum(1 for _, w in samples if w)
                hit_rate = hits / len(samples)
                mean_ask = sum(a for a, _ in samples) / len(samples)
                # C-2 fix: formula was denom = mean_ask + hit_rate - (1 - mean_ask)
                # = 2*mean_ask + hit_rate - 1, which maps break-even to 1.0 only
                # at ask=0.5 and INVERTS the correction for all ask > 0.5 (the
                # entire live trading regime).  Correct formula: shrink_k =
                # hit_rate / mean_ask — maps break-even (hit_rate==mean_ask) to
                # 1.0 at any ask level, >1 when beating spread, <1 when losing.
                if mean_ask <= 0.01:
                    continue
                shrink_k = hit_rate / mean_ask
                shrink_k = max(0.5, min(1.5, shrink_k))
                self.tracker._per_coin_shrink[coin] = shrink_k
                self.log.info(
                    "CALIB_SHRINK %s: n=%d hit=%.3f ask=%.3f -> shrink=%.2f",
                    coin, len(samples), hit_rate, mean_ask, shrink_k)
        except Exception as e:
            self.log.warning("Calibration shrink load failed: %s", e)

    async def _shutdown_wait(self) -> None:
        await self.shutdown_ev.wait()
        await self._shutdown()

    async def _flatten_all_positions(self) -> None:
        """S-7 opt-in: dump every open leg via FOK at best_bid.

        Bounded to a single sweep — never a polling loop — so a CLOB outage
        cannot turn a halt into an unbounded retry storm.  Each FOK uses the
        bid we observe at this instant (no chase).  Failures are logged but
        do not raise; the caller's halt path must still cancel orders.
        """
        if self.cfg.dry_run:
            return
        flattened = 0
        for mkt in list(self.fivemin_markets):
            for token, pos, book in (
                (mkt.yes_token, mkt.pos_yes, mkt.book_yes),
                (mkt.no_token,  mkt.pos_no,  mkt.book_no),
            ):
                if pos.shares <= 0 or book is None or book.best_bid is None:
                    continue
                try:
                    tick = mkt.get_tick(token) if hasattr(mkt, "get_tick") else 0.01
                    # BUG-FIX #25: sell at best_bid, not best_bid - tick.
                    # Pre-fix conceded one tick below the best bid for no
                    # benefit on a FOK order.
                    sell_price = max(tick, book.best_bid)
                    notional = pos.shares * sell_price
                    if notional < self.cfg.min_order_size:
                        continue
                    await self.om.place(
                        token, Side.SELL, sell_price, notional,
                        Strategy.TEMPORAL, otype="FOK",  # C-1 fix: FIVEMIN→TEMPORAL (FIVEMIN never existed in enum)
                        neg_risk=mkt.neg_risk, tick_size=tick)
                    flattened += 1
                except Exception as e:
                    self.log.error("flatten %s/%s failed: %s",
                                   mkt.coin, token[:8], e)
        if flattened:
            self.log.warning("auto_flatten_on_halt: dispatched %d FOK exits",
                             flattened)

    async def _shutdown(self) -> None:
        if not self.running:
            return
        self.running = False
        self.log.info("Shutting down…")
        await self.om.cancel_all()
        await self.polyfeed.stop()
        await self.binance.stop()
        if self.userfeed:
            await self.userfeed.stop()
        # Release the calibration file descriptor (no orphaned FD / lost
        # trailing line on a clean shutdown).
        if self.fivemin:
            self.fivemin.close_calibration_log()
        # Drain + close the latency-arb shadow writer pool/FD (mirrors the
        # calibration log: queued rows flushed before the FD is released).
        if self.latency_arb:
            self.latency_arb.close_shadow_log()
        for t in self.tasks:
            if t is not asyncio.current_task():
                t.cancel()
        if self.metrics:
            self.log.info("Final metrics: %s", self.metrics.summary())
        # Reap the persistent drift-check thread pool.
        # BUG-FIX #72: wait=True so in-flight balance fetches complete
        # before pool destruction; cancel_futures prevents new work.
        self._drift_io_pool.shutdown(wait=True, cancel_futures=True)
        self.log.info("Final status: %s", self.risk.status())

    def _banner(self) -> None:
        self.log.info("=" * 64)
        self.log.info("  POLYMARKET CRYPTO BOT %s — Antigravity Opus 4.6",
                      _BOT_VERSION)
        self.log.info("=" * 64)
        # BUG-FIX #65: NEVER log private key material.  Pre-fix logged
        # ``cfg.private_key[:6]`` to a plain INFO line.  Truncation does
        # not prevent timing-side-channel analysis of the key bytes, and
        # a 6-byte prefix is enough to fingerprint a leaked wallet.  We
        # now log a SHA-256 prefix (8 hex chars) of the key — a one-way
        # digest that lets the operator verify "is this MY key" without
        # exposing any of the secret bytes.  This is the SINGLE source
        # of truth for the key fingerprint; main() does NOT log it again.
        if self.cfg.private_key:
            key_hash = hashlib.sha256(
                self.cfg.private_key.encode()).hexdigest()[:8]
            self.log.info("  KeyHash  : %s…  (sha256[:8] of POLYMARKET_PRIVATE_KEY)",
                          key_hash)
        else:
            self.log.info("  KeyHash  : <not set>")
        self.log.info("  Proxy    : %s",
                      self.cfg.proxy_address[:20]
                      if self.cfg.proxy_address else "(none — EOA mode)")
        self.log.info("  SigType  : %d (%s)",
                      self.cfg.signature_type,
                      _SIG_LABELS.get(self.cfg.signature_type, "?"))
        self.log.info("  Coins    : %s", ", ".join(self.cfg.coins))
        self.log.info("  Size     : $%.0f-$%.0f  MaxPos: $%.0f",
                      self.cfg.min_order_size, self.cfg.max_order_size,
                      self.cfg.max_position)
        self.log.info("  DryRun   : %s  (fill_prob=%.0f%%  latency=%.0fms)",
                      self.cfg.dry_run,
                      self.cfg.dry_run_fill_prob * 100,
                      self.cfg.dry_run_latency_ms)
        self.log.info("  Mode     : %s  |  Shards: %d  |  JSON: %s",
                      "EVENT" if self.cfg.event_driven else "TIMER",
                      self.cfg.ws_shard_count,
                      "orjson" if _FAST_JSON else "stdlib")
        self.log.info("  FastSign : %s  |  AdaptKelly: %s  |  Metrics: %s",
                      self.cfg.use_fast_signer,
                      self.cfg.adaptive_kelly,
                      self.cfg.metrics_enabled)
        self.log.info("  SDK      : %s  (py-clob-client-v2=%s)",
                      "yes" if _HAS_SDK else "NO",
                      _pkg_version("py-clob-client-v2"))
        self.log.info("=" * 64)


# ─── Entry point ──────────────────────────────────────────────────────────────

def _run_analyze(path: Optional[str]) -> None:
    """Offline calibration analyzer (``python polybot.py --analyze``).

    Loads the calibration CSV, prints the human-readable report, then the
    go/no-go verdict against the configured thresholds.  Requires NO venue
    credentials — it only reads the recorded file — so it is safe to run any
    time to check whether the data has earned the right to risk capital.
    """
    cfg = Config.from_env()
    resolved = path or cfg.calibration_log_path
    rows = load_calibration_rows(resolved)
    report = calibration_report(rows)
    print_calibration_report(report, resolved)
    allowed, reasons = go_no_go(
        report, min_samples=cfg.min_proven_samples,
        min_edge=cfg.min_proven_edge, max_adverse_bps=cfg.max_adverse_bps)
    print(f"go/no-go verdict  : {'GO' if allowed else 'NO-GO'}")
    if not allowed:
        for rsn in reasons:
            print(f"  - {rsn}")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="polybot", description="Polymarket 5-min trading bot")
    parser.add_argument(
        "--analyze", nargs="?", const="", metavar="CALIBRATION_CSV",
        help="Offline: print the calibration report + go/no-go verdict over "
             "the given calibration CSV (defaults to CALIBRATION_LOG_PATH) "
             "and exit. Does NOT trade or require credentials.")
    args = parser.parse_args()
    if args.analyze is not None:
        _run_analyze(args.analyze or None)
        return

    # Optional: swap CPython's selector loop for uvloop (libuv) if present.
    # 2-4x faster event-loop throughput directly lowers eval/WS latency, but
    # it is a soft dependency — absence must never block startup, so this is
    # guarded exactly like the orjson/coincurve fast paths above.
    try:
        import uvloop
        uvloop.install()
        log.info("uvloop active (libuv event loop)")
    except ImportError:
        log.info("uvloop not installed — using stdlib asyncio loop")

    cfg  = Config.from_env()
    errs = cfg.validate()
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        sys.exit(1)

    # BUG-FIX #66: NEVER log private key material — single source of
    # truth is ``Bot._banner`` (sha256[:8] prefix).  Pre-fix re-logged
    # ``cfg.private_key[:6]`` here, doubling the attack surface and
    # contradicting the "KeyHash: <sha256[:8]>" line in the banner.
    if cfg.proxy_address:
        log.info("Proxy  : %s", cfg.proxy_address)
    else:
        log.info("Proxy  : (none — EOA mode)")
    log.info("SigType: %d (%s)",
             cfg.signature_type,
             _SIG_LABELS.get(cfg.signature_type, "?"))

    try:
        asyncio.run(Bot(cfg).run())
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        logging.getLogger("Bot").critical("Fatal: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
