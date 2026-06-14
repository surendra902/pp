from __future__ import annotations

import json
import urllib.request
from decimal import Decimal, InvalidOperation

from web3 import Web3

_USDC_DECIMALS = 6
_USDC_SCALE = 10 ** _USDC_DECIMALS


def parse_bal_micro(raw: object) -> int:
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
            micro = int(Decimal(s) * _USDC_SCALE)
            return sign * micro

        if "." in s:
            whole, frac = s.split(".", 1)
            if "." in frac:
                return 0
            whole = whole or "0"
            frac = frac[:_USDC_DECIMALS].ljust(_USDC_DECIMALS, "0")
            if not (whole.isdigit() and frac.isdigit()):
                return 0
            micro = int(whole) * _USDC_SCALE + int(frac)
            return sign * micro

        if not s.isdigit():
            return 0
        return sign * int(s)
    except (ValueError, TypeError, InvalidOperation):
        return 0


def parse_bal_float(raw: object) -> float:
    micro = parse_bal_micro(raw)
    if micro < 0:
        return 0.0
    return micro / _USDC_SCALE


def lookup_proxy_address_http(eoa: str, timeout_s: int = 5) -> str | None:
    """HTTP fallback for proxy address.

    BUGFIX: original bot_LIVE.py had stray braces in URL formatting.
    """
    eoa_cs = Web3.to_checksum_address(eoa)
    url = f"https://clob.polymarket.com/proxy-wallet?address={eoa_cs}"

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode())

    addr = data.get("proxyAddress") or data.get("proxy_address") or data.get("address")
    if not addr:
        return None
    if addr == "0x" + "0" * 40:
        return None
    return Web3.to_checksum_address(addr)
