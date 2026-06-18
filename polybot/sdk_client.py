from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any, Optional

from .logging_setup import get_logger


log = get_logger("sdk")


@dataclass
class ApiCreds:
    api_key: str
    api_secret: str
    api_passphrase: str


class SdkClient:
    """Minimal wrapper around py_clob_client_v2.

    Purpose: keep signing + order schema in the SDK (non-hallucinated) while we
    modularize the rest of the bot.

    This is NOT a full port of bot_LIVE.py; it is the smallest surface needed
    to:
      - derive API creds
      - compute WS user auth payload
      - submit/cancel orders via SDK

    Dry-run safety: call sites should gate all order placement behind cfg.dry_run.
    """

    def __init__(self, *, clob_url: str, chain_id: int, private_key: str, signature_type: int = 2, proxy_address: str = ""):
        self.clob_url = clob_url.rstrip("/")
        self.chain_id = int(chain_id)
        self.private_key = private_key
        self.signature_type = int(signature_type)
        self.proxy_address = proxy_address

        self._sdk: Optional[Any] = None
        self.creds: Optional[ApiCreds] = None
        self.trader_address: str = proxy_address

    @property
    def sdk(self) -> Any:
        if self._sdk is None:
            raise RuntimeError("SdkClient not initialized")
        return self._sdk

    async def initialize(self) -> None:
        try:
            from eth_account import Account
        except Exception as e:
            raise RuntimeError("Missing eth-account") from e

        try:
            from py_clob_client_v2.client import ClobClient  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing py-clob-client-v2. Install: pip install py-clob-client-v2") from e

        signer_eoa = Account.from_key(self.private_key).address
        if not self.trader_address:
            self.trader_address = signer_eoa

        kw: dict[str, Any] = {
            "host": self.clob_url,
            "chain_id": self.chain_id,
            "key": self.private_key,
        }
        if self.signature_type in (1, 2) and self.proxy_address:
            kw["signature_type"] = self.signature_type
            kw["funder"] = self.proxy_address
        elif self.signature_type == 0:
            kw["signature_type"] = 0

        loop = asyncio.get_running_loop()
        sdk = ClobClient(**kw)

        derive = getattr(sdk, "create_or_derive_api_key", None) or getattr(sdk, "create_or_derive_api_creds", None)
        if derive is None:
            raise RuntimeError("SDK incompatible: missing create_or_derive_api_key")

        creds = await loop.run_in_executor(None, derive)
        sdk.set_api_creds(creds)

        self._sdk = sdk
        self.creds = ApiCreds(api_key=creds.api_key, api_secret=creds.api_secret, api_passphrase=creds.api_passphrase)
        log.info("SDK initialized; trader=%s sig_type=%s", self.trader_address, self.signature_type)

    def hmac_headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        if self.creds is None:
            raise RuntimeError("SdkClient missing creds")
        ts = str(int(time.time()))
        msg = (ts + method + path + body).encode()
        # bot_LIVE.py: try base64 decode; fall back to raw
        try:
            secret = base64.urlsafe_b64decode(self.creds.api_secret)
        except Exception:
            secret = self.creds.api_secret.encode()
        sig = base64.urlsafe_b64encode(hmac.new(secret, msg, hashlib.sha256).digest()).decode()
        return {
            "POLY_ADDRESS": self.trader_address,
            "POLY_SIGNATURE": sig,
            "POLY_TIMESTAMP": ts,
            "POLY_NONCE": "0",
            "POLY_API_KEY": self.creds.api_key,
            "POLY_PASSPHRASE": self.creds.api_passphrase,
        }
