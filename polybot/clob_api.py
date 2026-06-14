from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .http_client import HttpClient, HttpConfig
from .retry import CircuitBreaker


@dataclass
class ClobAuth:
    """Authentication container.

    NOTE: Exact auth header construction depends on Polymarket's current API
    reference. This refactor intentionally centralizes auth in one place so
    the rest of the bot never re-implements headers ad-hoc.
    """

    api_key: str
    api_secret: str
    api_passphrase: str


class ClobApi:
    """Thin wrapper over Polymarket CLOB REST endpoints.

    Chairman rule: all requests flow through HttpClient for:
      - rate limit budgeting
      - retries + backoff
      - circuit breaker

    This file provides stable method signatures; implementation details can
    be filled in by extracting the exact payloads from bot_LIVE.py without
    changing call sites.
    """

    def __init__(self, *, base_url: str, http: HttpClient, auth: Optional[ClobAuth] = None):
        self.base_url = base_url.rstrip("/")
        self.http = http
        self.auth = auth

    def _headers(self) -> dict[str, str]:
        # TODO: build the required auth headers per Polymarket docs.
        # Keep empty for public endpoints.
        return {"Accept": "application/json"}

    async def get_orderbook_summary(self, token_id: str) -> dict[str, Any]:
        """Fetch summary including tick size / min order size.

        Verified in ecosystem SDK docs that summary exists; exact path is
        intentionally centralized here.
        """
        url = f"{self.base_url}/orderbook/{token_id}"
        return await self.http.request_json("GET", url, headers=self._headers())

    async def create_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a signed order payload."""
        url = f"{self.base_url}/order"
        return await self.http.request_json("POST", url, headers=self._headers(), json_body=payload)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        url = f"{self.base_url}/order/{order_id}"
        return await self.http.request_json("DELETE", url, headers=self._headers())

    async def list_trades(self, *, since_ts: int | None = None) -> dict[str, Any]:
        # TODO: fill exact endpoint and params.
        url = f"{self.base_url}/trades"
        return await self.http.request_json("GET", url, headers=self._headers())
