"""Polymarket API client — fetches real market data from the Gamma and CLOB APIs."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class PolymarketClient:
    """Read-only client for the Polymarket Gamma + CLOB public APIs.

    No authentication required for market data endpoints.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"

    # ------------------------------------------------------------------
    # Gamma API  (markets, events)
    # ------------------------------------------------------------------

    def list_events(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 20,
        offset: int = 0,
        order: str = "volume24hr",
        ascending: bool = False,
        tag_id: int | None = None,
    ) -> list[dict]:
        """List events from the Gamma API, sorted by volume by default."""
        params: dict[str, Any] = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if tag_id is not None:
            params["tag_id"] = tag_id
        resp = self._session.get(f"{GAMMA_BASE}/events", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def list_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 20,
        offset: int = 0,
        order: str = "volume24hr",
        ascending: bool = False,
        tag: str | None = None,
    ) -> list[dict]:
        """List markets from the Gamma API, optionally filtered by tag/category."""
        params: dict[str, Any] = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if tag:
            params["tag"] = tag
        resp = self._session.get(f"{GAMMA_BASE}/markets", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_market(self, market_id: str) -> dict:
        """Fetch a single market by its Gamma market ID or slug."""
        # Try by ID first, then by slug
        resp = self._session.get(f"{GAMMA_BASE}/markets/{market_id}", timeout=10)
        if resp.status_code == 404:
            resp = self._session.get(
                f"{GAMMA_BASE}/markets", params={"slug": market_id}, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                return data[0]
            raise ValueError(f"Market not found: {market_id}")
        resp.raise_for_status()
        return resp.json()

    def get_event(self, event_id: str) -> dict:
        """Fetch a single event by ID."""
        resp = self._session.get(f"{GAMMA_BASE}/events/{event_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # CLOB API  (orderbook, prices)
    # ------------------------------------------------------------------

    def get_orderbook(self, token_id: str) -> dict:
        """Fetch the order book for a YES token ID from the CLOB API."""
        resp = self._session.get(
            f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def get_midpoint(self, token_id: str) -> float:
        """Get the midpoint price for a token."""
        resp = self._session.get(
            f"{CLOB_BASE}/midpoint", params={"token_id": token_id}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("mid", data.get("mid_price", "0.5")))

    def get_price_history(
        self,
        token_id: str,
        interval: str = "1d",
        fidelity: int = 60,
    ) -> list[dict]:
        """Fetch price history for a token.

        Returns list of {t: timestamp, p: price} dicts.
        """
        resp = self._session.get(
            f"{CLOB_BASE}/prices-history",
            params={
                "market": token_id,
                "interval": interval,
                "fidelity": fidelity,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("history", [])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_clob_token_ids(market: dict) -> tuple[str | None, str | None]:
        """Extract (yes_token_id, no_token_id) from a Gamma market dict.

        The clobTokenIds field is a JSON-encoded string array: ["yes_id", "no_id"].
        """
        raw = market.get("clobTokenIds", "[]")
        if isinstance(raw, str):
            ids = json.loads(raw)
        else:
            ids = raw
        yes_id = ids[0] if len(ids) > 0 else None
        no_id = ids[1] if len(ids) > 1 else None
        return yes_id, no_id

    @staticmethod
    def parse_outcome_prices(market: dict) -> tuple[float, float]:
        """Extract (yes_price, no_price) from a Gamma market dict."""
        raw = market.get("outcomePrices", "[]")
        if isinstance(raw, str):
            prices = json.loads(raw)
        else:
            prices = raw
        yes = float(prices[0]) if len(prices) > 0 else 0.5
        no = float(prices[1]) if len(prices) > 1 else 0.5
        return yes, no
