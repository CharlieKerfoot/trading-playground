"""PolymarketAdapter — fetches real market data from the Polymarket API."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import Episode
from polymarket_playground.data.polymarket_client import PolymarketClient
from polymarket_playground.markets.base_adapter import MarketAdapter

logger = logging.getLogger(__name__)


class PolymarketAdapter(MarketAdapter):
    """Adapter that serves real Polymarket markets via the Gamma + CLOB APIs.

    On init, fetches the top active markets by 24h volume. Each call to
    get_markets() returns the cached list. fetch_signals() pulls live
    orderbook data from the CLOB API.
    """

    SIGNALS = [
        "best_bid",
        "best_ask",
        "spread",
        "book_depth_bid",
        "book_depth_ask",
        "last_trade_price",
        "volume_24hr",
        "liquidity",
    ]

    def __init__(
        self,
        client: PolymarketClient | None = None,
        n_markets: int = 20,
        min_liquidity: float = 10_000,
    ) -> None:
        self.client = client or PolymarketClient()
        self._n_markets = n_markets
        self._min_liquidity = min_liquidity
        # market_id -> full Gamma market dict
        self._market_cache: dict[str, dict] = {}
        # market_id -> yes_token_id
        self._token_cache: dict[str, str] = {}
        self._refresh_markets()

    def _refresh_markets(self) -> None:
        """Fetch active markets from the Gamma API."""
        try:
            markets = self.client.list_markets(
                active=True,
                closed=False,
                limit=self._n_markets * 2,  # fetch extra, filter later
                order="volume24hr",
                ascending=False,
            )
        except Exception as e:
            logger.warning("Failed to fetch markets from Polymarket: %s", e)
            return

        self._market_cache.clear()
        self._token_cache.clear()

        for m in markets:
            mid = str(m.get("id", ""))
            liq = m.get("liquidityNum") or 0
            if liq < self._min_liquidity:
                continue

            # Must have clob token IDs
            yes_id, _ = PolymarketClient.parse_clob_token_ids(m)
            if not yes_id:
                continue

            self._market_cache[mid] = m
            self._token_cache[mid] = yes_id

            if len(self._market_cache) >= self._n_markets:
                break

        logger.info("Loaded %d active Polymarket markets", len(self._market_cache))

    # ------------------------------------------------------------------
    # MarketAdapter interface
    # ------------------------------------------------------------------

    def get_markets(self) -> list[str]:
        if not self._market_cache:
            self._refresh_markets()
        return list(self._market_cache.keys())

    def fetch_signals(self, market_id: str, timestamp: datetime | None) -> dict[str, float]:
        """Fetch live orderbook signals from the CLOB API."""
        m = self._market_cache.get(market_id, {})
        token_id = self._token_cache.get(market_id)

        signals: dict[str, float] = {
            "best_bid": m.get("bestBid") or 0.0,
            "best_ask": m.get("bestAsk") or 0.0,
            "spread": 0.0,
            "book_depth_bid": 0.0,
            "book_depth_ask": 0.0,
            "last_trade_price": m.get("lastTradePrice") or 0.0,
            "volume_24hr": m.get("volume24hr") or 0.0,
            "liquidity": m.get("liquidityNum") or 0.0,
        }

        if not token_id:
            return signals

        try:
            book = self.client.get_orderbook(token_id)
            bids = book.get("bids", [])
            asks = book.get("asks", [])

            if bids:
                signals["best_bid"] = float(bids[0]["price"])
                signals["book_depth_bid"] = sum(float(b["size"]) for b in bids[:5])
            if asks:
                signals["best_ask"] = float(asks[0]["price"])
                signals["book_depth_ask"] = sum(float(a["size"]) for a in asks[:5])
            if bids and asks:
                signals["spread"] = signals["best_ask"] - signals["best_bid"]
        except Exception as e:
            logger.debug("Failed to fetch orderbook for %s: %s", market_id, e)

        return signals

    def build_agent_context(self, state: MarketState) -> str:
        m = self._market_cache.get(state.market_id, {})
        question = m.get("question") or state.question or state.market_id
        signals = state.signals

        return (
            f"Polymarket — {question}\n"
            f"YES {state.yes_price:.3f} / NO {state.no_price:.3f} | "
            f"Spread {state.spread:.4f}\n"
            f"Best bid: {signals.get('best_bid', 0):.3f} | "
            f"Best ask: {signals.get('best_ask', 0):.3f}\n"
            f"Book depth: bid ${signals.get('book_depth_bid', 0):,.0f} / "
            f"ask ${signals.get('book_depth_ask', 0):,.0f}\n"
            f"24h volume: ${signals.get('volume_24hr', 0):,.0f} | "
            f"Liquidity: ${signals.get('liquidity', 0):,.0f}\n"
            f"Time remaining: {state.time_remaining:.0f}s\n"
        )

    def select_episodes(self, filters: dict) -> list[Episode]:
        return []

    # ------------------------------------------------------------------
    # Extra: build MarketState from a Gamma market dict + orderbook
    # ------------------------------------------------------------------

    def refresh_market(self, market_id: str) -> dict:
        """Re-fetch a single market's data from the Gamma API."""
        try:
            m = self.client.get_market(market_id)
            self._market_cache[market_id] = m
            yes_id, _ = PolymarketClient.parse_clob_token_ids(m)
            if yes_id:
                self._token_cache[market_id] = yes_id
            return m
        except Exception:
            return self._market_cache.get(market_id, {})

    def build_state(self, market_id: str) -> MarketState:
        """Build a MarketState from live API data for the given market."""
        # Re-fetch fresh data from Gamma for up-to-date prices
        m = self.refresh_market(market_id)
        if not m:
            self._refresh_markets()
            m = self._market_cache.get(market_id)
            if not m:
                raise ValueError(f"Unknown market: {market_id}")

        yes_price, no_price = PolymarketClient.parse_outcome_prices(m)

        # Use Gamma's bestBid/bestAsk (tight market prices)
        best_bid = m.get("bestBid") or 0.0
        best_ask = m.get("bestAsk") or 0.0
        spread = m.get("spread") or (best_ask - best_bid if best_bid and best_ask else 0.0)

        # Fetch orderbook for depth signals
        signals = self.fetch_signals(market_id, None)
        # Override signals bid/ask with Gamma's tighter values
        signals["best_bid"] = best_bid
        signals["best_ask"] = best_ask
        signals["spread"] = spread

        book_depth = signals.get("book_depth_bid", 0) + signals.get("book_depth_ask", 0)

        # Parse end date for time remaining
        end_date_str = m.get("endDate")
        resolution_time = None
        time_remaining = 0.0
        if end_date_str:
            try:
                resolution_time = datetime.fromisoformat(
                    end_date_str.replace("Z", "+00:00")
                )
                from datetime import timezone
                now = datetime.now(tz=timezone.utc)
                time_remaining = max(0, (resolution_time - now).total_seconds())
            except (ValueError, TypeError):
                pass

        return MarketState(
            market_id=market_id,
            market_type="polymarket",
            question=m.get("question", ""),
            resolution_time=resolution_time,
            yes_price=yes_price,
            no_price=no_price,
            yes_bid=best_bid,
            yes_ask=best_ask,
            no_bid=round(1 - best_ask, 4) if best_ask else 0,
            no_ask=round(1 - best_bid, 4) if best_bid else 0,
            spread=spread,
            book_depth=book_depth,
            time_elapsed=0.0,
            time_remaining=time_remaining,
            timestamp=datetime.utcnow(),
            signals=signals,
            is_resolved=bool(m.get("closed", False)),
        )
