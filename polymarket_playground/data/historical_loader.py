"""HistoricalLoader — episode data source using cached market data.

Reads price history from a local MarketCache (SQLite) and replays
price series step-by-step for training agents. No live API calls.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone

from polymarket_playground.core.state import MarketState
from polymarket_playground.data.market_cache import MarketCache

logger = logging.getLogger(__name__)


class HistoricalLoader:
    """Produce MarketState episodes from cached Polymarket price data.

    Reads price history from MarketCache and replays random windows
    of historical prices each episode. All data comes from the local
    SQLite cache — no API calls at runtime.
    """

    def __init__(
        self,
        cache: MarketCache,
        max_steps: int = 50,
        seed: int | None = None,
    ) -> None:
        self.cache = cache
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._step: int = 0
        self._state: MarketState | None = None
        # Current episode's price series (sliced window from cache)
        self._episode_prices: list[float] = []
        # Cached per-episode data (avoid repeated DB queries)
        self._episode_outcome: float | None = None
        self._episode_meta: dict = {}
        # Available market IDs (refreshed on filter change)
        self._market_ids: list[str] = []
        self._refresh_market_list()

    def _refresh_market_list(self) -> None:
        """Refresh the list of available markets from cache."""
        markets = self.cache.get_markets()
        self._market_ids = [m["id"] for m in markets]
        logger.info(
            "Historical loader: %d markets available",
            len(self._market_ids),
        )

    def set_market_filter(self, category: str) -> None:
        """Filter markets to a specific category."""
        self._market_ids = self.cache.get_market_ids_by_category(category)
        logger.info(
            "Filtered to category '%s': %d markets",
            category,
            len(self._market_ids),
        )

    @property
    def available_markets(self) -> int:
        return len(self._market_ids)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def next_episode(self) -> MarketState:
        """Start a new episode from cached data."""
        self._step = 0

        if not self._market_ids:
            self._refresh_market_list()
            if not self._market_ids:
                raise RuntimeError(
                    "No markets in cache. Run sync first: POST /api/data/sync"
                )

        market_id = self._rng.choice(self._market_ids)
        return self._start_episode(market_id)

    def advance(self) -> MarketState:
        """Advance one step — replay next historical price."""
        if self._state is None:
            raise RuntimeError("Call next_episode() before advance().")
        if self.is_truncated():
            raise RuntimeError("Episode is already truncated.")

        self._step += 1
        return self._advance_historical()

    def is_truncated(self) -> bool:
        return self._step >= self.max_steps

    # ------------------------------------------------------------------
    # Historical replay
    # ------------------------------------------------------------------

    def _start_episode(self, market_id: str) -> MarketState:
        """Load a market's price history and start replaying a random window."""
        history = self.cache.get_price_history(market_id)
        if not history:
            raise RuntimeError(f"No price history for market {market_id}")

        prices = [h["p"] for h in history]
        meta = self.cache.get_market_metadata(market_id) or {}

        # Cache per-episode data
        self._episode_meta = meta

        row = self.cache._conn.execute(
            "SELECT outcome FROM markets WHERE id = ?", (market_id,)
        ).fetchone()
        self._episode_outcome = float(row["outcome"]) if row and row["outcome"] is not None else None

        # Slice a random window
        window_size = self.max_steps + 1
        if len(prices) <= window_size:
            self._episode_prices = self._resample(prices, window_size)
        else:
            start = self._rng.randint(0, len(prices) - window_size)
            self._episode_prices = prices[start : start + window_size]

        question = meta.get("question", market_id)

        # Build initial state
        yes_price = self._episode_prices[0]
        self._state = self._build_state_from_price(
            market_id=market_id,
            question=question,
            yes_price=yes_price,
            time_elapsed=0.0,
            is_resolved=False,
            resolution=None,
            market_meta=meta,
            outcome=self._episode_outcome,
        )
        return self._state

    def _advance_historical(self) -> MarketState:
        """Replay the next price point."""
        idx = min(self._step, len(self._episode_prices) - 1)
        yes_price = self._episode_prices[idx]
        elapsed_frac = self._step / self.max_steps

        # Resolve on the last step
        is_last = self._step >= self.max_steps - 1
        outcome = self._episode_outcome

        is_resolved = is_last and outcome is not None
        resolution = outcome if is_resolved else None

        self._state = self._build_state_from_price(
            market_id=self._state.market_id,
            question=self._state.question,
            yes_price=yes_price,
            time_elapsed=round(elapsed_frac, 4),
            is_resolved=is_resolved,
            resolution=resolution,
            market_meta=self._episode_meta,
            outcome=outcome,
        )
        return self._state

    def _build_state_from_price(
        self,
        market_id: str,
        question: str,
        yes_price: float,
        time_elapsed: float,
        is_resolved: bool,
        resolution: float | None,
        market_meta: dict,
        outcome: float | None = None,
    ) -> MarketState:
        """Build a MarketState from a single historical price point."""
        yes_price = max(0.001, min(0.999, yes_price))
        no_price = round(1.0 - yes_price, 4)

        # Realistic spread based on price level
        base_spread = 0.01 + 0.02 * (1 - abs(yes_price - 0.5) * 2)
        spread = round(base_spread * self._rng.uniform(0.5, 1.5), 4)
        half_spread = spread / 2.0

        yes_bid = round(max(0.001, yes_price - half_spread), 4)
        yes_ask = round(min(0.999, yes_price + half_spread), 4)
        no_bid = round(max(0.001, no_price - half_spread), 4)
        no_ask = round(min(0.999, no_price + half_spread), 4)

        base_depth = float(market_meta.get("liquidityNum", 1000) or 1000)
        book_depth = round(base_depth * self._rng.uniform(0.5, 1.5), 2)

        volume_24hr = float(market_meta.get("volume24hr", 0) or 0)
        liquidity = float(market_meta.get("liquidityNum", 0) or 0)
        signals = {
            "best_bid": yes_bid,
            "best_ask": yes_ask,
            "spread": spread,
            "book_depth_bid": round(book_depth * 0.5, 2),
            "book_depth_ask": round(book_depth * 0.5, 2),
            "last_trade_price": round(yes_price, 4),
            "volume_24hr": volume_24hr,
            "liquidity": liquidity,
        }

        return MarketState(
            market_id=market_id,
            market_type="polymarket",
            question=question,
            resolution_time=None,
            yes_price=round(yes_price, 4),
            no_price=no_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread=spread,
            book_depth=book_depth,
            time_elapsed=time_elapsed,
            time_remaining=max(0.0, (1.0 - time_elapsed) * 3600),
            timestamp=datetime.now(tz=timezone.utc),
            signals=signals,
            is_resolved=is_resolved,
            resolution=resolution,
        )

    def _resample(self, prices: list[float], target_len: int) -> list[float]:
        """Resample a short price series to fill target_len via interpolation."""
        if len(prices) == 0:
            return [0.5] * target_len
        if len(prices) == 1:
            return prices * target_len

        result = []
        for i in range(target_len):
            frac = i / (target_len - 1)
            src_idx = frac * (len(prices) - 1)
            lo = int(src_idx)
            hi = min(lo + 1, len(prices) - 1)
            t = src_idx - lo
            result.append(prices[lo] * (1 - t) + prices[hi] * t)
        return result
