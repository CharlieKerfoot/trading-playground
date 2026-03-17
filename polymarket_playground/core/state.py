"""MarketState — universal market snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MarketState:
    """Universal market snapshot seen by all agents."""

    # Contract identity
    market_id: str = ""
    market_type: str = ""  # "btc_price" | "election" | "sports" | ...
    question: str = ""  # Human-readable
    resolution_time: datetime | None = None

    # Polymarket prices
    yes_price: float = 0.5
    no_price: float = 0.5
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    no_bid: float = 0.0
    no_ask: float = 0.0
    spread: float = 0.0
    book_depth: float = 0.0  # liquidity at best bid/ask (USDC)

    # Time
    time_elapsed: float = 0.0  # fraction of contract life elapsed [0,1]
    time_remaining: float = 0.0  # seconds until resolution
    timestamp: datetime | None = None

    # External signals (market-type specific)
    signals: dict[str, float] = field(default_factory=dict)

    # Resolution
    is_resolved: bool = False
    resolution: float | None = None  # 1.0 = YES, 0.0 = NO

    # Agent's current position (populated by env after each step)
    agent_yes_shares: float = 0.0
    agent_no_shares: float = 0.0
    agent_cost_basis: float = 0.0
    agent_realized_pnl: float = 0.0


class ObservationBuilder:
    """Converts MarketState to numpy observation vectors."""

    # Fixed signal keys in canonical order (must match obs space size)
    SIGNAL_KEYS = [
        "best_bid", "best_ask", "spread", "book_depth_bid",
        "book_depth_ask", "last_trade_price", "volume_24hr", "liquidity",
    ]

    # Normalization constants for features with large ranges
    _BOOK_DEPTH_SCALE = 10_000.0  # typical max book depth in USDC
    _TIME_REMAINING_SCALE = 3600.0  # max time remaining (1 hour)
    _VOLUME_SCALE = 1_000_000.0  # typical max 24h volume
    _LIQUIDITY_SCALE = 100_000.0  # typical max liquidity

    @staticmethod
    def flatten(state: MarketState) -> "numpy.ndarray":
        import numpy as np

        # All prices/spreads already in [0, 1]
        # Normalize book_depth and time_remaining to ~[0, 1]
        norm_depth = min(state.book_depth / ObservationBuilder._BOOK_DEPTH_SCALE, 1.0)
        norm_time_rem = min(state.time_remaining / ObservationBuilder._TIME_REMAINING_SCALE, 1.0)

        base = [
            state.yes_price,
            state.no_price,
            state.yes_bid,
            state.yes_ask,
            state.no_bid,
            state.no_ask,
            state.spread * 10.0,  # scale spread up (typically 0.01-0.03)
            norm_depth,
            state.time_elapsed,
            norm_time_rem,
        ]

        # Normalize signal values
        signals = state.signals
        signal_values = [
            signals.get("best_bid", 0.0),  # [0, 1]
            signals.get("best_ask", 0.0),  # [0, 1]
            signals.get("spread", 0.0) * 10.0,  # scale up
            min(signals.get("book_depth_bid", 0.0) / ObservationBuilder._BOOK_DEPTH_SCALE, 1.0),
            min(signals.get("book_depth_ask", 0.0) / ObservationBuilder._BOOK_DEPTH_SCALE, 1.0),
            signals.get("last_trade_price", 0.0),  # [0, 1]
            min(signals.get("volume_24hr", 0.0) / ObservationBuilder._VOLUME_SCALE, 1.0),
            min(signals.get("liquidity", 0.0) / ObservationBuilder._LIQUIDITY_SCALE, 1.0),
        ]
        return np.array(base + signal_values, dtype=np.float32)
