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


class ObservationBuilder:
    """Converts MarketState to numpy observation vectors."""

    # Fixed signal keys in canonical order (must match obs space size)
    SIGNAL_KEYS = [
        "best_bid", "best_ask", "spread", "book_depth_bid",
        "book_depth_ask", "last_trade_price", "volume_24hr", "liquidity",
    ]

    @staticmethod
    def flatten(state: MarketState) -> "numpy.ndarray":
        import numpy as np

        base = [
            state.yes_price,
            state.no_price,
            state.yes_bid,
            state.yes_ask,
            state.no_bid,
            state.no_ask,
            state.spread,
            state.book_depth,
            state.time_elapsed,
            state.time_remaining,
        ]
        # Use fixed signal keys to guarantee consistent vector size
        signal_values = [
            state.signals.get(k, 0.0) for k in ObservationBuilder.SIGNAL_KEYS
        ]
        return np.array(base + signal_values, dtype=np.float32)
