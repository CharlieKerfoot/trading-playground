"""RiskGate — enforces safety limits before orders reach the market."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from polymarket_playground.core.action import Action
from polymarket_playground.core.portfolio import Portfolio
from polymarket_playground.core.position import Position
from polymarket_playground.core.state import MarketState

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk control parameters."""

    max_position_usd: float = 50.0
    max_total_exposure_usd: float = 200.0
    max_loss_per_market_usd: float = 25.0
    max_loss_per_session_usd: float = 100.0
    max_orders_per_minute: int = 10
    cooldown_after_close_seconds: float = 60.0


@dataclass
class RiskState:
    """Mutable state tracked by the risk gate."""

    order_timestamps: list[float] = field(default_factory=list)
    close_timestamps: dict[str, float] = field(default_factory=dict)
    session_start_capital: float = 0.0
    kill_switch: bool = False
    interventions: list[dict] = field(default_factory=list)


class RiskGate:
    """Validates actions against risk limits before execution."""

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self._state = RiskState()

    def init_session(self, portfolio: Portfolio) -> None:
        """Call at the start of a live session."""
        self._state = RiskState(session_start_capital=portfolio.starting_capital)

    def check(
        self,
        action: Action,
        state: MarketState,
        position: Position,
        portfolio: Portfolio,
    ) -> Action | None:
        """Check an action against risk limits.

        Returns:
            The action if allowed, a modified action (e.g. forced close),
            or None if blocked.
        """
        if action.is_hold:
            return action

        if self._state.kill_switch:
            self._log_intervention(state.market_id, action, "kill_switch")
            return None

        # Session loss check
        prices = {mid: 0.5 for mid in portfolio.active_markets}
        prices[state.market_id] = state.yes_price
        session_pnl = portfolio.total_pnl(prices)
        if session_pnl < -self.config.max_loss_per_session_usd:
            self._state.kill_switch = True
            self._log_intervention(
                state.market_id, action,
                f"session_loss_limit (pnl={session_pnl:+.2f})",
            )
            logger.warning("KILL SWITCH: session loss %.2f exceeds limit", session_pnl)
            return None

        # Per-market loss check — force close if losing too much
        if not position.is_flat:
            market_pnl = position.close_value(state.yes_price) - position.cost_basis
            if market_pnl < -self.config.max_loss_per_market_usd:
                self._log_intervention(
                    state.market_id, action,
                    f"market_stop_loss (pnl={market_pnl:+.2f})",
                )
                logger.warning(
                    "Stop loss on %s: pnl=%.2f, forcing close",
                    state.market_id, market_pnl,
                )
                now = time.time()
                self._state.order_timestamps = [
                    t for t in self._state.order_timestamps if now - t < 60
                ]
                if len(self._state.order_timestamps) >= self.config.max_orders_per_minute:
                    self._log_intervention(state.market_id, action, "rate_limit")
                    return None
                self._state.order_timestamps.append(now)
                self._state.close_timestamps[state.market_id] = now
                return Action(direction=3, size=1.0, reasoning="risk_gate: stop loss")

        # For buy actions, check position and exposure limits
        if action.direction in (1, 2):  # buy_yes or buy_no
            # Position size limit
            current_cost = position.cost_basis
            new_cost = action.size * (
                state.yes_ask if action.direction == 1 else state.no_ask
            )
            if current_cost + new_cost > self.config.max_position_usd:
                self._log_intervention(
                    state.market_id, action,
                    f"position_limit (current={current_cost:.2f} + new={new_cost:.2f})",
                )
                return None

            # Total exposure limit
            total_exposure = portfolio.total_exposure()
            if total_exposure + new_cost > self.config.max_total_exposure_usd:
                self._log_intervention(
                    state.market_id, action,
                    f"exposure_limit (total={total_exposure:.2f} + new={new_cost:.2f})",
                )
                return None

            # Capital check
            if new_cost > portfolio.available_capital:
                self._log_intervention(
                    state.market_id, action,
                    f"insufficient_capital (need={new_cost:.2f}, have={portfolio.available_capital:.2f})",
                )
                return None

            # Cooldown after close
            last_close = self._state.close_timestamps.get(state.market_id, 0)
            if time.time() - last_close < self.config.cooldown_after_close_seconds:
                self._log_intervention(state.market_id, action, "cooldown_after_close")
                return None

        # Rate limiting
        now = time.time()
        self._state.order_timestamps = [
            t for t in self._state.order_timestamps if now - t < 60
        ]
        if len(self._state.order_timestamps) >= self.config.max_orders_per_minute:
            self._log_intervention(state.market_id, action, "rate_limit")
            return None
        self._state.order_timestamps.append(now)

        # Close action — record timestamp for cooldown
        if action.direction == 3:
            self._state.close_timestamps[state.market_id] = now

        return action

    def _log_intervention(self, market_id: str, action: Action, reason: str):
        entry = {
            "timestamp": time.time(),
            "market_id": market_id,
            "action": action.direction_name,
            "reason": reason,
        }
        self._state.interventions.append(entry)
        logger.info("RiskGate blocked: %s", entry)

    @property
    def interventions(self) -> list[dict]:
        return self._state.interventions

    @property
    def is_killed(self) -> bool:
        return self._state.kill_switch
