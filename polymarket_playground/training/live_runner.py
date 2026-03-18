"""LiveRunner — real-time orchestration loop for live/paper trading."""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field

from polymarket_playground.agents.base_agent import BaseAgent
from polymarket_playground.core.portfolio import Portfolio
from polymarket_playground.data.session_log import SessionLog
from polymarket_playground.execution.base_executor import BaseExecutor
from polymarket_playground.execution.risk_gate import RiskConfig, RiskGate
from polymarket_playground.markets.polymarket import PolymarketAdapter

logger = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    """Configuration for a live trading session."""

    market_ids: list[str] = field(default_factory=list)
    interval_seconds: int = 60
    starting_capital: float = 100.0
    risk: RiskConfig = field(default_factory=RiskConfig)
    mode: str = "paper"  # "paper" | "live" | "dry-run"
    max_ticks: int = 0  # 0 = unlimited


class LiveRunner:
    """Runs an agent against live market data in real-time.

    Modes:
        - paper: real market data, simulated execution (PaperExecutor)
        - live: real market data, real order placement (LiveExecutor)
        - dry-run: real market data, logs actions but executes nothing
    """

    def __init__(
        self,
        agent: BaseAgent,
        executor: BaseExecutor,
        adapter: PolymarketAdapter,
        config: LiveConfig,
        session_log: SessionLog | None = None,
        signal_provider: object | None = None,
    ):
        self.agent = agent
        self.executor = executor
        self.adapter = adapter
        self.config = config
        self.portfolio = Portfolio(starting_capital=config.starting_capital)
        self.risk_gate = RiskGate(config.risk)
        self.session_log = session_log or SessionLog()
        self.signal_provider = signal_provider
        self._running = False
        self._tick_count = 0

    def run(self) -> dict:
        """Main loop. Blocks until stopped or max_ticks reached.

        Returns a summary dict with session results.
        """
        self._running = True
        self.risk_gate.init_session(self.portfolio)

        # Handle graceful shutdown
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)

        session_id = self.session_log.start_session(
            agent_type=type(self.agent).__name__,
            mode=self.config.mode,
            starting_capital=self.config.starting_capital,
            config={
                "market_ids": self.config.market_ids,
                "interval_seconds": self.config.interval_seconds,
                "risk": {
                    "max_position_usd": self.config.risk.max_position_usd,
                    "max_total_exposure_usd": self.config.risk.max_total_exposure_usd,
                    "max_loss_per_session_usd": self.config.risk.max_loss_per_session_usd,
                },
            },
        )

        logger.info(
            "LiveRunner started: mode=%s markets=%d interval=%ds capital=%.2f",
            self.config.mode, len(self.config.market_ids),
            self.config.interval_seconds, self.config.starting_capital,
        )

        try:
            while self._running:
                if self.risk_gate.is_killed:
                    logger.warning("Kill switch triggered, stopping.")
                    break
                if self.config.max_ticks and self._tick_count >= self.config.max_ticks:
                    logger.info("Max ticks reached (%d), stopping.", self.config.max_ticks)
                    break

                self._tick()

                if self._running:
                    time.sleep(self.config.interval_seconds)
        finally:
            signal.signal(signal.SIGINT, original_sigint)

        # End session
        prices = self._get_current_prices()
        total_pnl = self.portfolio.total_pnl(prices)
        self.session_log.end_session(
            final_capital=self.portfolio.available_capital,
            total_pnl=total_pnl,
            markets_traded=len(self.portfolio.active_markets),
            total_ticks=self._tick_count,
            risk_interventions=len(self.risk_gate.interventions),
        )

        summary = {
            "session_id": session_id,
            "ticks": self._tick_count,
            "total_pnl": total_pnl,
            "final_capital": self.portfolio.available_capital,
            "active_positions": len(self.portfolio.active_markets),
            "risk_interventions": len(self.risk_gate.interventions),
            "kill_switch": self.risk_gate.is_killed,
        }
        logger.info("Session complete: %s", summary)
        return summary

    def stop(self):
        """Signal the runner to stop after the current tick."""
        self._running = False

    def _tick(self):
        """Process one tick across all markets."""
        for market_id in self.config.market_ids:
            try:
                self._tick_market(market_id)
            except Exception:
                logger.exception("Error processing market %s", market_id)
        self._tick_count += 1

    def _tick_market(self, market_id: str):
        """Process one tick for a single market."""
        # Fetch fresh state
        try:
            state = self.adapter.build_state(market_id)
        except Exception:
            logger.exception("Failed to fetch state for %s", market_id)
            return

        # Enrich signals from signal provider (if any)
        if self.signal_provider is not None:
            try:
                enriched = self.signal_provider.get_signals(
                    market_id, state.question, state.yes_price
                )
                state.signals.update(enriched)
            except Exception:
                logger.warning("Signal provider failed for %s", market_id, exc_info=True)

        # Inject position info into state
        position = self.portfolio.get_position(market_id)
        state.agent_yes_shares = position.yes_shares
        state.agent_no_shares = position.no_shares
        state.agent_cost_basis = position.cost_basis
        state.agent_realized_pnl = position.realized_pnl

        # Agent decides
        action = self.agent.act(state)

        # Risk check
        approved_action = self.risk_gate.check(
            action, state, position, self.portfolio,
        )

        risk_blocked = approved_action is None
        risk_reason = None
        if risk_blocked:
            risk_reason = (
                self.risk_gate.interventions[-1]["reason"]
                if self.risk_gate.interventions else "unknown"
            )

        fill = None
        fill_dict = {}

        if approved_action and not approved_action.is_hold:
            if self.config.mode == "dry-run":
                logger.info(
                    "DRY-RUN: %s on %s (size=%.2f)",
                    approved_action.direction_name, market_id, approved_action.size,
                )
            else:
                fill = self.executor.execute(approved_action, state, position)
                if fill:
                    self.portfolio.update(market_id, fill)
                    fill_dict = {
                        "direction": fill.direction,
                        "price": fill.price,
                        "size": fill.size,
                        "fees": fill.fees,
                    }

        # Handle resolved markets
        if state.is_resolved and state.resolution is not None:
            self.portfolio.settle(market_id, state.resolution)

        # Log
        self.session_log.log_tick(
            market_id=market_id,
            state={
                "yes_price": state.yes_price,
                "no_price": state.no_price,
                "spread": state.spread,
            },
            action={
                "direction": action.direction_name,
                "size": action.size,
            },
            fill=fill_dict,
            position={
                "yes_shares": position.yes_shares,
                "no_shares": position.no_shares,
                "cost_basis": position.cost_basis,
            },
            portfolio={
                "available_capital": self.portfolio.available_capital,
                "total_exposure": self.portfolio.total_exposure(),
            },
            risk_blocked=risk_blocked,
            risk_reason=risk_reason,
        )

    def _get_current_prices(self) -> dict[str, float]:
        """Fetch current YES prices for all active markets."""
        prices = {}
        for market_id in self.portfolio.active_markets:
            try:
                state = self.adapter.build_state(market_id)
                prices[market_id] = state.yes_price
            except Exception:
                prices[market_id] = 0.5
        return prices

    def _handle_signal(self, signum, frame):
        logger.info("Received signal %d, stopping gracefully...", signum)
        self._running = False
