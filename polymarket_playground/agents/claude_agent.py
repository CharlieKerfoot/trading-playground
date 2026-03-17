"""Claude API agent with multi-turn conversation history."""

from __future__ import annotations

import json
import logging
import os

import anthropic

from polymarket_playground.core.action import Action
from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import EpisodeResult

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a prediction market trader on Polymarket. You will be shown the current \
state of a prediction market contract — including prices, spreads, external \
signals, your current position, and time remaining — and must decide what to do.

Your goal is to maximize total P&L across the episode. Think about:
- Edge: is the market mispriced relative to signals?
- Position management: should you add, hold, or reduce exposure?
- Timing: how much time remains before resolution?
- Risk: what is your downside if you are wrong?

Respond ONLY with valid JSON (no markdown, no explanation outside the JSON):
{
    "action": "hold" | "buy_yes" | "buy_no" | "close",
    "size": <float 0.0-1.0>,
    "reasoning": "<brief explanation>"
}

Actions:
- "hold": do nothing this step
- "buy_yes": buy YES shares (bet on the event happening)
- "buy_no": buy NO shares (bet against the event)
- "close": close your current position
"""


def _build_context(state: MarketState) -> str:
    """Build a human-readable context string from a MarketState."""
    signals = state.signals
    return (
        f"Polymarket — {state.question}\n"
        f"YES {state.yes_price:.3f} / NO {state.no_price:.3f} | "
        f"Spread {state.spread:.4f}\n"
        f"Best bid: {signals.get('best_bid', 0):.3f} | "
        f"Best ask: {signals.get('best_ask', 0):.3f}\n"
        f"Book depth: bid ${signals.get('book_depth_bid', 0):,.0f} / "
        f"ask ${signals.get('book_depth_ask', 0):,.0f}\n"
        f"24h volume: ${signals.get('volume_24hr', 0):,.0f} | "
        f"Liquidity: ${signals.get('liquidity', 0):,.0f}\n"
        f"Time elapsed: {state.time_elapsed:.1%} | "
        f"Time remaining: {state.time_remaining:.0f}s\n"
    )


class ClaudeAgent(BaseAgent):
    """Claude agent that reasons over market context.

    Builds context directly from MarketState, then calls the Anthropic API
    and parses the structured JSON response into an Action.

    Conversation history is maintained within an episode so Claude can reason
    about its own prior actions. History is cleared on ``on_reset()``.
    A sliding window keeps token usage bounded.
    """

    MAX_HISTORY_MESSAGES = 20  # keep last N messages to bound token usage

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if resolved_key:
            self.client = anthropic.Anthropic(api_key=resolved_key)
        else:
            logger.warning("No ANTHROPIC_API_KEY found. ClaudeAgent will return hold actions.")
            self.client = None
        self.history: list[dict[str, str]] = []
        self._step: int = 0
        self._cumulative_reward: float = 0.0

    def act(self, state: MarketState) -> Action:
        self._step += 1
        context = _build_context(state)

        # Use actual position from the environment (not local tracking)
        unrealized_pnl = (
            state.agent_yes_shares * state.yes_price
            + state.agent_no_shares * (1.0 - state.yes_price)
            - state.agent_cost_basis
        )
        total_pnl = unrealized_pnl + state.agent_realized_pnl
        context += (
            f"\n--- Your Position (Step {self._step}) ---\n"
            f"YES shares: {state.agent_yes_shares:.2f} | "
            f"NO shares: {state.agent_no_shares:.2f}\n"
            f"Cost basis: {state.agent_cost_basis:.4f}\n"
            f"Unrealized P&L: {unrealized_pnl:+.4f} | "
            f"Realized P&L: {state.agent_realized_pnl:+.4f}\n"
            f"Total P&L: {total_pnl:+.4f}\n"
        )

        self.history.append({"role": "user", "content": context})

        # Sliding window to bound token usage
        if len(self.history) > self.MAX_HISTORY_MESSAGES:
            self.history = self.history[-self.MAX_HISTORY_MESSAGES:]

        if self.client is None:
            raw = '{"action": "hold", "size": 0.0, "reasoning": "no API key"}'
            self.history.append({"role": "assistant", "content": raw})
            return Action(direction=0, size=0.0, reasoning="no API key")

        try:
            response = self.client.messages.create(
                model=self.model,
                system=SYSTEM_PROMPT,
                messages=self.history,
                max_tokens=512,
            )
            raw = response.content[0].text
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            parsed = json.loads(cleaned)
            action = Action.from_dict(parsed)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, anthropic.AuthenticationError, anthropic.APIError) as exc:
            logger.warning("Failed to parse Claude response, defaulting to hold: %s", exc)
            raw = '{"action": "hold", "size": 0.0, "reasoning": "parse error fallback"}'
            action = Action(direction=0, size=0.0, reasoning="parse error fallback")

        self.history.append({"role": "assistant", "content": raw})
        return action

    def on_reset(self, state: MarketState) -> None:
        self.history = []
        self._step = 0
        self._cumulative_reward = 0.0

    def on_episode_end(self, result: EpisodeResult) -> None:
        self._cumulative_reward = result.total_reward
