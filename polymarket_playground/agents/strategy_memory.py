"""Strategy memory — persistent learning across training batches.

Stores accumulated trading insights in a markdown file that gets
injected into the Claude agent's system prompt. Human-readable,
inspectable, and editable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from polymarket_playground.agents.claude_helper import (
    MODEL_ANALYSIS,
    call_claude,
    get_client,
)
from polymarket_playground.core.types import EpisodeResult

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = Path("strategy_memory.md")
MAX_MEMORY_TOKENS = 2000  # approximate token budget for system prompt injection


class StrategyMemory:
    """Persistent strategy memory for Claude agent learning.

    After each training batch, summarizes what worked and what didn't,
    then updates a markdown file. On the next batch, the accumulated
    insights are injected into Claude's system prompt.

    Usage::

        memory = StrategyMemory()

        # At start of training batch
        strategy_context = memory.load()
        # ... inject into Claude's system prompt ...

        # After training batch
        memory.update(results)
    """

    SUMMARIZE_PROMPT = """\
You are a trading strategy analyst. Review the batch results and the
existing strategy document, then produce an updated strategy document.

Rules:
- Keep the document under 800 words
- Focus on actionable patterns (what market types, price levels, timing)
- Remove insights that are contradicted by new results
- Add new insights from this batch
- Be specific: "buy YES when price < 0.30 in election markets" not "look for undervalued markets"
- Track win rate by strategy type if patterns emerge

Output ONLY the updated strategy document as plain text (no code fences, no JSON).
"""

    def __init__(self, path: str | Path = DEFAULT_MEMORY_PATH) -> None:
        self.path = Path(path)

    def load(self) -> str:
        """Load the current strategy memory.

        Returns the contents of the strategy file, or empty string
        if no memory exists yet (first run).
        """
        if not self.path.exists():
            return ""
        try:
            content = self.path.read_text(encoding="utf-8").strip()
            # Truncate if too long
            if len(content) > MAX_MEMORY_TOKENS * 4:  # rough char estimate
                content = content[: MAX_MEMORY_TOKENS * 4]
                logger.warning("Strategy memory truncated to fit token budget")
            return content
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to read strategy memory, starting fresh: %s", exc)
            return ""

    def update(self, results: list[EpisodeResult]) -> None:
        """Update strategy memory with insights from a training batch.

        Uses Claude (Haiku) to summarize batch results and merge with
        existing strategy document.
        """
        if not results:
            return

        client = get_client()
        if client is None:
            logger.warning("No API key — skipping strategy memory update")
            return

        existing = self.load()
        batch_summary = self._summarize_batch(results)

        prompt = f"EXISTING STRATEGY DOCUMENT:\n{existing or '(empty — first batch)'}\n\n"
        prompt += f"NEW BATCH RESULTS:\n{batch_summary}\n\n"
        prompt += "Produce the updated strategy document."

        updated = call_claude(
            [{"role": "user", "content": prompt}],
            system=self.SUMMARIZE_PROMPT,
            model=MODEL_ANALYSIS,
            client=client,
            max_tokens=1500,
        )

        if updated is None:
            logger.warning("Strategy summarization failed, keeping existing memory")
            return

        try:
            self.path.write_text(updated.strip(), encoding="utf-8")
            logger.info(
                "Strategy memory updated (%d chars, %d episodes analyzed)",
                len(updated),
                len(results),
            )
        except OSError as exc:
            logger.error("Failed to write strategy memory: %s", exc)

    def clear(self) -> None:
        """Remove the strategy memory file."""
        if self.path.exists():
            self.path.unlink()
            logger.info("Strategy memory cleared")

    @staticmethod
    def _summarize_batch(results: list[EpisodeResult]) -> str:
        """Build a human-readable summary of a training batch."""
        n = len(results)
        rewards = [r.total_reward for r in results]
        wins = sum(1 for r in rewards if r > 0)
        losses = sum(1 for r in rewards if r < 0)
        mean_reward = sum(rewards) / n if n else 0
        best = max(rewards) if rewards else 0
        worst = min(rewards) if rewards else 0

        # Compute per-episode action distributions
        total_holds = 0
        total_buys_yes = 0
        total_buys_no = 0
        total_closes = 0
        for result in results:
            for action in result.actions:
                d = action.direction if hasattr(action, "direction") else 0
                if d == 0:
                    total_holds += 1
                elif d == 1:
                    total_buys_yes += 1
                elif d == 2:
                    total_buys_no += 1
                elif d == 3:
                    total_closes += 1

        total_actions = total_holds + total_buys_yes + total_buys_no + total_closes

        lines = [
            f"Batch: {n} episodes",
            f"Win/Loss: {wins}W / {losses}L ({(wins / n) if n else 0:.0%} win rate)",
            f"Mean reward: {mean_reward:+.6f}",
            f"Best episode: {best:+.6f}",
            f"Worst episode: {worst:+.6f}",
        ]

        if total_actions > 0:
            lines.append(
                f"Actions: {total_holds} holds, {total_buys_yes} buy_yes, "
                f"{total_buys_no} buy_no, {total_closes} closes"
            )

        # Winning vs losing episode characteristics
        winning = [r for r in results if r.total_reward > 0]
        losing = [r for r in results if r.total_reward < 0]
        if winning:
            avg_win_steps = sum(r.n_steps for r in winning) / len(winning)
            lines.append(f"Avg steps in winning episodes: {avg_win_steps:.0f}")
        if losing:
            avg_loss_steps = sum(r.n_steps for r in losing) / len(losing)
            lines.append(f"Avg steps in losing episodes: {avg_loss_steps:.0f}")

        return "\n".join(lines)
