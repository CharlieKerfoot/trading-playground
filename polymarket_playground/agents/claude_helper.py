"""Shared Claude API helper — single point for all Claude calls."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone

import anthropic

logger = logging.getLogger(__name__)

# Model defaults per use case
MODEL_TRADING = "claude-sonnet-4-6"
MODEL_ANALYSIS = "claude-haiku-4-5"

# Pricing per million tokens (USD)
_PRICING = {
    MODEL_ANALYSIS: {"input": 0.80, "output": 4.00},
    MODEL_TRADING: {"input": 3.00, "output": 15.00},
}


class CostTracker:
    """Accumulates API token usage and costs, persists to SQLite."""

    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL
            )"""
        )
        conn.commit()
        conn.close()

    def log_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Record a single API call's token usage."""
        pricing = _PRICING.get(model, {"input": 3.00, "output": 15.00})
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.call_count += 1

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO api_usage (timestamp, model, input_tokens, output_tokens, cost_usd) "
                "VALUES (?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), model, input_tokens, output_tokens, cost),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("Failed to persist API usage: %s", exc)

    def summary(self) -> str:
        """Return a human-readable cost summary for the current session."""
        if self.call_count == 0:
            return "API cost: $0.00 (0 calls)"
        in_k = self.total_input_tokens / 1000
        out_k = self.total_output_tokens / 1000
        return (
            f"API cost: ${self.total_cost_usd:.2f} "
            f"({self.call_count} calls, input: {in_k:.1f}k tokens, output: {out_k:.1f}k tokens)"
        )

    def reset(self):
        """Reset session counters (does not affect persisted data)."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0


# Module-level singleton
cost_tracker = CostTracker()


def get_client(api_key: str | None = None) -> anthropic.Anthropic | None:
    """Return an Anthropic client, or None if no API key is available."""
    resolved = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved:
        logger.warning("No ANTHROPIC_API_KEY found.")
        return None
    return anthropic.Anthropic(api_key=resolved)


def call_claude(
    messages: list[dict[str, str]],
    *,
    system: str = "",
    model: str = MODEL_ANALYSIS,
    max_tokens: int = 1024,
    client: anthropic.Anthropic | None = None,
    api_key: str | None = None,
    max_retries: int = 2,
) -> str | None:
    """Call Claude and return the text response.

    Handles retries on rate limits and timeouts. Returns None on failure.
    """
    if client is None:
        client = get_client(api_key)
    if client is None:
        return None

    for attempt in range(max_retries + 1):
        try:
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if system:
                kwargs["system"] = system
            response = client.messages.create(**kwargs)
            cost_tracker.log_usage(
                model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            return response.content[0].text

        except anthropic.RateLimitError:
            wait = 2 ** attempt
            logger.warning("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries + 1)
            time.sleep(wait)

        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            wait = 2 ** attempt
            logger.warning("API connection issue: %s, retrying in %ds", exc, wait)
            time.sleep(wait)

        except anthropic.AuthenticationError as exc:
            logger.error("Authentication failed: %s", exc)
            return None

        except anthropic.APIError as exc:
            logger.error("Claude API error: %s", exc)
            return None

    logger.error("All %d retries exhausted", max_retries + 1)
    return None


def call_claude_json(
    messages: list[dict[str, str]],
    *,
    system: str = "",
    model: str = MODEL_ANALYSIS,
    max_tokens: int = 1024,
    client: anthropic.Anthropic | None = None,
    api_key: str | None = None,
) -> dict | None:
    """Call Claude and parse the response as JSON.

    Strips markdown code fences if present. Returns None on failure.
    """
    raw = call_claude(
        messages,
        system=system,
        model=model,
        max_tokens=max_tokens,
        client=client,
        api_key=api_key,
    )
    if raw is None:
        return None

    # Strip markdown code fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse Claude JSON response: %s", exc)
        return None
