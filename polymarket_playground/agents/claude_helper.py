"""Shared Claude API helper — single point for all Claude calls."""

from __future__ import annotations

import json
import logging
import os
import time

import anthropic

logger = logging.getLogger(__name__)

# Model defaults per use case
MODEL_TRADING = "claude-sonnet-4-6"
MODEL_ANALYSIS = "claude-haiku-4-5"


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
