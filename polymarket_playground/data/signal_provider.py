"""Signal providers — enrich MarketState with information signals.

Two modes:
- Historical: Claude analyzes the market question + price pattern (no web search)
- Live: Web search + Claude analysis for real-time information edge
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from polymarket_playground.agents.claude_helper import (
    MODEL_ANALYSIS,
    call_claude_json,
    get_client,
)

logger = logging.getLogger(__name__)


class SignalProvider(ABC):
    """Abstract signal provider interface."""

    @abstractmethod
    def get_signals(self, market_id: str, question: str, yes_price: float) -> dict[str, float]:
        """Return enriched signals for a market.

        Returns a dict of signal_name -> float value, merged into
        MarketState.signals alongside price-derived signals.
        """
        ...


class HistoricalSignalProvider(SignalProvider):
    """Analyzes market question + price pattern using Claude (no web search).

    Used for training on resolved markets where web search wouldn't return
    relevant results. Claude reasons about:
    - Market type (election, sports, crypto, policy, etc.)
    - Implied probability vs price
    - Question structure and resolution criteria
    """

    SYSTEM_PROMPT = """\
You are analyzing a prediction market for a trading agent's training data.
Given the market question and current price, provide a structured analysis.

Respond ONLY with valid JSON (no markdown):
{
    "market_category_score": <float 0-1, how well-defined the category is>,
    "resolution_clarity": <float 0-1, how clear the resolution criteria are>,
    "sentiment_score": <float -1 to 1, implied sentiment from question framing>,
    "volatility_expectation": <float 0-1, expected price volatility>,
    "information_density": <float 0-1, how much public info likely exists>
}
"""

    def __init__(self, cache_path: str | Path = "signal_cache.db") -> None:
        self._client = get_client()
        self._cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._cache_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS signal_cache ("
            "  cache_key TEXT PRIMARY KEY,"
            "  signals TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    def get_signals(self, market_id: str, question: str, yes_price: float) -> dict[str, float]:
        cache_key = self._cache_key(market_id, question)

        # Check cache
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Generate signals via Claude
        signals = self._analyze(question, yes_price)

        # Cache the result
        self._put_cached(cache_key, signals)
        return signals

    def _analyze(self, question: str, yes_price: float) -> dict[str, float]:
        if self._client is None:
            return self._default_signals()

        prompt = (
            f"Market question: {question}\n"
            f"Current YES price: {yes_price:.3f} (implied probability: {yes_price:.1%})\n"
        )

        result = call_claude_json(
            [{"role": "user", "content": prompt}],
            system=self.SYSTEM_PROMPT,
            model=MODEL_ANALYSIS,
            client=self._client,
            max_tokens=256,
        )

        if result is None:
            return self._default_signals()

        return {
            "market_category_score": float(result.get("market_category_score", 0.5)),
            "resolution_clarity": float(result.get("resolution_clarity", 0.5)),
            "sentiment_score": max(-1.0, min(1.0, float(result.get("sentiment_score", 0.0)))),
            "volatility_expectation": float(result.get("volatility_expectation", 0.5)),
            "information_density": float(result.get("information_density", 0.5)),
        }

    @staticmethod
    def _default_signals() -> dict[str, float]:
        return {
            "market_category_score": 0.5,
            "resolution_clarity": 0.5,
            "sentiment_score": 0.0,
            "volatility_expectation": 0.5,
            "information_density": 0.5,
        }

    def _cache_key(self, market_id: str, question: str) -> str:
        raw = f"{market_id}:{question}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> dict[str, float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT signals FROM signal_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def _put_cached(self, key: str, signals: dict[str, float]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO signal_cache (cache_key, signals) VALUES (?, ?)",
                (key, json.dumps(signals)),
            )
            self._conn.commit()


class LiveSignalProvider(SignalProvider):
    """Web search + Claude analysis for live/paper trading.

    Fetches recent news about the market question, then uses Claude to
    score relevance and sentiment. Provides real information edge.
    """

    SYSTEM_PROMPT = """\
You are analyzing news and information about a prediction market.
Given the market question, current price, and search results, assess whether
the market is mispriced based on available information.

Respond ONLY with valid JSON (no markdown):
{
    "sentiment_score": <float -1 to 1, -1=strongly NO, 1=strongly YES>,
    "information_edge": <float 0-1, how much the news disagrees with the price>,
    "news_recency": <float 0-1, how recent/relevant the news is>,
    "confidence": <float 0-1, your confidence in the analysis>,
    "key_signal": "<one sentence: most important finding>"
}
"""

    def __init__(
        self,
        search_api_key: str | None = None,
        cache_ttl_seconds: int = 300,
        cache_path: str | Path = "signal_cache.db",
    ) -> None:
        import os

        self._client = get_client()
        self._search_api_key = search_api_key or os.environ.get("TAVILY_API_KEY", "")
        self._cache_ttl = cache_ttl_seconds
        self._lock = threading.Lock()
        self._cache_path = Path(cache_path)
        self._conn = sqlite3.connect(str(self._cache_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS live_signal_cache ("
            "  cache_key TEXT PRIMARY KEY,"
            "  signals TEXT NOT NULL,"
            "  created_at INTEGER NOT NULL"
            ")"
        )
        self._conn.commit()

    def get_signals(self, market_id: str, question: str, yes_price: float) -> dict[str, float]:
        import time

        cache_key = hashlib.sha256(f"live:{market_id}".encode()).hexdigest()[:16]
        now = int(time.time())

        # Check TTL cache
        cached = self._get_cached_ttl(cache_key, now)
        if cached is not None:
            return cached

        # Search for news
        search_results = self._web_search(question)

        # Analyze with Claude
        signals = self._analyze(question, yes_price, search_results)

        # Cache with TTL
        self._put_cached_ttl(cache_key, signals, now)
        return signals

    def _web_search(self, question: str) -> str:
        """Search the web for information about the market question."""
        if not self._search_api_key:
            logger.warning("No TAVILY_API_KEY set, skipping web search")
            return "No search results available."

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self._search_api_key)
            response = client.search(
                query=question,
                search_depth="basic",
                max_results=5,
            )
            results = response.get("results", [])
            if not results:
                return "No relevant search results found."

            # Format results for Claude — sanitize to reduce injection risk
            formatted = []
            for r in results[:5]:
                title = (r.get("title", "") or "")[:200]
                content = (r.get("content", "") or "")[:500]
                formatted.append(f"- {title}: {content}")
            return "\n".join(formatted)

        except ImportError:
            logger.error("tavily package not installed. Run: uv add tavily-python")
            return "Search unavailable (tavily not installed)."
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return "Search failed."

    def _analyze(self, question: str, yes_price: float, search_results: str) -> dict[str, float]:
        if self._client is None:
            return self._default_signals()

        prompt = (
            f"Market question: {question}\n"
            f"Current YES price: {yes_price:.3f} (implied probability: {yes_price:.1%})\n\n"
            f"Recent news and information:\n{search_results}\n\n"
            f"Analyze whether the market price reflects the available information."
        )

        result = call_claude_json(
            [{"role": "user", "content": prompt}],
            system=self.SYSTEM_PROMPT,
            model=MODEL_ANALYSIS,
            client=self._client,
            max_tokens=512,
        )

        if result is None:
            return self._default_signals()

        return {
            "sentiment_score": max(-1.0, min(1.0, float(result.get("sentiment_score", 0.0)))),
            "information_edge": max(0.0, min(1.0, float(result.get("information_edge", 0.0)))),
            "news_recency": max(0.0, min(1.0, float(result.get("news_recency", 0.0)))),
            "signal_confidence": max(0.0, min(1.0, float(result.get("confidence", 0.0)))),
        }

    @staticmethod
    def _default_signals() -> dict[str, float]:
        return {
            "sentiment_score": 0.0,
            "information_edge": 0.0,
            "news_recency": 0.0,
            "signal_confidence": 0.0,
        }

    def _get_cached_ttl(self, key: str, now: int) -> dict[str, float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT signals, created_at FROM live_signal_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row and (now - row[1]) < self._cache_ttl:
            return json.loads(row[0])
        return None

    def _put_cached_ttl(self, key: str, signals: dict[str, float], now: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO live_signal_cache (cache_key, signals, created_at) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(signals), now),
            )
            self._conn.commit()
