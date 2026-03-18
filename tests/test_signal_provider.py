"""Tests for signal providers."""

import tempfile
from unittest.mock import patch

from polymarket_playground.data.signal_provider import (
    HistoricalSignalProvider,
    LiveSignalProvider,
)


def test_historical_default_signals():
    """When no API key is available, returns default neutral signals."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("polymarket_playground.data.signal_provider.get_client", return_value=None):
            provider = HistoricalSignalProvider(cache_path=tmp.name)
            signals = provider.get_signals("market_1", "Will X happen?", 0.5)

    assert "sentiment_score" in signals
    assert "resolution_clarity" in signals
    assert signals["sentiment_score"] == 0.0
    assert signals["resolution_clarity"] == 0.5


def test_historical_caches_results():
    """Second call should return cached result without calling Claude."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("polymarket_playground.data.signal_provider.get_client", return_value=None):
            provider = HistoricalSignalProvider(cache_path=tmp.name)

            s1 = provider.get_signals("m1", "question?", 0.5)
            s2 = provider.get_signals("m1", "question?", 0.5)

    assert s1 == s2


def test_historical_with_claude_response():
    """When Claude responds, parse the JSON signals."""
    mock_response = {
        "market_category_score": 0.8,
        "resolution_clarity": 0.9,
        "sentiment_score": 0.3,
        "volatility_expectation": 0.6,
        "information_density": 0.7,
    }

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("polymarket_playground.data.signal_provider.get_client") as mock_client, \
             patch("polymarket_playground.data.signal_provider.call_claude_json", return_value=mock_response):
            mock_client.return_value = "fake_client"
            provider = HistoricalSignalProvider(cache_path=tmp.name)
            signals = provider.get_signals("m2", "Will BTC hit 100k?", 0.65)

    assert signals["market_category_score"] == 0.8
    assert signals["sentiment_score"] == 0.3


def test_live_default_signals_no_search_key():
    """Without Tavily key, should still return default signals."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("polymarket_playground.data.signal_provider.get_client", return_value=None), \
             patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False):
            provider = LiveSignalProvider(cache_path=tmp.name)
            signals = provider.get_signals("m1", "Will X happen?", 0.5)

    assert "sentiment_score" in signals
    assert signals["information_edge"] == 0.0


def test_live_signal_clamping():
    """Signal values should be clamped to valid ranges."""
    mock_response = {
        "sentiment_score": 5.0,  # should be clamped to 1.0
        "information_edge": -2.0,  # should be clamped to 0.0
        "news_recency": 0.8,
        "confidence": 0.9,
    }

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with patch("polymarket_playground.data.signal_provider.get_client") as mock_client, \
             patch("polymarket_playground.data.signal_provider.call_claude_json", return_value=mock_response), \
             patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False):
            mock_client.return_value = "fake_client"
            provider = LiveSignalProvider(cache_path=tmp.name)
            # Bypass web search
            provider._web_search = lambda q: "test results"
            signals = provider.get_signals("m1", "test?", 0.5)

    assert signals["sentiment_score"] == 1.0
    assert signals["information_edge"] == 0.0
