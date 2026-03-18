"""Tests for ClaudeAgent response parsing."""

import json
from unittest.mock import MagicMock, patch

from polymarket_playground.agents.claude_agent import ClaudeAgent
from polymarket_playground.core.state import MarketState


def _make_state() -> MarketState:
    return MarketState(
        market_id="test",
        question="Will it rain?",
        yes_price=0.50,
        no_price=0.50,
        yes_bid=0.49,
        yes_ask=0.51,
        no_bid=0.49,
        no_ask=0.51,
        spread=0.02,
        book_depth=1000.0,
        time_elapsed=0.5,
        time_remaining=1800.0,
        signals={
            "best_bid": 0.49,
            "best_ask": 0.51,
            "spread": 0.02,
            "book_depth_bid": 500.0,
            "book_depth_ask": 500.0,
            "last_trade_price": 0.50,
            "volume_24hr": 10000.0,
            "liquidity": 5000.0,
        },
    )


def test_valid_json_response():
    """Valid JSON response is parsed into correct Action."""
    agent = ClaudeAgent(api_key="test-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps({
        "action": "buy_yes",
        "size": 0.5,
        "reasoning": "price is low",
    })

    with patch.object(agent.client.messages, "create", return_value=mock_response):
        action = agent.act(_make_state())

    assert action.direction == 1  # buy_yes
    assert action.size == 0.5
    assert action.reasoning == "price is low"


def test_markdown_code_fences_stripped():
    """Response wrapped in ```json ... ``` is parsed correctly."""
    agent = ClaudeAgent(api_key="test-key")

    raw = '```json\n{"action": "buy_no", "size": 0.3, "reasoning": "overpriced"}\n```'
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = raw

    with patch.object(agent.client.messages, "create", return_value=mock_response):
        action = agent.act(_make_state())

    assert action.direction == 2  # buy_no
    assert action.size == 0.3


def test_invalid_json_falls_back_to_hold():
    """Malformed JSON defaults to hold action."""
    agent = ClaudeAgent(api_key="test-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "I think you should buy YES because..."

    with patch.object(agent.client.messages, "create", return_value=mock_response):
        action = agent.act(_make_state())

    assert action.direction == 0  # hold
    assert action.reasoning == "parse error fallback"


def test_empty_response_falls_back_to_hold():
    """Empty API response defaults to hold."""
    agent = ClaudeAgent(api_key="test-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = ""

    with patch.object(agent.client.messages, "create", return_value=mock_response):
        action = agent.act(_make_state())

    assert action.direction == 0  # hold


def test_no_api_key_returns_hold():
    """Agent with no API key returns hold without calling API."""
    with patch.dict("os.environ", {}, clear=True):
        agent = ClaudeAgent(api_key=None)

    assert agent.client is None

    action = agent.act(_make_state())
    assert action.direction == 0
    assert action.reasoning == "no API key"
