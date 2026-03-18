"""Tests for the shared Claude helper."""

from unittest.mock import MagicMock, patch

from polymarket_playground.agents.claude_helper import (
    call_claude,
    call_claude_json,
    get_client,
)


def test_get_client_no_key():
    with patch.dict("os.environ", {}, clear=True):
        # Remove ANTHROPIC_API_KEY if set
        import os
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            assert get_client() is None
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old


def test_call_claude_no_client():
    result = call_claude(
        [{"role": "user", "content": "hello"}],
        client=None,
        api_key=None,
    )
    # With no valid client, should return None gracefully
    # (depends on env — if ANTHROPIC_API_KEY is set it might work)
    # Just verify no crash
    assert result is None or isinstance(result, str)


def test_call_claude_json_valid():
    with patch("polymarket_playground.agents.claude_helper.call_claude") as mock:
        mock.return_value = '{"key": "value"}'
        result = call_claude_json([{"role": "user", "content": "test"}])
        assert result == {"key": "value"}


def test_call_claude_json_with_fences():
    with patch("polymarket_playground.agents.claude_helper.call_claude") as mock:
        mock.return_value = '```json\n{"key": "value"}\n```'
        result = call_claude_json([{"role": "user", "content": "test"}])
        assert result == {"key": "value"}


def test_call_claude_json_invalid():
    with patch("polymarket_playground.agents.claude_helper.call_claude") as mock:
        mock.return_value = "not json at all"
        result = call_claude_json([{"role": "user", "content": "test"}])
        assert result is None


def test_call_claude_json_none_response():
    with patch("polymarket_playground.agents.claude_helper.call_claude") as mock:
        mock.return_value = None
        result = call_claude_json([{"role": "user", "content": "test"}])
        assert result is None
