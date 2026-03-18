"""Tests for strategy memory persistence."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from polymarket_playground.agents.strategy_memory import StrategyMemory
from polymarket_playground.core.types import EpisodeResult


def _make_results(n: int = 10) -> list[EpisodeResult]:
    """Create dummy episode results for testing."""
    from polymarket_playground.core.action import Action

    results = []
    for i in range(n):
        er = EpisodeResult()
        reward = 0.1 if i % 2 == 0 else -0.05
        action = Action(direction=1 if i % 3 == 0 else 0, size=0.5)
        er.record(action, reward, {})
        er.total_reward = reward
        results.append(er)
    return results


def test_load_no_file():
    """Loading from non-existent file returns empty string."""
    memory = StrategyMemory("/tmp/nonexistent_strategy_test_12345.md")
    assert memory.load() == ""


def test_load_existing_file():
    """Loading from existing file returns contents."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("Buy low, sell high.")
        path = f.name

    memory = StrategyMemory(path)
    assert memory.load() == "Buy low, sell high."
    Path(path).unlink()


def test_clear():
    """Clearing removes the file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("test")
        path = f.name

    memory = StrategyMemory(path)
    assert Path(path).exists()
    memory.clear()
    assert not Path(path).exists()


def test_clear_no_file():
    """Clearing when no file exists doesn't crash."""
    memory = StrategyMemory("/tmp/nonexistent_strategy_test_12345.md")
    memory.clear()  # should not raise


def test_summarize_batch():
    """Batch summary includes key statistics."""
    results = _make_results(10)
    summary = StrategyMemory._summarize_batch(results)

    assert "10 episodes" in summary
    assert "Win/Loss" in summary
    assert "Mean reward" in summary


def test_update_no_api_key():
    """Update without API key logs warning and doesn't crash."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        path = f.name

    with patch("polymarket_playground.agents.strategy_memory.get_client", return_value=None):
        memory = StrategyMemory(path)
        results = _make_results(5)
        memory.update(results)  # should not raise

    Path(path).unlink(missing_ok=True)


def test_update_with_claude():
    """Update with mocked Claude should write to file."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        path = f.name

    with patch("polymarket_playground.agents.strategy_memory.get_client") as mock_gc, \
         patch("polymarket_playground.agents.strategy_memory.call_claude") as mock_call:
        mock_gc.return_value = "fake_client"
        mock_call.return_value = "Updated strategy: buy when price < 0.3"

        memory = StrategyMemory(path)
        results = _make_results(10)
        memory.update(results)

    content = Path(path).read_text()
    assert "buy when price < 0.3" in content
    Path(path).unlink()


def test_update_empty_results():
    """Update with empty results should be a no-op."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        path = f.name

    memory = StrategyMemory(path)
    memory.update([])  # should not raise or write anything
    Path(path).unlink(missing_ok=True)
