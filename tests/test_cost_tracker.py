"""Tests for the API Cost Tracker."""

from polymarket_playground.agents.claude_helper import CostTracker


def test_log_usage(tmp_path):
    tracker = CostTracker(db_path=str(tmp_path / "test.db"))
    tracker.log_usage("claude-sonnet-4-6", 1000, 500)

    assert tracker.call_count == 1
    assert tracker.total_input_tokens == 1000
    assert tracker.total_output_tokens == 500
    assert tracker.total_cost_usd > 0


def test_summary_no_calls(tmp_path):
    tracker = CostTracker(db_path=str(tmp_path / "test.db"))
    assert "0 calls" in tracker.summary()


def test_summary_with_calls(tmp_path):
    tracker = CostTracker(db_path=str(tmp_path / "test.db"))
    tracker.log_usage("claude-haiku-4-5", 5000, 1000)

    summary = tracker.summary()
    assert "1 calls" in summary
    assert "$" in summary


def test_reset(tmp_path):
    tracker = CostTracker(db_path=str(tmp_path / "test.db"))
    tracker.log_usage("claude-sonnet-4-6", 1000, 500)
    assert tracker.call_count == 1

    tracker.reset()
    assert tracker.call_count == 0
    assert tracker.total_cost_usd == 0.0


def test_pricing_calculation(tmp_path):
    tracker = CostTracker(db_path=str(tmp_path / "test.db"))
    # 1M input tokens of Haiku at $0.80/M = $0.80
    tracker.log_usage("claude-haiku-4-5", 1_000_000, 0)
    assert abs(tracker.total_cost_usd - 0.80) < 0.001
