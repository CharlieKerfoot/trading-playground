"""Tests for the Sim-to-Real Fidelity Validator."""

from datetime import datetime, timezone

from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import Fill
from polymarket_playground.eval.fidelity import FidelityValidator


def _make_state(**overrides) -> MarketState:
    defaults = dict(
        market_id="test-market",
        yes_price=0.50,
        no_price=0.50,
        yes_bid=0.49,
        yes_ask=0.51,
        no_bid=0.49,
        no_ask=0.51,
        spread=0.02,
        book_depth=1000.0,
        timestamp=datetime.now(tz=timezone.utc),
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def test_empty_report(tmp_path):
    validator = FidelityValidator(db_path=tmp_path / "test.db")
    report = validator.report()
    assert report.n_records == 0
    assert report.fidelity_score == 0.0
    validator.close()


def test_record_and_report(tmp_path):
    validator = FidelityValidator(db_path=tmp_path / "test.db")
    state = _make_state()

    # Record a buy fill slightly above ask
    fill = Fill(direction="yes", size=5.0, price=0.52)
    record = validator.record_fill(fill, state)
    assert record.slippage_error == fill.price - state.yes_ask  # 0.52 - 0.51 = 0.01

    report = validator.report()
    assert report.n_records == 1
    assert report.mean_slippage_error > 0
    assert 0 < report.fidelity_score <= 1.0
    validator.close()


def test_perfect_fidelity(tmp_path):
    validator = FidelityValidator(db_path=tmp_path / "test.db")
    state = _make_state()

    # Fill exactly at the ask — zero slippage error
    fill = Fill(direction="yes", size=5.0, price=0.51)
    validator.record_fill(fill, state)

    report = validator.report()
    assert report.mean_abs_error < 1e-9
    assert report.fidelity_score > 0.99
    validator.close()


def test_close_fill_uses_bid(tmp_path):
    validator = FidelityValidator(db_path=tmp_path / "test.db")
    state = _make_state()

    fill = Fill(direction="close", size=5.0, price=0.49)
    record = validator.record_fill(fill, state)
    assert record.slippage_error == 0.49 - 0.49  # exactly at bid
    validator.close()


def test_snapshot_recording(tmp_path):
    validator = FidelityValidator(db_path=tmp_path / "test.db")
    state = _make_state()

    validator.record_snapshot(state)
    validator.record_snapshot(state)

    # Snapshots don't affect fidelity report (only fills do)
    report = validator.report()
    assert report.n_records == 0
    validator.close()


def test_clear(tmp_path):
    validator = FidelityValidator(db_path=tmp_path / "test.db")
    state = _make_state()

    fill = Fill(direction="yes", size=5.0, price=0.52)
    validator.record_fill(fill, state)
    assert validator.report().n_records == 1

    validator.clear()
    assert validator.report().n_records == 0
    validator.close()
