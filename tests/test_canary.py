"""Tests for the Canary Paper Trading manager."""

import time

from polymarket_playground.training.canary import CanaryConfig, CanaryManager


def test_start_and_get_status(tmp_path):
    manager = CanaryManager(db_path=tmp_path / "test.db")
    config = CanaryConfig(
        agent_name="rule",
        market_ids=["market-1", "market-2"],
        duration_hours=24.0,
    )

    canary_id = manager.start_canary(config)
    assert canary_id is not None

    status = manager.get_status(canary_id)
    assert status.agent_name == "rule"
    assert status.is_running
    assert len(status.market_ids) == 2
    manager.close()


def test_record_ticks(tmp_path):
    manager = CanaryManager(db_path=tmp_path / "test.db")
    config = CanaryConfig(agent_name="rule", market_ids=["m1"])
    canary_id = manager.start_canary(config)

    manager.record_tick(canary_id, "m1", 0.55, "buy_yes", 0.01, 0.01)
    manager.record_tick(canary_id, "m1", 0.56, "hold", 0.005, 0.015)

    status = manager.get_status(canary_id)
    assert status.ticks == 2
    assert status.total_pnl == 0.015
    manager.close()


def test_stop_canary(tmp_path):
    manager = CanaryManager(db_path=tmp_path / "test.db")
    config = CanaryConfig(agent_name="rule", market_ids=["m1"])
    canary_id = manager.start_canary(config)

    manager.stop_canary(canary_id)

    status = manager.get_status(canary_id)
    assert not status.is_running
    manager.close()


def test_compute_confidence_no_ticks(tmp_path):
    manager = CanaryManager(db_path=tmp_path / "test.db")
    config = CanaryConfig(agent_name="rule", market_ids=["m1"])
    canary_id = manager.start_canary(config)

    confidence = manager.compute_confidence(canary_id)
    assert confidence == 0.0
    manager.close()


def test_compute_confidence_with_positive_ticks(tmp_path):
    manager = CanaryManager(db_path=tmp_path / "test.db")
    config = CanaryConfig(agent_name="rule", market_ids=["m1"], duration_hours=1.0)
    canary_id = manager.start_canary(config)

    # Record positive ticks
    for i in range(20):
        pnl = 0.01 * (i + 1)
        manager.record_tick(canary_id, "m1", 0.55, "hold", 0.01, pnl)

    confidence = manager.compute_confidence(canary_id)
    assert confidence > 0.0
    assert confidence <= 1.0
    manager.close()


def test_get_all_canaries(tmp_path):
    manager = CanaryManager(db_path=tmp_path / "test.db")
    config1 = CanaryConfig(agent_name="rule", market_ids=["m1"])
    config2 = CanaryConfig(agent_name="claude", market_ids=["m2"])

    manager.start_canary(config1)
    manager.start_canary(config2)

    all_canaries = manager.get_all_canaries()
    assert len(all_canaries) == 2
    manager.close()


def test_nonexistent_canary(tmp_path):
    manager = CanaryManager(db_path=tmp_path / "test.db")
    status = manager.get_status(999)
    assert status is None
    manager.close()
