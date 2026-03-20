"""Tests for the Agent Tournament / ELO system."""

from polymarket_playground.eval.tournament import (
    DEFAULT_ELO,
    Tournament,
    TournamentResult,
)


def test_rankings_empty(tmp_path):
    t = Tournament(db_path=tmp_path / "test.db")
    assert t.get_rankings() == []
    t.close()


def test_elo_update(tmp_path):
    t = Tournament(db_path=tmp_path / "test.db")

    # Agent A wins completely
    t._update_elo("agent_a", "agent_b", score_a=1.0, score_b=0.0)

    elo_a = t._get_elo("agent_a")
    elo_b = t._get_elo("agent_b")

    # Winner gains, loser loses
    assert elo_a > DEFAULT_ELO
    assert elo_b < DEFAULT_ELO
    # ELO is zero-sum
    assert abs((elo_a - DEFAULT_ELO) + (elo_b - DEFAULT_ELO)) < 0.01
    t.close()


def test_elo_draw(tmp_path):
    t = Tournament(db_path=tmp_path / "test.db")

    t._update_elo("agent_a", "agent_b", score_a=0.5, score_b=0.5)

    elo_a = t._get_elo("agent_a")
    elo_b = t._get_elo("agent_b")

    # Both should stay near default (equal expected score)
    assert abs(elo_a - DEFAULT_ELO) < 1.0
    assert abs(elo_b - DEFAULT_ELO) < 1.0
    t.close()


def test_record_and_retrieve_rankings(tmp_path):
    t = Tournament(db_path=tmp_path / "test.db")

    results = [
        TournamentResult(
            agent_a="alice", agent_b="bob",
            pnl_a=0.5, pnl_b=-0.3, winner="alice",
        ),
        TournamentResult(
            agent_a="alice", agent_b="bob",
            pnl_a=0.1, pnl_b=0.1, winner="draw",
        ),
    ]

    t._update_elo("alice", "bob", score_a=0.75, score_b=0.25)
    t._record_results("alice", "bob", results, wins_a=1, wins_b=0, draws=1)

    rankings = t.get_rankings()
    assert len(rankings) == 2

    # Alice should be ranked first (higher ELO)
    assert rankings[0].agent_name == "alice"
    assert rankings[0].elo > rankings[1].elo
    assert rankings[0].wins == 1
    assert rankings[0].draws == 1
    t.close()


def test_reset(tmp_path):
    t = Tournament(db_path=tmp_path / "test.db")
    t._update_elo("alice", "bob", score_a=1.0, score_b=0.0)
    assert len(t.get_rankings()) == 2

    t.reset()
    assert len(t.get_rankings()) == 0
    t.close()
