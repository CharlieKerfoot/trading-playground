"""Head-to-head agent comparison utilities."""

from __future__ import annotations

from polymarket_playground.core.types import ComparisonReport
from polymarket_playground.eval.metrics import (
    avg_pnl,
    max_drawdown,
    sharpe_ratio,
    spread_capture,
    win_rate,
)


def compare_agents(report: ComparisonReport) -> dict[str, dict[str, float]]:
    """Return per-agent metrics from a comparison report.

    Returns a dict mapping each agent name to a metrics dict with keys:
    ``sharpe_ratio``, ``win_rate``, ``avg_pnl``, ``max_drawdown``,
    ``spread_capture``.
    """
    comparison: dict[str, dict[str, float]] = {}
    for name, results in report.results.items():
        comparison[name] = {
            "sharpe_ratio": sharpe_ratio(results),
            "win_rate": win_rate(results),
            "avg_pnl": avg_pnl(results),
            "max_drawdown": max_drawdown(results),
            "spread_capture": spread_capture(results),
        }
    return comparison
