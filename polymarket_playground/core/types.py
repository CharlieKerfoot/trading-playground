"""Shared types: Fill, Episode, EpisodeResult, Resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Fill:
    """Represents a single executed trade."""
    direction: str  # "yes" | "no"
    size: float
    price: float
    fees: float = 0.0
    spread_capture: float = 0.0
    timestamp: datetime | None = None


@dataclass
class Resolution:
    """Market resolution outcome."""
    market_id: str
    outcome: float  # 1.0 = YES, 0.0 = NO
    timestamp: datetime | None = None


@dataclass
class Episode:
    """Metadata for a single trading episode."""
    market_id: str
    market_type: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    resolution: Resolution | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Accumulated results from running one episode."""
    actions: list = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    n_steps: int = 0

    def record(self, action, reward: float, info: dict):
        self.actions.append(action)
        self.rewards.append(reward)
        self.infos.append(info)
        self.total_reward += reward
        self.n_steps += 1


@dataclass
class ComparisonReport:
    """Multi-agent comparison results."""
    agent_names: list[str]
    results: dict[str, list[EpisodeResult]] = field(default_factory=dict)

    def add(self, agent_name: str, results: list[EpisodeResult]):
        self.results[agent_name] = results

    def summary(self) -> dict[str, dict]:
        summary = {}
        for name, results in self.results.items():
            rewards = [r.total_reward for r in results]
            summary[name] = {
                "episodes": len(results),
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "total_reward": sum(rewards),
                "win_rate": sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0,
            }
        return summary
