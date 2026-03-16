"""Episode runner and multi-agent comparison harness."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from polymarket_playground.core.types import ComparisonReport, EpisodeResult

if TYPE_CHECKING:
    from polymarket_playground.agents.base_agent import BaseAgent
    from polymarket_playground.core.env import TradingEnv


class EpisodeRunner:
    """Runs episodes of a single agent in an environment."""

    def __init__(self, env: TradingEnv, agent: BaseAgent) -> None:
        self.env = env
        self.agent = agent

    def run(
        self,
        n_episodes: int,
        on_episode_done: callable | None = None,
    ) -> list[EpisodeResult]:
        """Run *n_episodes* and collect results.

        Parameters
        ----------
        on_episode_done:
            Optional callback invoked after each episode with (episode_index, result).
        """
        results: list[EpisodeResult] = []
        for i in range(n_episodes):
            result = self._run_one()
            results.append(result)
            if on_episode_done:
                on_episode_done(i, result)
        return results

    def run_seeded(self, seeds: list[int]) -> list[EpisodeResult]:
        """Run one episode per seed for reproducibility."""
        results: list[EpisodeResult] = []
        for seed in seeds:
            random.seed(seed)
            self.env.reset(seed=seed)
            result = self._run_episode()
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_one(self) -> EpisodeResult:
        self.env.reset()
        return self._run_episode()

    def _run_episode(self) -> EpisodeResult:
        result = EpisodeResult()
        self.agent.on_reset(self.env.state)

        done = False
        while not done:
            action = self.agent.act(self.env.state)
            _obs, reward, terminated, truncated, info = self.env.step(action)
            result.record(action, reward, info)
            done = terminated or truncated

        self.agent.on_episode_end(result)
        return result


class MultiAgentComparison:
    """Run several agents on the same seeded episodes and compare."""

    def __init__(self, env: TradingEnv, agents: dict[str, BaseAgent]) -> None:
        self.env = env
        self.agents = agents

    def run(self, n_episodes: int) -> ComparisonReport:
        """Run all agents on identical seeded episodes."""
        seeds = [random.randint(0, 2**31 - 1) for _ in range(n_episodes)]
        report = ComparisonReport(agent_names=list(self.agents.keys()))

        for name, agent in self.agents.items():
            runner = EpisodeRunner(self.env, agent)
            results = runner.run_seeded(seeds)
            report.add(name, results)

        return report
