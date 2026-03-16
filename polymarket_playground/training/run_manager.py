"""RunManager — orchestrates batch training runs over cached markets."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from polymarket_playground.core.env import TradingEnv
from polymarket_playground.core.types import EpisodeResult
from polymarket_playground.eval.metrics import (
    avg_pnl,
    max_drawdown,
    sharpe_ratio,
    spread_capture,
    win_rate,
)

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TrainingRun:
    """Represents a batch training run over multiple episodes."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_type: str = "rule"
    agent_config: dict = field(default_factory=dict)
    num_episodes: int = 50
    status: RunStatus = RunStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    results: list[EpisodeResult] = field(default_factory=list)
    current_episode: int = 0
    error: str | None = None

    def metrics(self) -> dict:
        """Compute aggregate metrics from completed episodes."""
        if not self.results:
            return {}
        return {
            "sharpe_ratio": round(sharpe_ratio(self.results), 4),
            "win_rate": round(win_rate(self.results), 4),
            "avg_pnl": round(avg_pnl(self.results), 6),
            "max_drawdown": round(max_drawdown(self.results), 6),
            "spread_capture": round(spread_capture(self.results), 6),
            "total_episodes": len(self.results),
            "total_reward": round(sum(r.total_reward for r in self.results), 6),
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "agent_config": self.agent_config,
            "num_episodes": self.num_episodes,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_episode": self.current_episode,
            "error": self.error,
            "episode_results": [
                {
                    "episode": i + 1,
                    "total_reward": round(r.total_reward, 6),
                    "n_steps": r.n_steps,
                }
                for i, r in enumerate(self.results)
            ],
        }


class RunManager:
    """Manages training runs: create, start, cancel, list."""

    def __init__(self) -> None:
        self._runs: dict[str, TrainingRun] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def create_run(
        self,
        agent_type: str,
        num_episodes: int = 50,
        agent_config: dict | None = None,
    ) -> TrainingRun:
        run = TrainingRun(
            agent_type=agent_type,
            agent_config=agent_config or {},
            num_episodes=num_episodes,
        )
        self._runs[run.id] = run
        return run

    async def start_run(
        self,
        run_id: str,
        env: TradingEnv,
        on_progress: Callable[[TrainingRun], Any] | None = None,
    ) -> None:
        """Start a training run as a background task."""
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        async def _execute():
            run.status = RunStatus.RUNNING
            run.started_at = time.time()

            agent = self._make_agent(run.agent_type, run.agent_config, env)

            try:
                for i in range(run.num_episodes):
                    if run.status == RunStatus.CANCELLED:
                        break

                    env.reset()
                    agent.on_reset(env.state)

                    result = EpisodeResult()
                    done = False
                    while not done:
                        action = agent.act(env.state)
                        _obs, reward, terminated, truncated, info = env.step(action)
                        result.record(action, reward, info)
                        done = terminated or truncated

                    agent.on_episode_end(result)
                    run.results.append(result)
                    run.current_episode = i + 1

                    if on_progress:
                        await on_progress(run)

                    # Yield control to event loop
                    await asyncio.sleep(0)

                if run.status != RunStatus.CANCELLED:
                    run.status = RunStatus.COMPLETED
            except Exception as e:
                run.status = RunStatus.FAILED
                run.error = str(e)
                logger.exception("Run %s failed", run_id)
            finally:
                run.finished_at = time.time()

        task = asyncio.create_task(_execute())
        self._tasks[run_id] = task

    def cancel_run(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if not run:
            return False
        if run.status == RunStatus.RUNNING:
            run.status = RunStatus.CANCELLED
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        return True

    def get_run(self, run_id: str) -> TrainingRun | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[TrainingRun]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def delete_run(self, run_id: str) -> bool:
        self.cancel_run(run_id)
        return self._runs.pop(run_id, None) is not None

    @staticmethod
    def _make_agent(agent_type: str, config: dict, env: TradingEnv):
        if agent_type == "rule":
            from polymarket_playground.agents.rule_agent import RuleAgent
            return RuleAgent()
        elif agent_type == "claude":
            from polymarket_playground.agents.claude_agent import ClaudeAgent
            return ClaudeAgent(**{k: v for k, v in config.items() if k in ("model", "api_key")})
        elif agent_type == "random":
            from polymarket_playground.agents.rl_agent import RLAgent
            return RLAgent(policy=None)
        elif agent_type == "rl":
            from polymarket_playground.agents.rl_agent import RLAgent
            model_path = config.get("model_path", "")
            if model_path:
                algorithm = config.get("algorithm", "PPO")
                return RLAgent.load(model_path, algorithm=algorithm)
            return RLAgent(policy=None)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
