"""Trading agents."""

from __future__ import annotations

from .base_agent import BaseAgent
from .claude_agent import ClaudeAgent
from .rl_agent import RLAgent
from .rule_agent import RuleAgent

__all__ = [
    "BaseAgent",
    "ClaudeAgent",
    "RLAgent",
    "RuleAgent",
    "create_agent",
]


def create_agent(agent_type: str, config: dict | None = None) -> BaseAgent:
    """Create an agent by type string and config.

    Supported types: "rule", "claude", "random", "rl", "rl:<model_path>".
    """
    config = config or {}

    # Handle "rl:/path/to/model" shorthand
    if ":" in agent_type and agent_type.startswith("rl:"):
        model_path = agent_type.split(":", 1)[1]
        algorithm = config.get("algorithm", "PPO")
        return RLAgent.load(model_path, algorithm=algorithm)

    if agent_type == "rule":
        return RuleAgent()
    elif agent_type == "claude":
        claude_kwargs = {
            k: v for k, v in config.items()
            if k in ("model", "api_key", "strategy_memory_path")
        }
        return ClaudeAgent(**claude_kwargs)
    elif agent_type == "random":
        return RLAgent(policy=None)
    elif agent_type == "rl":
        model_path = config.get("model_path", "")
        if model_path:
            algorithm = config.get("algorithm", "PPO")
            return RLAgent.load(model_path, algorithm=algorithm)
        return RLAgent(policy=None)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
