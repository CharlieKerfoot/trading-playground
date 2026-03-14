"""CLI entry point for Trading Playground."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _get_adapter(market: str):
    """Get market adapter by name."""
    from polymarket_playground.markets.btc_price import BTCPriceAdapter
    from polymarket_playground.markets.elections import ElectionAdapter
    from polymarket_playground.markets.sports import SportsAdapter
    from polymarket_playground.markets.macro import MacroAdapter

    adapters = {
        "btc": BTCPriceAdapter,
        "elections": ElectionAdapter,
        "sports": SportsAdapter,
        "macro": MacroAdapter,
    }
    if market not in adapters:
        raise click.BadParameter(f"Unknown market: {market}. Choose from: {list(adapters.keys())}")
    return adapters[market]()


def _get_agent(agent_name: str, adapter=None):
    """Get agent by name."""
    from polymarket_playground.agents.rule_agent import RuleAgent
    from polymarket_playground.agents.claude_agent import ClaudeAgent
    from polymarket_playground.agents.rl_agent import RLAgent

    agents = {
        "rule": lambda: RuleAgent(),
        "claude": lambda: ClaudeAgent(adapter=adapter),
        "rl": lambda: RLAgent(),
        "random": lambda: RLAgent(policy=None),
    }

    # Support loading trained models: "rl:path/to/model"
    if ":" in agent_name and agent_name.startswith("rl:"):
        model_path = agent_name.split(":", 1)[1]
        return RLAgent.load(model_path)

    if agent_name not in agents:
        raise click.BadParameter(f"Unknown agent: {agent_name}. Choose from: {list(agents.keys())}")
    return agents[agent_name]()


def _load_config(market: str) -> dict:
    """Load config for a market type."""
    import yaml
    from pathlib import Path

    config_dir = Path(__file__).parent / "polymarket_playground" / "config"
    base_path = config_dir / "base.yaml"
    market_path = config_dir / f"{market}.yaml"

    config = {}
    if base_path.exists():
        with open(base_path) as f:
            config = yaml.safe_load(f) or {}

    if market_path.exists():
        with open(market_path) as f:
            market_config = yaml.safe_load(f) or {}
            # Merge market config over base
            for key, value in market_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value

    return config


def _find_free_port(start: int = 8000) -> int:
    """Find a free port starting from the given port."""
    import socket
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found")


def _launch_web(port: int = 8000) -> int:
    """Start the FastAPI server and open the browser. Returns the port used."""
    import subprocess
    import sys
    import time
    import webbrowser
    from pathlib import Path

    api_port = _find_free_port(port)

    # Check if the SvelteKit frontend is built
    web_dir = Path(__file__).parent / "web"
    has_frontend = (web_dir / "package.json").exists()

    # Start the FastAPI backend
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "polymarket_playground.server:app",
         "--host", "127.0.0.1", "--port", str(api_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for backend to be ready
    import socket
    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", api_port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)

    if has_frontend:
        # Start SvelteKit dev server, pointing its proxy at our backend
        dev_port = _find_free_port(5173)
        frontend_env = {**__import__("os").environ, "API_PORT": str(api_port)}
        frontend_proc = subprocess.Popen(
            ["npx", "vite", "dev", "--port", str(dev_port)],
            cwd=str(web_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=frontend_env,
        )
        # Wait for frontend
        for _ in range(30):
            try:
                with socket.create_connection(("127.0.0.1", dev_port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.3)

        url = f"http://localhost:{dev_port}"
    else:
        # No frontend build — just open the API docs
        url = f"http://localhost:{api_port}/docs"
        frontend_proc = None

    webbrowser.open(url)
    console.print(f"[bold green]Web interface opened at {url}[/bold green]")
    console.print(f"API server running on port {api_port}")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        server_proc.wait()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        server_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        server_proc.wait(timeout=5)
        if frontend_proc:
            frontend_proc.wait(timeout=5)

    return api_port


@click.group()
def cli():
    """Trading Playground - Agent Training & Backtesting."""
    pass


@cli.command()
@click.option("--agent", default="rule", help="Agent type: rule, claude, rl, random")
@click.option("--market", default="btc", help="Market type: btc, elections, sports, macro")
@click.option("--episodes", default=10, help="Number of episodes to run")
@click.option("--headless", is_flag=True, help="Run without web interface (CLI only)")
def run(agent: str, market: str, episodes: int, headless: bool):
    """Run an agent on a market."""
    if not headless:
        _launch_web()
        return

    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.eval.runner import EpisodeRunner
    from polymarket_playground.eval.metrics import avg_pnl, win_rate, sharpe_ratio

    config = _load_config(market)
    adapter = _get_adapter(market)
    agent_obj = _get_agent(agent, adapter=adapter)
    env = TradingEnv(adapter=adapter, config=config.get("env", {}))

    console.print(f"[bold]Running {agent} agent on {market} market[/bold]")
    console.print(f"Episodes: {episodes}")
    console.print()

    runner = EpisodeRunner(env, agent_obj)
    results = runner.run(n_episodes=episodes)

    # Display results
    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Episodes", str(len(results)))
    table.add_row("Avg P&L", f"{avg_pnl(results):.4f}")
    table.add_row("Win Rate", f"{win_rate(results):.2%}")
    table.add_row("Sharpe Ratio", f"{sharpe_ratio(results):.4f}")
    console.print(table)


@cli.command()
@click.option("--agent", default="rl", help="Agent to train: rl")
@click.option("--market", default="btc", help="Market type")
@click.option("--timesteps", default=100000, help="Training timesteps")
@click.option("--algorithm", default="PPO", help="RL algorithm: PPO, A2C")
@click.option("--save-path", default=None, help="Path to save trained model")
def train(agent: str, market: str, timesteps: int, algorithm: str, save_path: str | None):
    """Train an RL agent."""
    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.eval.train import RLTrainer

    config = _load_config(market)
    adapter = _get_adapter(market)
    env = TradingEnv(adapter=adapter, config=config.get("env", {}))

    console.print(f"[bold]Training {algorithm} agent on {market} market[/bold]")
    console.print(f"Timesteps: {timesteps:,}")
    console.print()

    trainer = RLTrainer(env, algorithm=algorithm)
    trainer.train(timesteps=timesteps)

    if save_path is None:
        save_path = f"models/{algorithm.lower()}_{market}"
    trainer.save(save_path)
    console.print(f"[green]Model saved to {save_path}[/green]")


@cli.command()
@click.option("--agents", multiple=True, required=True, help="Agent types to compare")
@click.option("--market", default="btc", help="Market type")
@click.option("--episodes", default=10, help="Number of episodes")
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("--headless", is_flag=True, help="Run without web interface (CLI only)")
def compare(agents: tuple[str], market: str, episodes: int, seed: int, headless: bool):
    """Compare multiple agents head-to-head."""
    if not headless:
        _launch_web()
        return

    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.eval.runner import MultiAgentComparison
    from polymarket_playground.eval.compare import compare_agents

    config = _load_config(market)
    adapter = _get_adapter(market)
    env = TradingEnv(adapter=adapter, config=config.get("env", {}))

    agent_objs = {}
    for name in agents:
        agent_objs[name] = _get_agent(name, adapter=adapter)

    console.print(f"[bold]Comparing agents: {', '.join(agents)} on {market}[/bold]")
    comparison = MultiAgentComparison(env, agent_objs)
    report = comparison.run(n_episodes=episodes)

    # Display comparison
    stats = compare_agents(report)
    table = Table(title="Agent Comparison")
    table.add_column("Agent", style="cyan")
    table.add_column("Avg P&L", style="green")
    table.add_column("Win Rate", style="green")
    table.add_column("Sharpe", style="green")

    for name, metrics in stats.items():
        table.add_row(
            name,
            f"{metrics.get('avg_pnl', 0):.4f}",
            f"{metrics.get('win_rate', 0):.2%}",
            f"{metrics.get('sharpe_ratio', 0):.4f}",
        )
    console.print(table)


if __name__ == "__main__":
    cli()
