"""Trading Playground — batch training entry point."""

from __future__ import annotations

import click
import uvicorn
from rich.console import Console
from rich.table import Table

console = Console()


def _get_agent(agent_name: str, config: dict | None = None):
    """Get agent by name."""
    from polymarket_playground.agents.rule_agent import RuleAgent
    from polymarket_playground.agents.claude_agent import ClaudeAgent
    from polymarket_playground.agents.rl_agent import RLAgent

    config = config or {}

    agents = {
        "rule": lambda: RuleAgent(),
        "claude": lambda: ClaudeAgent(**{k: v for k, v in config.items() if k in ("model", "api_key")}),
        "rl": lambda: RLAgent(policy=None),
        "random": lambda: RLAgent(policy=None),
    }

    if ":" in agent_name and agent_name.startswith("rl:"):
        model_path = agent_name.split(":", 1)[1]
        return RLAgent.load(model_path)

    if agent_name not in agents:
        raise click.BadParameter(f"Unknown agent: {agent_name}. Choose from: {list(agents.keys())}")
    return agents[agent_name]()


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


@click.group()
def cli():
    """Polymarket Trading Playground — train agents on historical markets."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, help="Server port")
def serve(host: str, port: int):
    """Start the API server and SvelteKit frontend."""
    import subprocess
    import sys
    import time
    import webbrowser
    import socket
    from pathlib import Path

    from polymarket_playground.data.market_cache import MarketCache

    cache = MarketCache()
    stats = cache.stats()

    if stats["total_markets"] == 0:
        console.print(
            "[yellow]Market cache is empty.[/yellow] "
            "Run [bold]python main.py sync[/bold] first, "
            "or use the web UI at /data to sync."
        )
    else:
        console.print(
            f"[green]Cache ready:[/green] {stats['total_markets']} markets, "
            f"{stats['total_price_points']:,} price points"
        )

    api_port = _find_free_port(port)

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
    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", api_port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)

    if has_frontend:
        dev_port = _find_free_port(5173)
        frontend_env = {**__import__("os").environ, "API_PORT": str(api_port)}
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(dev_port)],
            cwd=str(web_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=frontend_env,
        )
        for _ in range(30):
            try:
                with socket.create_connection(("127.0.0.1", dev_port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.3)

        url = f"http://localhost:{dev_port}"
    else:
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


@cli.command()
@click.option("--limit", default=100, help="Max markets to fetch")
@click.option("--category", default=None, help="Filter by category/tag")
def sync(limit: int, category: str | None):
    """Bulk download resolved markets from Polymarket."""
    from polymarket_playground.data.market_cache import MarketCache

    cache = MarketCache()
    console.print(f"Syncing up to {limit} resolved markets...")

    def on_progress(fetched, skipped, errors):
        console.print(
            f"  [blue]fetched={fetched}[/blue] skipped={skipped} errors={errors}",
            end="\r",
        )

    result = cache.sync(category=category, limit=limit, on_progress=on_progress)
    console.print()
    console.print(f"[green]Done:[/green] {result}")

    stats = cache.stats()
    console.print(
        f"Cache total: {stats['total_markets']} markets, "
        f"{stats['total_price_points']:,} price points"
    )


@cli.command()
@click.option("--agent", default="rule", help="Agent type: rule, claude, random, rl")
@click.option("--episodes", default=50, help="Number of episodes")
@click.option("--category", default=None, help="Filter by category")
def train(agent: str, episodes: int, category: str | None):
    """Run episode-based training from the CLI."""
    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.data.market_cache import MarketCache
    from polymarket_playground.eval.metrics import (
        avg_pnl,
        max_drawdown,
        sharpe_ratio,
        win_rate,
    )
    from polymarket_playground.training.run_manager import RunManager

    cache = MarketCache()
    stats = cache.stats()
    if stats["total_markets"] == 0:
        console.print("[red]No markets in cache.[/red] Run sync first.")
        return

    env = TradingEnv(cache=cache, config={"max_steps": 100})
    if category:
        env.data.set_market_filter(category)

    agent_obj = RunManager._make_agent(agent, {}, env)

    console.print(
        f"Training [bold]{agent}[/bold] agent for {episodes} episodes "
        f"({env.data.available_markets} markets available)"
    )

    from polymarket_playground.core.types import EpisodeResult

    results: list[EpisodeResult] = []
    for i in range(episodes):
        env.reset()
        agent_obj.on_reset(env.state)

        result = EpisodeResult()
        done = False
        while not done:
            action = agent_obj.act(env.state)
            _obs, reward, terminated, truncated, info = env.step(action)
            result.record(action, reward, info)
            done = terminated or truncated

        agent_obj.on_episode_end(result)
        results.append(result)

        console.print(
            f"  Episode {i + 1}/{episodes}: reward={result.total_reward:+.4f} "
            f"steps={result.n_steps}",
            end="\r",
        )

    console.print()
    console.print("[green]Results:[/green]")
    console.print(f"  Sharpe Ratio:  {sharpe_ratio(results):.4f}")
    console.print(f"  Win Rate:      {win_rate(results):.1%}")
    console.print(f"  Avg P&L:       {avg_pnl(results):+.6f}")
    console.print(f"  Max Drawdown:  {max_drawdown(results):+.6f}")


@cli.command(name="train-rl")
@click.option("--timesteps", default=100_000, help="Training timesteps")
@click.option("--algorithm", default="PPO", help="RL algorithm: PPO, A2C")
@click.option("--category", default=None, help="Filter by category")
@click.option("--save-path", default=None, help="Path to save trained model")
@click.option("--eval-episodes", default=50, help="Episodes to run for evaluation after training")
def train_rl(timesteps: int, algorithm: str, category: str | None, save_path: str | None, eval_episodes: int):
    """Train an RL agent using Stable-Baselines3, then evaluate."""
    from pathlib import Path
    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.data.market_cache import MarketCache
    from polymarket_playground.eval.train import RLTrainer
    from polymarket_playground.eval.metrics import (
        avg_pnl,
        max_drawdown,
        sharpe_ratio,
        win_rate,
    )
    from polymarket_playground.eval.runner import EpisodeRunner
    from polymarket_playground.agents.rl_agent import RLAgent

    cache = MarketCache()
    stats = cache.stats()
    if stats["total_markets"] == 0:
        console.print("[red]No markets in cache.[/red] Run sync first.")
        return

    env = TradingEnv(cache=cache, config={"max_steps": 100})
    if category:
        env.data.set_market_filter(category)

    console.print(f"[bold]Training {algorithm} agent[/bold]")
    console.print(f"Timesteps: {timesteps:,}")
    console.print(f"Markets available: {env.data.available_markets}")

    trainer = RLTrainer(env, algorithm=algorithm)
    trainer.train(timesteps=timesteps)

    if save_path is None:
        save_path = f"models/{algorithm.lower()}_agent"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(save_path)
    console.print(f"[green]Model saved to {save_path}[/green]")

    # Evaluate the trained model using the same normalization stats
    console.print(f"\n[bold]Evaluating trained model ({eval_episodes} episodes)...[/bold]")
    from stable_baselines3.common.evaluation import evaluate_policy
    from polymarket_playground.eval.train import GymWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    eval_env = TradingEnv(cache=cache, config={"max_steps": 100})
    if category:
        eval_env.data.set_market_filter(category)
    eval_wrapped = GymWrapper(eval_env)
    eval_vec = DummyVecEnv([lambda: eval_wrapped])
    # Copy normalization stats from training
    eval_norm = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, training=False)
    eval_norm.obs_rms = trainer.vec_env.obs_rms
    eval_norm.ret_rms = trainer.vec_env.ret_rms

    mean_reward, std_reward = evaluate_policy(
        trainer.model, eval_norm, n_eval_episodes=eval_episodes, deterministic=True
    )

    console.print("[green]Evaluation Results:[/green]")
    console.print(f"  Mean Reward:   {mean_reward:+.4f} (+/- {std_reward:.4f})")

    # Also run through EpisodeRunner for detailed metrics
    agent = RLAgent(policy=trainer.model, position_limit=env.position_limit)
    runner = EpisodeRunner(eval_env, agent)
    results = runner.run(eval_episodes)
    console.print(f"  Sharpe Ratio:  {sharpe_ratio(results):.4f}")
    console.print(f"  Win Rate:      {win_rate(results):.1%}")
    console.print(f"  Avg P&L:       {avg_pnl(results):+.6f}")
    console.print(f"  Max Drawdown:  {max_drawdown(results):+.6f}")


@cli.command()
@click.option("--agents", multiple=True, required=True, help="Agent types to compare")
@click.option("--episodes", default=50, help="Number of episodes")
@click.option("--category", default=None, help="Filter by category")
def compare(agents: tuple[str], episodes: int, category: str | None):
    """Compare multiple agents head-to-head on identical episodes."""
    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.data.market_cache import MarketCache
    from polymarket_playground.eval.runner import MultiAgentComparison
    from polymarket_playground.eval.compare import compare_agents

    cache = MarketCache()
    stats = cache.stats()
    if stats["total_markets"] == 0:
        console.print("[red]No markets in cache.[/red] Run sync first.")
        return

    env = TradingEnv(cache=cache, config={"max_steps": 100})
    if category:
        env.data.set_market_filter(category)

    agent_objs = {}
    for name in agents:
        agent_objs[name] = _get_agent(name)

    console.print(f"[bold]Comparing agents: {', '.join(agents)}[/bold]")
    console.print(f"Episodes: {episodes}, Markets: {env.data.available_markets}")

    comparison = MultiAgentComparison(env, agent_objs)
    report = comparison.run(n_episodes=episodes)

    results = compare_agents(report)
    table = Table(title="Agent Comparison")
    table.add_column("Agent", style="cyan")
    table.add_column("Avg P&L", style="green")
    table.add_column("Win Rate", style="green")
    table.add_column("Sharpe", style="green")
    table.add_column("Max DD", style="red")

    for name, metrics in results.items():
        table.add_row(
            name,
            f"{metrics['avg_pnl']:+.4f}",
            f"{metrics['win_rate']:.1%}",
            f"{metrics['sharpe_ratio']:.4f}",
            f"{metrics['max_drawdown']:+.4f}",
        )
    console.print(table)


@cli.command()
def stats():
    """Show cached data statistics."""
    from polymarket_playground.data.market_cache import MarketCache

    cache = MarketCache()
    s = cache.stats()

    console.print(f"[bold]Markets:[/bold] {s['total_markets']}")
    console.print(f"[bold]Price Points:[/bold] {s['total_price_points']:,}")
    console.print()
    if s["categories"]:
        console.print("[bold]Categories:[/bold]")
        for cat, count in sorted(s["categories"].items(), key=lambda x: -x[1]):
            console.print(f"  {cat or 'uncategorized'}: {count}")
    else:
        console.print("[dim]No data cached yet.[/dim]")


@cli.command()
@click.option("--agent", default="rule", help="Agent type: rule, claude, random, rl:<path>")
@click.option("--markets", required=True, help="Comma-separated market IDs")
@click.option("--mode", default="paper", type=click.Choice(["paper", "live", "dry-run"]))
@click.option("--capital", default=100.0, help="Starting capital (USD)")
@click.option("--interval", default=60, help="Seconds between ticks")
@click.option("--max-ticks", default=0, help="Max ticks (0 = unlimited)")
@click.option("--max-position", default=50.0, help="Max position size per market (USD)")
@click.option("--max-exposure", default=200.0, help="Max total exposure (USD)")
@click.option("--max-loss", default=100.0, help="Session kill switch loss (USD)")
def live(
    agent: str,
    markets: str,
    mode: str,
    capital: float,
    interval: int,
    max_ticks: int,
    max_position: float,
    max_exposure: float,
    max_loss: float,
):
    """Run an agent against live Polymarket data."""
    import logging
    from polymarket_playground.execution.risk_gate import RiskConfig
    from polymarket_playground.markets.polymarket import PolymarketAdapter
    from polymarket_playground.training.live_runner import LiveConfig, LiveRunner
    from polymarket_playground.data.session_log import SessionLog

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    market_ids = [m.strip() for m in markets.split(",") if m.strip()]
    if not market_ids:
        console.print("[red]No market IDs provided.[/red]")
        return

    agent_obj = _get_agent(agent)
    adapter = PolymarketAdapter()

    # Validate markets exist
    console.print(f"[bold]Mode: {mode}[/bold]")
    console.print(f"Agent: {type(agent_obj).__name__}")
    console.print(f"Markets: {len(market_ids)}")
    console.print(f"Capital: ${capital:.2f}")
    console.print(f"Interval: {interval}s")
    console.print()

    for mid in market_ids:
        try:
            state = adapter.build_state(mid)
            console.print(f"  [green]{mid}[/green]: {state.question[:60]}... YES={state.yes_price:.3f}")
        except Exception as e:
            console.print(f"  [red]{mid}[/red]: {e}")
            return

    # Build executor based on mode
    if mode == "live":
        from polymarket_playground.execution.live_executor import LiveExecutor
        executor = LiveExecutor(max_slippage=0.02)
        console.print("\n[bold red]LIVE MODE — real orders will be placed![/bold red]")
        if not click.confirm("Continue?"):
            return
    else:
        from polymarket_playground.execution.paper_executor import PaperExecutor
        executor = PaperExecutor()

    risk_config = RiskConfig(
        max_position_usd=max_position,
        max_total_exposure_usd=max_exposure,
        max_loss_per_session_usd=max_loss,
    )

    config = LiveConfig(
        market_ids=market_ids,
        interval_seconds=interval,
        starting_capital=capital,
        risk=risk_config,
        mode=mode,
        max_ticks=max_ticks,
    )

    runner = LiveRunner(
        agent=agent_obj,
        executor=executor,
        adapter=adapter,
        config=config,
        session_log=SessionLog(),
    )

    console.print("\n[bold green]Starting...[/bold green] (Ctrl+C to stop)\n")
    summary = runner.run()

    console.print("\n[bold]Session Summary:[/bold]")
    console.print(f"  Ticks:              {summary['ticks']}")
    console.print(f"  Total P&L:          {summary['total_pnl']:+.4f}")
    console.print(f"  Final Capital:      ${summary['final_capital']:.2f}")
    console.print(f"  Active Positions:   {summary['active_positions']}")
    console.print(f"  Risk Interventions: {summary['risk_interventions']}")
    if summary['kill_switch']:
        console.print("  [red]Kill switch was triggered![/red]")


@cli.command()
@click.option("--limit", "n", default=20, help="Number of sessions to show")
def sessions(n: int):
    """List past live/paper trading sessions."""
    from polymarket_playground.data.session_log import SessionLog
    from datetime import datetime

    log = SessionLog()
    records = log.list_sessions(limit=n)

    if not records:
        console.print("[dim]No sessions recorded yet.[/dim]")
        return

    table = Table(title="Trading Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Agent")
    table.add_column("Mode")
    table.add_column("Started")
    table.add_column("P&L", style="green")
    table.add_column("Ticks")
    table.add_column("Risk Blocks")

    for r in records:
        started = datetime.fromtimestamp(r.started_at).strftime("%Y-%m-%d %H:%M")
        pnl = f"{r.total_pnl:+.4f}" if r.total_pnl is not None else "—"
        table.add_row(
            str(r.session_id),
            r.agent_type,
            r.mode,
            started,
            pnl,
            str(r.total_ticks),
            str(r.risk_interventions),
        )

    console.print(table)


@cli.command(name="session-report")
@click.argument("session_id", type=int)
def session_report(session_id: int):
    """Show detailed report for a trading session."""
    from polymarket_playground.data.session_log import SessionLog
    from datetime import datetime

    log = SessionLog()
    ticks = log.get_session_ticks(session_id)

    if not ticks:
        console.print(f"[red]No ticks found for session {session_id}.[/red]")
        return

    console.print(f"[bold]Session {session_id} — {len(ticks)} ticks[/bold]\n")

    # Group by market
    markets: dict[str, list] = {}
    for t in ticks:
        mid = t["market_id"]
        if mid not in markets:
            markets[mid] = []
        markets[mid].append(t)

    for mid, market_ticks in markets.items():
        console.print(f"[cyan]Market: {mid}[/cyan]")
        actions_taken = [t for t in market_ticks if t["action"] and t["action"] != "hold"]
        risk_blocked = [t for t in market_ticks if t["risk_blocked"]]
        console.print(f"  Ticks: {len(market_ticks)}, Actions: {len(actions_taken)}, Blocked: {len(risk_blocked)}")

        for t in actions_taken[:10]:  # show first 10 actions
            ts = datetime.fromtimestamp(t["timestamp"]).strftime("%H:%M:%S")
            fill_info = ""
            if t["fill_direction"]:
                fill_info = f" -> fill {t['fill_direction']} {t['fill_size']:.2f}@{t['fill_price']:.4f}"
            console.print(f"    [{ts}] {t['action']} size={t['action_size']:.2f}{fill_info}")

        if len(actions_taken) > 10:
            console.print(f"    ... and {len(actions_taken) - 10} more actions")
        console.print()


if __name__ == "__main__":
    cli()
