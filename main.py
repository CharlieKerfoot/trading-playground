"""Trading Playground — batch training entry point."""

from __future__ import annotations

import logging

import click
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()


def _get_agent(agent_name: str, config: dict | None = None):
    """Get agent by name.

    Passes through config to the agent factory. For Claude agents,
    config can include 'strategy_memory_path' to enable learning loop.
    """
    from polymarket_playground.agents import create_agent

    try:
        return create_agent(agent_name, config)
    except ValueError as e:
        raise click.BadParameter(str(e))


def _get_signal_provider(mode: str = "historical"):
    """Create the appropriate signal provider.

    mode: "historical" for training on resolved markets (Claude-only),
          "live" for live/paper trading (web search + Claude).
    """
    if mode == "live":
        from polymarket_playground.data.signal_provider import LiveSignalProvider
        return LiveSignalProvider()
    else:
        from polymarket_playground.data.signal_provider import HistoricalSignalProvider
        return HistoricalSignalProvider()


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
@click.option("--signals/--no-signals", default=False, help="Enable Claude signal analysis (costs API calls)")
def train(agent: str, episodes: int, category: str | None, signals: bool):
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

    signal_provider = _get_signal_provider("historical") if signals else None
    env = TradingEnv(cache=cache, config={"max_steps": 100}, signal_provider=signal_provider)
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

    from polymarket_playground.agents.claude_helper import cost_tracker
    if cost_tracker.call_count > 0:
        console.print(f"\n[dim]{cost_tracker.summary()}[/dim]")


@cli.command(name="train-rl")
@click.option("--timesteps", default=100_000, help="Training timesteps")
@click.option("--algorithm", default="PPO", help="RL algorithm: PPO, A2C")
@click.option("--category", default=None, help="Filter by category")
@click.option("--save-path", default=None, help="Path to save trained model")
@click.option("--eval-episodes", default=50, help="Episodes to run for evaluation after training")
@click.option("--signals/--no-signals", default=False, help="Enable Claude signal analysis (costs API calls)")
def train_rl(timesteps: int, algorithm: str, category: str | None, save_path: str | None, eval_episodes: int, signals: bool):
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

    signal_provider = _get_signal_provider("historical") if signals else None
    env = TradingEnv(cache=cache, config={"max_steps": 100}, signal_provider=signal_provider)
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

    eval_env = TradingEnv(cache=cache, config={"max_steps": 100}, signal_provider=signal_provider)
    if category:
        eval_env.data.set_market_filter(category)
    eval_wrapped = GymWrapper(eval_env)
    eval_vec = DummyVecEnv([lambda: eval_wrapped])
    # Copy normalization stats from training
    eval_norm = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, training=False)
    if hasattr(trainer.vec_env, "obs_rms") and hasattr(trainer.vec_env, "ret_rms"):
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
@click.option("--signals/--no-signals", default=False, help="Enable Claude signal analysis (costs API calls)")
def compare(agents: tuple[str], episodes: int, category: str | None, signals: bool):
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

    signal_provider = _get_signal_provider("historical") if signals else None
    env = TradingEnv(cache=cache, config={"max_steps": 100}, signal_provider=signal_provider)
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

    # Show regime breakdown if available
    if s.get("regimes"):
        console.print()
        console.print("[bold]Regimes:[/bold]")
        for regime, count in sorted(s["regimes"].items(), key=lambda x: -x[1]):
            console.print(f"  {regime}: {count}")


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
@click.option("--signals/--no-signals", default=False, help="Enable live signal analysis (web search + Claude)")
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
    signals: bool,
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
        import os as _os
        _os.environ["TRADING_MODE"] = "live"
        from polymarket_playground.execution.live_executor import LiveExecutor
        executor = LiveExecutor(max_slippage=0.02)
        console.print("\n[bold red]LIVE MODE — real orders will be placed![/bold red]")
        if not click.confirm("Continue?"):
            return
    elif mode == "dry-run":
        from polymarket_playground.execution.dry_run_executor import DryRunExecutor
        executor = DryRunExecutor()
        console.print("\n[bold yellow]DRY-RUN MODE — orders will be logged, not sent.[/bold yellow]")
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

    signal_provider = _get_signal_provider("live") if signals else None
    runner = LiveRunner(
        agent=agent_obj,
        executor=executor,
        adapter=adapter,
        config=config,
        session_log=SessionLog(),
        signal_provider=signal_provider,
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


@cli.command()
@click.option("--agents", multiple=True, default=("rule", "claude"), help="Agent types to train")
@click.option("--episodes", default=100, help="Episodes per agent")
@click.option("--train-split", default=0.7, help="Fraction of markets for training")
@click.option("--sync-limit", default=200, help="Max markets to sync")
@click.option("--category", default=None, help="Filter by category")
@click.option("--significance", default=0.05, help="p-value threshold for significance")
@click.option("--signals/--no-signals", default=False, help="Enable Claude signal analysis (costs API calls)")
def pipeline(
    agents: tuple[str],
    episodes: int,
    train_split: float,
    sync_limit: int,
    category: str | None,
    significance: float,
    signals: bool,
):
    """End-to-end pipeline: sync → train → validate → rank → recommend."""
    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.data.market_cache import MarketCache
    from polymarket_playground.eval.metrics import (
        avg_pnl,
        sharpe_ratio,
        validate_strategy,
        compare_vs_random,
        win_rate,
    )
    from polymarket_playground.eval.runner import EpisodeRunner

    cache = MarketCache()
    signal_provider = _get_signal_provider("historical") if signals else None

    # --- Stage 1: Sync ---
    console.print("[bold]Stage 1: Sync[/bold]")
    stats = cache.stats()
    if stats["total_markets"] < 20:
        console.print(f"  Syncing up to {sync_limit} markets...")
        result = cache.sync(limit=sync_limit, category=category)
        console.print(f"  Synced: {result}")
    else:
        console.print(f"  Cache ready: {stats['total_markets']} markets")

    # --- Stage 2: Split ---
    console.print("\n[bold]Stage 2: Train/Test Split[/bold]")
    earliest, latest = cache.get_earliest_latest_timestamps()
    if earliest == 0:
        console.print("[red]No price data available.[/red]")
        return

    cutoff = earliest + int((latest - earliest) * train_split)
    train_ids = cache.get_market_ids_by_date_range(before=cutoff)
    test_ids = cache.get_market_ids_by_date_range(after=cutoff)

    # Remove overlap — markets that span the cutoff go to train
    test_ids = [mid for mid in test_ids if mid not in set(train_ids)]

    if len(train_ids) < 5:
        console.print(f"[red]Too few training markets ({len(train_ids)}). Need ≥5.[/red]")
        return
    if len(test_ids) == 0:
        console.print(
            f"[red]No test markets after temporal split.[/red] "
            f"Try syncing more markets or adjusting --train-split."
        )
        return
    if len(test_ids) < 3:
        console.print(f"[yellow]Warning: only {len(test_ids)} test markets.[/yellow]")

    console.print(f"  Train: {len(train_ids)} markets | Test: {len(test_ids)} markets")

    # --- Stage 3: Train ---
    console.print("\n[bold]Stage 3: Train[/bold]")
    agent_results: dict[str, dict] = {}

    for agent_name in agents:
        console.print(f"\n  Training [cyan]{agent_name}[/cyan]...")
        try:
            env = TradingEnv(cache=cache, config={"max_steps": 100}, signal_provider=signal_provider)
            env.data.set_market_ids(train_ids)

            agent_obj = _get_agent(agent_name)
            runner = EpisodeRunner(env, agent_obj)

            # Train
            train_results = runner.run(episodes)
            console.print(
                f"    Train: Sharpe={sharpe_ratio(train_results):.4f} "
                f"WinRate={win_rate(train_results):.1%} "
                f"AvgPnL={avg_pnl(train_results):+.6f}"
            )

            # Test (out-of-sample)
            if test_ids:
                test_env = TradingEnv(cache=cache, config={"max_steps": 100}, signal_provider=signal_provider)
                test_env.data.set_market_ids(test_ids)
                test_runner = EpisodeRunner(test_env, agent_obj)
                test_results = test_runner.run(min(episodes, len(test_ids) * 3))
            else:
                test_results = []

            agent_results[agent_name] = {
                "train": train_results,
                "test": test_results,
            }

            if test_results:
                console.print(
                    f"    Test:  Sharpe={sharpe_ratio(test_results):.4f} "
                    f"WinRate={win_rate(test_results):.1%} "
                    f"AvgPnL={avg_pnl(test_results):+.6f}"
                )

        except Exception as exc:
            console.print(f"    [red]Failed: {exc}[/red]")
            agent_results[agent_name] = {"train": [], "test": [], "error": str(exc)}

    # --- Stage 4: Validate ---
    console.print("\n[bold]Stage 4: Statistical Validation[/bold]")

    # Run random baseline for comparison
    random_env = TradingEnv(cache=cache, config={"max_steps": 100}, signal_provider=signal_provider)
    if test_ids:
        random_env.data.set_market_ids(test_ids)
    random_agent = _get_agent("random")
    random_runner = EpisodeRunner(random_env, random_agent)
    random_results = random_runner.run(episodes)

    validated: list[tuple[str, dict]] = []
    for agent_name, results in agent_results.items():
        if "error" in results:
            continue

        test_results = results.get("test", results["train"])
        if not test_results:
            continue

        validation = validate_strategy(test_results, significance_level=significance)
        beats_random, vs_random_p = compare_vs_random(test_results, random_results, significance)

        console.print(f"\n  [cyan]{agent_name}[/cyan]:")
        console.print(f"    {validation}")
        console.print(f"    Beats random: {'YES' if beats_random else 'NO'} (p={vs_random_p:.4f})")

        validated.append((agent_name, {
            "validation": validation,
            "beats_random": beats_random,
            "vs_random_p": vs_random_p,
            "test_results": test_results,
        }))

    # --- Stage 5: Rank & Recommend ---
    console.print("\n[bold]Stage 5: Recommendation[/bold]")

    # Sort by Sharpe ratio on test set
    ranked = sorted(
        validated,
        key=lambda x: x[1]["validation"].sharpe,
        reverse=True,
    )

    table = Table(title="Strategy Ranking")
    table.add_column("Rank", style="dim")
    table.add_column("Agent", style="cyan")
    table.add_column("Sharpe", style="green")
    table.add_column("Win Rate", style="green")
    table.add_column("Significant?", style="bold")
    table.add_column("Beats Random?", style="bold")
    table.add_column("Recommendation")

    for i, (name, data) in enumerate(ranked):
        v = data["validation"]
        rec = "[green]DEPLOY[/green]" if v.is_significant and data["beats_random"] else "[red]SKIP[/red]"
        table.add_row(
            str(i + 1),
            name,
            f"{v.sharpe:.4f}",
            f"{win_rate(data['test_results']):.1%}",
            "YES" if v.is_significant else "NO",
            "YES" if data["beats_random"] else "NO",
            rec,
        )

    console.print(table)

    deployable = [n for n, d in ranked if d["validation"].is_significant and d["beats_random"]]
    if deployable:
        console.print(f"\n[bold green]Deployable strategies: {', '.join(deployable)}[/bold green]")
        console.print("Run with: python main.py live --agent <name> --markets <ids> --mode paper")
    else:
        console.print("\n[yellow]No strategies passed validation. More training or better signals needed.[/yellow]")

    from polymarket_playground.agents.claude_helper import cost_tracker
    if cost_tracker.call_count > 0:
        console.print(f"\n[dim]{cost_tracker.summary()}[/dim]")


@cli.command()
@click.option("--agent", default="claude", help="Agent type")
@click.option("--episodes", default=50, help="Episodes per batch")
@click.option("--batches", default=5, help="Number of training batches")
@click.option("--memory-path", default="strategy_memory.md", help="Path to strategy memory file")
@click.option("--category", default=None, help="Filter by category")
@click.option("--signals/--no-signals", default=False, help="Enable Claude signal analysis (costs API calls)")
def train_with_memory(
    agent: str,
    episodes: int,
    batches: int,
    memory_path: str,
    category: str | None,
    signals: bool,
):
    """Train Claude agent with learning loop (strategy memory across batches)."""
    from polymarket_playground.agents.strategy_memory import StrategyMemory
    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.data.market_cache import MarketCache
    from polymarket_playground.eval.metrics import avg_pnl, sharpe_ratio, win_rate
    from polymarket_playground.eval.runner import EpisodeRunner

    cache = MarketCache()
    stats = cache.stats()
    if stats["total_markets"] == 0:
        console.print("[red]No markets in cache.[/red] Run sync first.")
        return

    signal_provider = _get_signal_provider("historical") if signals else None
    memory = StrategyMemory(memory_path)

    console.print(f"[bold]Training {agent} with learning loop[/bold]")
    console.print(f"Batches: {batches} × {episodes} episodes = {batches * episodes} total")
    console.print(f"Strategy memory: {memory_path}")
    if signals:
        console.print("[dim]  (signal analysis enabled)[/dim]")
    if memory.load():
        console.print("[dim]  (existing memory found, will build on it)[/dim]")
    else:
        console.print("[dim]  (starting fresh)[/dim]")

    all_results = []
    for batch in range(batches):
        console.print(f"\n[bold]Batch {batch + 1}/{batches}[/bold]")

        env = TradingEnv(cache=cache, config={"max_steps": 100}, signal_provider=signal_provider)
        if category:
            env.data.set_market_filter(category)

        agent_obj = _get_agent(agent, {"strategy_memory_path": memory_path})
        runner = EpisodeRunner(env, agent_obj)
        results = runner.run(episodes)
        all_results.extend(results)

        console.print(
            f"  Sharpe={sharpe_ratio(results):.4f} "
            f"WinRate={win_rate(results):.1%} "
            f"AvgPnL={avg_pnl(results):+.6f}"
        )

        # Update strategy memory after batch
        console.print("  Updating strategy memory...")
        memory.update(results)

    console.print(f"\n[bold green]Overall ({len(all_results)} episodes):[/bold green]")
    console.print(f"  Sharpe: {sharpe_ratio(all_results):.4f}")
    console.print(f"  Win Rate: {win_rate(all_results):.1%}")
    console.print(f"  Avg P&L: {avg_pnl(all_results):+.6f}")


@cli.command()
@click.option("--agent", default="rule", help="Agent type: rule, claude, random")
@click.option("--market-id", default=None, help="Specific market ID (random if omitted)")
@click.option("--max-steps", default=50, help="Max steps per episode")
@click.option("--category", default=None, help="Filter by category")
@click.option("--signals/--no-signals", default=False, help="Enable Claude signal analysis (costs API calls)")
def replay(agent: str, market_id: str | None, max_steps: int, category: str | None, signals: bool):
    """Replay an episode with step-by-step price, trades, P&L, and reasoning."""
    from polymarket_playground.core.action import DIRECTION_NAMES
    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.core.types import EpisodeResult
    from polymarket_playground.data.market_cache import MarketCache

    cache = MarketCache()
    stats = cache.stats()
    if stats["total_markets"] == 0:
        console.print("[red]No markets in cache.[/red] Run sync first.")
        return

    signal_provider = _get_signal_provider("historical") if signals else None
    env = TradingEnv(cache=cache, config={"max_steps": max_steps}, signal_provider=signal_provider)
    if category:
        env.data.set_market_filter(category)
    if market_id:
        env.data.set_market_ids([market_id])

    agent_obj = _get_agent(agent)

    # Run one episode, collecting detailed data
    env.reset()
    agent_obj.on_reset(env.state)

    episode_market_id = env.state.market_id
    question = env.state.question
    result = EpisodeResult()
    prices: list[float] = [env.state.yes_price]
    pnl_curve: list[float] = [0.0]

    done = False
    while not done:
        action = agent_obj.act(env.state)
        _obs, reward, terminated, truncated, info = env.step(action)
        result.record(action, reward, info)
        prices.append(info["state"]["yes_price"])
        pnl_curve.append(pnl_curve[-1] + reward)
        done = terminated or truncated

    agent_obj.on_episode_end(result)

    # --- Display ---
    console.print(f"\n[bold]Episode Replay[/bold]")
    console.print(f"Market: [cyan]{episode_market_id}[/cyan]")
    console.print(f"Question: {question}")
    console.print(f"Agent: {agent} | Steps: {result.n_steps} | Total P&L: {result.total_reward:+.4f}")

    # ASCII price chart
    console.print(f"\n[bold]Price Chart (YES)[/bold]")
    _print_ascii_chart(prices, label="Price", color="blue")

    # P&L curve
    console.print(f"\n[bold]Cumulative P&L[/bold]")
    _print_ascii_chart(pnl_curve, label="P&L", color="green")

    # Trade log
    console.print(f"\n[bold]Trade Log[/bold]")
    trade_table = Table()
    trade_table.add_column("Step", style="dim")
    trade_table.add_column("Action", style="cyan")
    trade_table.add_column("Size")
    trade_table.add_column("Price")
    trade_table.add_column("Reward", style="green")
    trade_table.add_column("Cum P&L")
    trade_table.add_column("Position")
    trade_table.add_column("Reasoning")

    for i, (action, reward, info) in enumerate(
        zip(result.actions, result.rewards, result.infos)
    ):
        action_name = DIRECTION_NAMES.get(action.direction, "hold")
        fill = info.get("fill", {})
        pos = info.get("position", {})
        reasoning = action.reasoning[:60] if action.reasoning else ""

        # Only show non-hold actions in detail, but always show trades
        if action.direction != 0 or abs(reward) > 0.001:
            pos_str = ""
            if pos.get("yes_shares", 0) > 0:
                pos_str += f"Y:{pos['yes_shares']:.1f}"
            if pos.get("no_shares", 0) > 0:
                pos_str += f" N:{pos['no_shares']:.1f}"
            if not pos_str:
                pos_str = "flat"

            trade_table.add_row(
                str(i + 1),
                action_name,
                f"{fill.get('size', 0):.2f}",
                f"{fill.get('price', 0):.4f}",
                f"{reward:+.4f}",
                f"{pnl_curve[i + 1]:+.4f}",
                pos_str,
                reasoning,
            )

    console.print(trade_table)

    # Final summary
    final_info = result.infos[-1] if result.infos else {}
    final_state = final_info.get("state", {})
    final_pos = final_info.get("position", {})
    console.print(f"\n[bold]Final State[/bold]")
    console.print(f"  Resolved: {final_state.get('is_resolved', False)}")
    console.print(f"  Final price: {final_state.get('yes_price', 0):.4f}")
    console.print(f"  Realized P&L: {final_pos.get('realized_pnl', 0):+.4f}")
    console.print(f"  Total reward: {result.total_reward:+.4f}")


@cli.command(name="fidelity-report")
def fidelity_report():
    """Show sim-to-real fidelity report from recorded paper fills."""
    from polymarket_playground.eval.fidelity import FidelityValidator

    validator = FidelityValidator()
    report = validator.report()

    if report.n_records == 0:
        console.print("[dim]No paper fills recorded yet.[/dim]")
        console.print("Run a paper trading session with --mode paper to collect data.")
        return

    console.print(f"[bold]Sim-to-Real Fidelity Report[/bold]")
    console.print(f"  Fills compared: {report.n_records}")
    console.print(f"  Fidelity Score: [bold]{report.fidelity_score:.4f}[/bold]")
    console.print(f"  Mean slippage error: {report.mean_slippage_error:+.6f}")
    console.print(f"  Mean |error|:        {report.mean_abs_error:.6f}")
    console.print(f"  Max  |error|:        {report.max_slippage_error:.6f}")

    if report.fidelity_score >= 0.9:
        console.print("\n[green]High fidelity — paper fills closely match real book.[/green]")
    elif report.fidelity_score >= 0.7:
        console.print("\n[yellow]Moderate fidelity — paper fills have some divergence from real book.[/yellow]")
    else:
        console.print("\n[red]Low fidelity — paper fills diverge significantly from real book.[/red]")


@cli.command()
@click.option("--agents", multiple=True, required=True, help="Agent types to compete")
@click.option("--episodes", default=20, help="Episodes per match")
@click.option("--category", default=None, help="Filter by category")
@click.option("--signals/--no-signals", default=False, help="Enable Claude signal analysis")
def tournament(agents: tuple[str], episodes: int, category: str | None, signals: bool):
    """Run a tournament between agents and update ELO rankings."""
    from itertools import combinations

    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.data.market_cache import MarketCache
    from polymarket_playground.eval.tournament import Tournament

    cache = MarketCache()
    stats = cache.stats()
    if stats["total_markets"] == 0:
        console.print("[red]No markets in cache.[/red] Run sync first.")
        return

    signal_provider = _get_signal_provider("historical") if signals else None
    env = TradingEnv(cache=cache, config={"max_steps": 100}, signal_provider=signal_provider)
    if category:
        env.data.set_market_filter(category)

    t = Tournament()
    agent_objs = {}
    for name in agents:
        agent_objs[name] = _get_agent(name)

    pairs = list(combinations(agents, 2))
    console.print(f"[bold]Tournament: {len(pairs)} matches ({episodes} episodes each)[/bold]")

    for a_name, b_name in pairs:
        console.print(f"\n  [cyan]{a_name}[/cyan] vs [cyan]{b_name}[/cyan]...")
        results = t.run_match(
            env, a_name, agent_objs[a_name], b_name, agent_objs[b_name],
            n_episodes=episodes,
        )
        wins_a = sum(1 for r in results if r.winner == a_name)
        wins_b = sum(1 for r in results if r.winner == b_name)
        draws = sum(1 for r in results if r.winner == "draw")
        console.print(f"    {a_name}: {wins_a}W | {b_name}: {wins_b}W | Draws: {draws}")

    # Show updated rankings
    _show_rankings(t)


@cli.command()
def rankings():
    """Show current ELO rankings from tournament history."""
    from polymarket_playground.eval.tournament import Tournament

    t = Tournament()
    _show_rankings(t)


def _show_rankings(t):
    """Display ELO rankings table."""
    rankings = t.get_rankings()
    if not rankings:
        console.print("[dim]No rankings yet. Run a tournament first.[/dim]")
        return

    table = Table(title="Agent ELO Rankings")
    table.add_column("Rank", style="dim")
    table.add_column("Agent", style="cyan")
    table.add_column("ELO", style="bold")
    table.add_column("Matches")
    table.add_column("W/L/D")
    table.add_column("Win Rate", style="green")

    for i, r in enumerate(rankings):
        table.add_row(
            str(i + 1),
            r.agent_name,
            f"{r.elo:.0f}",
            str(r.matches),
            f"{r.wins}/{r.losses}/{r.draws}",
            f"{r.win_rate:.1%}",
        )
    console.print(table)


@cli.command(name="canary-start")
@click.option("--agent", required=True, help="Agent type to paper trade")
@click.option("--markets", required=True, help="Comma-separated market IDs")
@click.option("--hours", default=24.0, help="Duration in hours")
@click.option("--interval", default=60, help="Seconds between ticks")
@click.option("--capital", default=100.0, help="Starting capital (USD)")
def canary_start(agent: str, markets: str, hours: float, interval: int, capital: float):
    """Start a canary paper trading run for deployment validation."""
    from polymarket_playground.training.canary import CanaryConfig, CanaryManager

    market_ids = [m.strip() for m in markets.split(",") if m.strip()]
    if not market_ids:
        console.print("[red]No market IDs provided.[/red]")
        return

    config = CanaryConfig(
        agent_name=agent,
        market_ids=market_ids,
        duration_hours=hours,
        interval_seconds=interval,
        starting_capital=capital,
    )

    manager = CanaryManager()
    canary_id = manager.start_canary(config)

    console.print(f"[bold green]Canary #{canary_id} started[/bold green]")
    console.print(f"  Agent: {agent}")
    console.print(f"  Markets: {len(market_ids)}")
    console.print(f"  Duration: {hours}h")
    console.print(f"  Interval: {interval}s")
    console.print(f"  Capital: ${capital:.2f}")
    console.print()

    # Run the canary inline (blocking)
    import signal as sig

    from polymarket_playground.execution.paper_executor import PaperExecutor
    from polymarket_playground.core.portfolio import Portfolio
    from polymarket_playground.markets.polymarket import PolymarketAdapter

    agent_obj = _get_agent(agent)
    executor = PaperExecutor()
    adapter = PolymarketAdapter()
    portfolio = Portfolio(starting_capital=capital)
    running = True
    import time
    deadline = time.time() + hours * 3600

    def handle_signal(signum, frame):
        nonlocal running
        running = False

    old_handler = sig.signal(sig.SIGINT, handle_signal)

    console.print("[bold]Running canary...[/bold] (Ctrl+C to stop)\n")
    tick_count = 0
    latest_prices: dict[str, float] = {}
    try:
        while running and time.time() < deadline:
            for mid in market_ids:
                try:
                    state = adapter.build_state(mid)
                    position = portfolio.get_position(mid)
                    state.agent_yes_shares = position.yes_shares
                    state.agent_no_shares = position.no_shares
                    state.agent_cost_basis = position.cost_basis
                    state.agent_realized_pnl = position.realized_pnl

                    action = agent_obj.act(state)
                    fill = None
                    if not action.is_hold:
                        fill = executor.execute(action, state, position)
                        if fill:
                            portfolio.update(mid, fill)

                    latest_prices[mid] = state.yes_price
                    pnl = portfolio.total_pnl(latest_prices)
                    manager.record_tick(
                        canary_id, mid, state.yes_price,
                        action.direction_name, 0.0, pnl,
                    )
                except Exception as exc:
                    logger.warning("Canary tick error on %s: %s", mid, exc)

            tick_count += 1
            elapsed_h = (time.time() - (deadline - hours * 3600)) / 3600
            console.print(
                f"  Tick {tick_count} | "
                f"Elapsed: {elapsed_h:.1f}h / {hours}h | "
                f"P&L: {portfolio.total_pnl(latest_prices):.4f}",
                end="\r",
            )
            if running:
                time.sleep(interval)
    finally:
        sig.signal(sig.SIGINT, old_handler)
        manager.stop_canary(canary_id)

    console.print()
    confidence = manager.compute_confidence(canary_id)
    console.print(f"\n[bold]Canary #{canary_id} complete[/bold]")
    console.print(f"  Ticks: {tick_count}")
    console.print(f"  Confidence Score: [bold]{confidence:.4f}[/bold]")

    if confidence >= 0.7:
        console.print("[green]READY for deployment[/green]")
    elif confidence >= 0.4:
        console.print("[yellow]CAUTIOUS — consider extending canary duration[/yellow]")
    else:
        console.print("[red]NOT READY — canary results are poor[/red]")


@cli.command(name="canary-status")
@click.option("--id", "canary_id", default=None, type=int, help="Specific canary ID")
def canary_status(canary_id: int | None):
    """Check status of canary paper trading runs."""
    from polymarket_playground.training.canary import CanaryManager
    from datetime import datetime

    manager = CanaryManager()

    if canary_id:
        status = manager.get_status(canary_id)
        if not status:
            console.print(f"[red]Canary #{canary_id} not found.[/red]")
            return
        statuses = [status]
    else:
        statuses = manager.get_all_canaries()

    if not statuses:
        console.print("[dim]No canary runs found.[/dim]")
        return

    table = Table(title="Canary Runs")
    table.add_column("ID", style="cyan")
    table.add_column("Agent")
    table.add_column("Started")
    table.add_column("Elapsed")
    table.add_column("Ticks")
    table.add_column("P&L", style="green")
    table.add_column("Confidence", style="bold")
    table.add_column("Status")

    for s in statuses:
        started = datetime.fromtimestamp(s.started_at).strftime("%Y-%m-%d %H:%M")
        status_str = "[green]RUNNING[/green]" if s.is_running else "[dim]STOPPED[/dim]"
        table.add_row(
            str(s.canary_id),
            s.agent_name,
            started,
            f"{s.elapsed_hours:.1f}h",
            str(s.ticks),
            f"{s.total_pnl:+.4f}",
            f"{s.confidence_score:.4f}",
            status_str,
        )

    console.print(table)


@cli.command(name="api-costs")
def api_costs():
    """Show API cost summary from token usage tracking."""
    from polymarket_playground.agents.claude_helper import cost_tracker

    import sqlite3
    try:
        conn = sqlite3.connect("market_data.db")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT model, SUM(input_tokens) as total_in, "
            "SUM(output_tokens) as total_out, SUM(cost_usd) as total_cost, "
            "COUNT(*) as calls FROM api_usage GROUP BY model"
        ).fetchall()
        conn.close()
    except Exception:
        console.print("[dim]No API usage data found.[/dim]")
        return

    if not rows:
        console.print("[dim]No API usage data found.[/dim]")
        return

    table = Table(title="API Cost Breakdown")
    table.add_column("Model", style="cyan")
    table.add_column("Calls")
    table.add_column("Input Tokens")
    table.add_column("Output Tokens")
    table.add_column("Cost (USD)", style="green")

    total_cost = 0.0
    total_calls = 0
    for row in rows:
        table.add_row(
            row["model"],
            f"{row['calls']:,}",
            f"{row['total_in']:,}",
            f"{row['total_out']:,}",
            f"${row['total_cost']:.2f}",
        )
        total_cost += row["total_cost"]
        total_calls += row["calls"]

    console.print(table)
    console.print(f"\n[bold]Total: ${total_cost:.2f} across {total_calls:,} calls[/bold]")

    # Session summary
    console.print(f"\n[dim]Current session: {cost_tracker.summary()}[/dim]")


def _print_ascii_chart(
    values: list[float], label: str = "", color: str = "white", width: int = 60, height: int = 12,
) -> None:
    """Print a simple ASCII chart using Rich."""
    if not values:
        return

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val or 0.001

    # Resample to fit width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    # Build rows
    rows: list[str] = []
    for row in range(height):
        threshold = max_val - (row / (height - 1)) * val_range
        line = ""
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        # Y-axis label on first, middle, last rows
        if row == 0:
            rows.append(f"  {max_val:>8.4f} │{line}│")
        elif row == height - 1:
            rows.append(f"  {min_val:>8.4f} │{line}│")
        elif row == height // 2:
            mid = (max_val + min_val) / 2
            rows.append(f"  {mid:>8.4f} │{line}│")
        else:
            rows.append(f"           │{line}│")

    for r in rows:
        console.print(f"[{color}]{r}[/{color}]")


if __name__ == "__main__":
    cli()
