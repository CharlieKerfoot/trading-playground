"""Trading Playground — batch training entry point."""

from __future__ import annotations

import click
import uvicorn
from rich.console import Console

console = Console()


@click.group()
def cli():
    """Polymarket Trading Playground — train agents on historical markets."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, help="Server port")
def serve(host: str, port: int):
    """Start the web server."""
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

    console.print(f"\nStarting server at http://{host}:{port}")
    uvicorn.run(
        "polymarket_playground.server:app",
        host=host,
        port=port,
        reload=False,
    )


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
def train_rl(timesteps: int, algorithm: str, category: str | None, save_path: str | None):
    """Train an RL agent using Stable-Baselines3."""
    from pathlib import Path
    from polymarket_playground.core.env import TradingEnv
    from polymarket_playground.data.market_cache import MarketCache
    from polymarket_playground.eval.train import RLTrainer

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


if __name__ == "__main__":
    cli()
