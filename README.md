# Trading Playground

Agent-agnostic, market-agnostic training environment for prediction market agents. Uses **real Polymarket data** — historical prices, orderbooks, and signals from the Gamma and CLOB APIs. Plug in Claude, RL policies, or rule-based bots — all against the same Gymnasium-compatible environment.

## Quick Start

```bash
# Install dependencies
uv sync
cd web && npm install && cd ..

# Sync resolved markets from Polymarket
uv run python cli.py sync --limit 100

# Launch the web interface (opens browser automatically)
uv run python cli.py serve

# Or run an agent from the CLI
uv run python cli.py run --agent rule --episodes 50
```

## Web Interface

The web dashboard starts with `cli.py serve`. It launches both the FastAPI backend and the SvelteKit frontend, then opens your browser.

**Pages:**

- **Dashboard** (`/`) — Overview stats (total markets, price points, categories, runs) and recent training runs
- **Data** (`/data`) — Sync resolved Polymarket markets into the local SQLite cache, view category breakdowns
- **Train** (`/train`) — Configure and launch training runs: pick an agent, set parameters, choose episode count and category filter
- **Runs** (`/runs`) — Browse all training runs, click into any run for live WebSocket progress, episode-by-episode rewards, and final metrics (Sharpe, win rate, P&L, max drawdown)

### Manual startup

If you prefer to start the servers separately:

```bash
# Terminal 1 — backend
uv run uvicorn polymarket_playground.server:app --port 8000

# Terminal 2 — frontend
cd web && npm run dev
```

## CLI Commands

### `serve` — Launch the web UI

```bash
uv run python cli.py serve [--host 0.0.0.0] [--port 8000]
```

### `sync` — Download market data

```bash
uv run python cli.py sync --limit 100 [--category btc]
```

### `run` — Run an agent

```bash
uv run python cli.py run \
  --agent rule \        # rule | claude | rl | random | rl:path/to/model
  --episodes 50 \
  --category btc        # optional: filter markets by category
```

### `train` — Train an RL agent

```bash
uv run python cli.py train \
  --timesteps 100000 \
  --algorithm PPO \       # PPO | A2C
  --category btc \
  --save-path models/my_model
```

### `compare` — Head-to-head agent comparison

```bash
uv run python cli.py compare \
  --agents rule --agents random \
  --episodes 50 \
  --category btc
```

### `stats` — Show cached data statistics

```bash
uv run python cli.py stats
```

## Architecture

```
polymarket_playground/
├── core/           # TradingEnv, MarketState, Action, Position
├── agents/         # Claude, RL, rule-based agents
├── markets/        # Pluggable market adapters (Polymarket)
├── data/           # Polymarket client, SQLite cache, historical replay
├── execution/      # Paper executor with slippage model
├── eval/           # Episode runner, metrics, comparison, RL training
├── training/       # Run manager for batch training
├── config/         # YAML configs for environment parameters
└── server.py       # FastAPI + WebSocket backend

web/                # SvelteKit frontend (Svelte 5 + TypeScript)
├── src/
│   ├── lib/
│   │   ├── api.ts          # TypeScript API client
│   │   └── components/     # MetricsCard, RunsTable, AgentConfigurator, etc.
│   └── routes/
│       ├── +page.svelte    # Dashboard
│       ├── data/           # Data sync & management
│       ├── train/          # Training configuration
│       └── runs/           # Run listing & detail views
└── vite.config.ts          # Vite proxy to FastAPI backend
```

### Design Principles

1. **Agent-agnostic** — All agents implement `BaseAgent.act(state) -> Action`. Claude, RL policies, and rule-based bots share the same interface.
2. **Market-agnostic** — Market-specific logic lives in swappable `MarketAdapter` plugins. Adding a new market = one file in `markets/`.
3. **Training-focused** — Historical replay with paper execution. No live trading — all agents train and evaluate against resolved market data.

### Key Abstractions

| Abstraction | Role |
|-------------|------|
| `MarketState` | Universal market snapshot: prices, spreads, book depth, time, external signals |
| `MarketAdapter` | Pluggable market logic — fetches signals and builds agent context |
| `TradingEnv` | Gymnasium-compatible environment with configurable fees, position limits, and step count |
| `BaseAgent` | Agent interface: `act(state) -> Action` |
| `PaperExecutor` | Simulated fills with configurable slippage model |

## Agents

| Agent | Description |
|-------|-------------|
| `rule` | Deterministic baseline — buys YES below 0.3, NO above 0.7 |
| `claude` | Claude API agent with multi-turn conversation history per episode |
| `rl` | SB3-compatible RL policy wrapper (load trained model with `rl:path/to/model`) |
| `random` | Random actions (RL agent with no policy) |

## Markets

Resolved markets synced from Polymarket's public API into a local SQLite cache. No API key required.

| Signal | Description |
|--------|-------------|
| `best_bid` / `best_ask` | Top-of-book prices from CLOB orderbook |
| `spread` | Bid-ask spread |
| `book_depth_bid` / `book_depth_ask` | Total size at top 5 price levels |
| `last_trade_price` | Most recent trade |
| `volume_24hr` | 24-hour trading volume (USDC) |
| `liquidity` | Available liquidity |

## Configuration

YAML configs in `polymarket_playground/config/` control environment parameters:

```yaml
# base.yaml — defaults
env:
  max_steps: 50
  position_limit: 10.0
  fee_rate: 0.02
```

## Claude Agent Setup

The Claude agent requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python cli.py run --agent claude --episodes 5
```

Without the key, the agent degrades gracefully (holds every step).
