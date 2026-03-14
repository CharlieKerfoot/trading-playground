# Trading Playground

Agent-agnostic, market-agnostic training environment for prediction market agents. Uses **real Polymarket market data** — historical prices, orderbooks, and signals from the Gamma and CLOB APIs. Plug in Claude, RL policies, or rule-based bots — all against the same Gymnasium-compatible environment.

## Quick Start

```bash
# Install dependencies
uv sync
cd web && npm install && cd ..

# Launch the web interface (opens browser automatically)
uv run python cli.py run

# Run headless (CLI only, no web interface)
uv run python cli.py run --headless --agent rule --market btc --episodes 10
```

By default, all commands launch the web dashboard. Add `--headless` to run in CLI-only mode.

## Web Interface

The web dashboard starts automatically when you run any command. It launches both the FastAPI backend and the SvelteKit frontend, then opens your browser.

### Features

- **Real Polymarket data** — Historical prices, orderbooks, and signals from the Polymarket API
- **Manual trading** — Buy YES/NO, Hold, or Close positions with adjustable size
- **Agent visualization** — Watch Rule, Claude, or Random agents trade in real-time
- **Live dashboard** — Price display, position tracking, P&L chart, signals panel, trade history
- **WebSocket streaming** — Agent actions broadcast in real-time via WebSocket
- **Synthetic fallback** — BTC, Elections, Sports, and Macro synthetic markets for offline testing

### Manual startup

If you prefer to start the servers separately:

```bash
# Terminal 1 — backend
uv run uvicorn polymarket_playground.server:app --port 8000

# Terminal 2 — frontend
cd web && npm run dev
```

## CLI Commands

### `run` — Run an agent

```bash
# Opens web interface (default)
uv run python cli.py run

# Headless mode
uv run python cli.py run --headless \
  --agent rule \        # rule | claude | rl | random
  --market btc \        # btc | elections | sports | macro
  --episodes 10
```

### `compare` — Head-to-head agent comparison

```bash
uv run python cli.py compare --headless \
  --agents rule --agents random \
  --market btc \
  --episodes 50 \
  --seed 42
```

### `train` — Train an RL agent

```bash
uv run python cli.py train \
  --market btc \          # btc | elections | sports | macro
  --timesteps 100000 \    # training steps
  --algorithm PPO \       # PPO | A2C
  --save-path models/my_model
```

## Architecture

```
polymarket_playground/
├── core/           # TradingEnv, MarketState, Action, Position
├── markets/        # Pluggable market adapters (BTC, elections, sports, macro)
├── data/           # Polymarket client, historical replay
├── execution/      # Paper executor with slippage model
├── agents/         # Claude, RL, rule-based agents
├── eval/           # Episode runner, metrics, comparison, visualization, RL training
├── config/         # YAML configs per market type
└── server.py       # FastAPI + WebSocket backend for the web interface

web/                # SvelteKit frontend
├── src/
│   ├── lib/
│   │   ├── api.ts              # TypeScript API client
│   │   └── components/         # Svelte 5 UI components
│   │       ├── PriceDisplay    # YES/NO prices with bid/ask spreads
│   │       ├── PositionPanel   # Position sizes and P&L
│   │       ├── TradePanel      # Manual trading controls
│   │       ├── SignalsPanel    # Market signals and time progress
│   │       ├── PnlChart        # SVG cumulative P&L + price chart
│   │       └── HistoryTable    # Scrollable trade history
│   └── routes/
│       └── +page.svelte        # Main dashboard
└── vite.config.ts              # Vite proxy to FastAPI backend
```

### Design Principles

1. **Agent-agnostic** — Claude, RL policies, and rule-based bots all implement the same `BaseAgent.act(state) -> Action` interface.

2. **Market-agnostic** — Market-specific logic (signals, context formatting, episode selection) lives in swappable `MarketAdapter` plugins. Adding a new market type = adding one file in `markets/`.

3. **Training-focused** — Historical replay with paper execution. All agents train and evaluate against resolved market data.

### Key Abstractions

- **`MarketState`** — Universal market snapshot (prices, spreads, book depth, time, external signals). Every agent sees the same struct.
- **`MarketAdapter`** — Pluggable market logic. Fetches market-specific signals and builds agent context.
- **`TradingEnv`** — Gymnasium-compatible environment. Market-agnostic core loop.
- **`BaseAgent`** — Agent interface. `act(state) -> Action`. Agents extract what they need from `MarketState` (language for Claude, numpy for RL, raw fields for rules).
- **`PaperExecutor`** — Simulated fills with slippage model.

## Agents

| Agent | Description |
|-------|-------------|
| `rule` | Deterministic baseline — buys YES below 0.3, NO above 0.7 |
| `claude` | Claude API agent with multi-turn conversation history per episode |
| `rl` | SB3-compatible RL policy wrapper (load with `rl:path/to/model`) |
| `random` | Random actions (RL agent with no policy) |

## Markets

### Polymarket (Historical)

The app loads resolved markets from Polymarket for historical replay. Each market provides signals:

| Signal | Description |
|--------|-------------|
| `best_bid` / `best_ask` | Top-of-book prices from CLOB orderbook |
| `spread` | Bid-ask spread |
| `book_depth_bid` / `book_depth_ask` | Total size at top 5 price levels |
| `last_trade_price` | Most recent trade |
| `volume_24hr` | 24-hour trading volume (USDC) |
| `liquidity` | Available liquidity |

No API key required — Polymarket's market data endpoints are public.

### Synthetic Markets (Offline Fallback)

| Market | Adapter | Signals |
|--------|---------|---------|
| `btc` | `BTCPriceAdapter` | btc_spot, btc_delta_1m/5m, funding_rate, book_imbalance, realized_vol |
| `elections` | `ElectionAdapter` | poll_average, news_sentiment, prediction_market_consensus |
| `sports` | `SportsAdapter` | live_score, time_remaining, odds_from_bookmakers |
| `macro` | `MacroAdapter` | prev_reading, analyst_consensus, market_implied_prob |

## Configuration

YAML configs in `polymarket_playground/config/` control environment parameters:

```yaml
# base.yaml — defaults
env:
  max_steps: 50
  position_limit: 10.0
  fee_rate: 0.02
```

Market-specific configs (e.g. `btc.yaml`) override base settings.

## Claude Agent Setup

The Claude agent requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python cli.py run --headless --agent claude --market btc --episodes 5
```

Without the key, the agent degrades gracefully (holds every step). The agent:
- Receives natural-language market context from the adapter
- Maintains multi-turn conversation history within each episode
- Tracks its own position and P&L in the prompt
- Responds with structured JSON actions
