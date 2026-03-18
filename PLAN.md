# Trading Playground — Architecture

## Design Philosophy

Three hard requirements drive every decision:

1. **Agent-agnostic** — Claude agents, RL policies, rule-based bots all plug in identically via `BaseAgent`. No agent knows or cares how the env works underneath.

2. **Market-agnostic** — All markets are represented as binary outcome contracts with YES/NO prices. Market-specific details live in cached metadata, not the core env.

3. **Training-focused** — The environment uses historical replay from cached Polymarket data and paper execution only. All agents train and evaluate against resolved market data.

---

## Directory Structure

```
polymarket_playground/
│
├── core/
│   ├── env.py                  # TradingEnv (Gymnasium-compatible)
│   ├── state.py                # MarketState — universal market snapshot
│   ├── position.py             # Position tracking, P&L, settlement
│   ├── action.py               # Action schema (direction, size)
│   ├── types.py                # Shared types: Fill, Episode, EpisodeResult
│   └── portfolio.py            # Multi-market portfolio (for live trading)
│
├── data/
│   ├── market_cache.py         # SQLite-backed cache for resolved markets
│   ├── historical_loader.py    # Episode replay from cached price history
│   ├── polymarket_client.py    # Polymarket CLOB REST API client
│   ├── episode_store.py        # Persistent run/episode storage (SQLite)
│   ├── signal_provider.py      # Signal providers (Historical + Live)
│   └── session_log.py          # Live trading session log
│
├── execution/
│   ├── base_executor.py        # Abstract executor interface
│   ├── paper_executor.py       # Simulated fills with slippage model
│   ├── slippage_model.py       # Quadratic market impact model
│   ├── live_executor.py        # Real order placement (py-clob-client)
│   └── risk_gate.py            # Risk controls for live trading
│
├── agents/
│   ├── __init__.py             # create_agent() factory + re-exports
│   ├── base_agent.py           # Abstract agent interface
│   ├── claude_agent.py         # Claude API agent (multi-turn, JSON output)
│   ├── claude_helper.py        # Shared Claude API wrapper (retry, rate limits)
│   ├── strategy_memory.py      # Persistent strategy memory for learning loop
│   ├── rl_agent.py             # RL policy wrapper (SB3-compatible)
│   └── rule_agent.py           # Deterministic threshold-based baseline
│
├── eval/
│   ├── runner.py               # Episode runner + multi-agent comparison
│   ├── train.py                # SB3 RL training (PPO/A2C) + GymWrapper
│   ├── metrics.py              # Sharpe, win rate, avg P&L, validation, bootstrap CI
│   └── compare.py              # Head-to-head comparison analysis
│
├── training/
│   ├── run_manager.py          # Async batch run orchestration (FastAPI)
│   └── live_runner.py          # Real-time trading loop (paper/live/dry-run)
│
├── markets/
│   └── polymarket.py           # PolymarketAdapter (live state builder)
│
└── server.py                   # FastAPI backend + WebSocket streaming

main.py                         # CLI entry point (click)
```

---

## Core Abstractions

### `MarketState` — the universal snapshot

Every timestep produces this struct. Agents see only this.

```python
@dataclass
class MarketState:
    market_id: str
    market_type: str
    question: str

    yes_price: float          # [0, 1]
    no_price: float           # [0, 1]
    yes_bid, yes_ask: float
    no_bid, no_ask: float
    spread: float
    book_depth: float

    time_elapsed: float       # fraction of episode elapsed [0, 1]
    time_remaining: float     # seconds
    timestamp: datetime

    signals: dict[str, float] # best_bid, best_ask, volume_24hr, etc.

    is_resolved: bool
    resolution: float | None  # 1.0 = YES, 0.0 = NO

    # Agent position (injected by env after each step)
    agent_yes_shares: float
    agent_no_shares: float
    agent_cost_basis: float
    agent_realized_pnl: float
```

### Agent Interface

Every agent implements `BaseAgent`:

```python
class BaseAgent(ABC):
    @abstractmethod
    def act(self, state: MarketState) -> Action: ...
    def on_reset(self, state: MarketState): ...
    def on_episode_end(self, result: EpisodeResult): ...
```

Agents are created via `create_agent(agent_type, config)` from `polymarket_playground.agents`.

---

## Data Flow

```
                    ┌─────────────────────────────────┐
                    │          MarketCache             │
                    │   (SQLite: markets + prices)     │
                    └────────────┬────────────────────┘
                                 │ get_price_history()
                    ┌────────────▼────────────────────┐
                    │      HistoricalLoader            │
                    │  (random window, resample,       │
                    │   resolve on last step)          │
                    └────────────┬────────────────────┘
                                 │ MarketState
                    ┌────────────▼────────────────────┐
                    │         TradingEnv               │
                    │  (Gymnasium, paper execution,    │
                    │   position tracking, rewards)    │
                    └────┬──────────────┬─────────────┘
                         │              │
              MarketState│              │Action
                         │              │
          ┌──────────────▼──┐    ┌──────▼──────────────┐
          │   ClaudeAgent   │    │   RL / Rule / Random │
          │  (language in,  │    │  (obs vector in,     │
          │   JSON out)     │    │   discrete out)      │
          └─────────────────┘    └─────────────────────┘
```

---

## CLI

```bash
# Sync resolved markets from Polymarket
uv run python main.py sync --limit 200

# Train a rule-based agent (--signals enables Claude signal analysis)
uv run python main.py train --agent rule --episodes 100 --signals

# Train RL (PPO) and evaluate
uv run python main.py train-rl --timesteps 500000 --algorithm PPO

# Train Claude with learning loop (strategy memory across batches)
uv run python main.py train-with-memory --batches 5 --episodes 50 --signals

# Compare agents head-to-head
uv run python main.py compare --agents rule --agents claude --episodes 50

# End-to-end pipeline: sync → split → train → validate → rank → recommend
uv run python main.py pipeline --signals

# Replay a single episode with price chart, trades, P&L, reasoning
uv run python main.py replay --agent claude --signals

# Run agent against live Polymarket (--signals enables web search + Claude)
uv run python main.py live --agent claude --markets <id> --mode paper --signals

# Start web UI + API
uv run python main.py serve
```

---

## Key Design Decisions

**Why `MarketState` instead of a numpy obs vector at the agent boundary?**
Claude agents need language, not floats. RL agents need floats, not language. By passing the full state object and letting each agent extract what it needs, neither is compromised.

**Why conversation history within episodes for Claude?**
A memoryless Claude that sees only the current tick can't reason about its own position or track how the market has evolved. Multi-turn history within an episode makes Claude a proper stateful agent.

**Why is RL optional / additive?**
The Claude agent and eval harness are fully functional without RL. RL plugs in as one more `BaseAgent` subclass — it doesn't change the env, the runner, or anything else.

**Why historical replay instead of live data for training?**
Resolved markets provide ground truth (the outcome is known), giving a clear learning signal. Live markets have unknown outcomes, making reward evaluation impossible until resolution.
