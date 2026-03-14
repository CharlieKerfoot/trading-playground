# Trading Playground — Architecture

## Design Philosophy

Three hard requirements drive every decision:

1. **Agent-agnostic** — Claude agents, RL policies, rule-based bots all plug in identically. No agent knows or cares how the env works underneath.

2. **Market-agnostic** — BTC contracts, election markets, sports, macro events. Market-specific
   logic lives in swappable `MarketAdapter` plugins, not the core env.

3. **Training-focused** — The environment uses historical replay data and paper execution only. All agents train and evaluate against resolved market data.

---

## Directory Structure

```
polymarket_playground/
│
├── core/
│   ├── env.py                  # Base TradingEnv (Gymnasium-compatible)
│   ├── state.py                # MarketState — universal market snapshot
│   ├── position.py             # Position tracking, P&L, settlement
│   ├── action.py               # Action schema (direction, size, order type)
│   └── types.py                # Shared types: Fill, Episode, Resolution
│
├── markets/                    # Market-specific adapters (plug-in system)
│   ├── base_adapter.py         # Abstract MarketAdapter interface
│   ├── btc_price.py            # BTC prediction contracts + Binance signals
│   ├── elections.py            # Political markets
│   ├── sports.py               # Sports outcome markets
│   └── macro.py                # Fed rate, CPI, earnings markets
│
├── data/
│   ├── polymarket_client.py    # Polymarket CLOB REST + WebSocket
│   ├── external_feeds.py       # Generic external signal fetcher (Binance, news, etc.)
│   ├── historical_loader.py    # Replay from Polymarket history + Kaggle dump
│   └── episode_store.py        # Persistent episode cache (SQLite or Parquet)
│
├── execution/
│   ├── base_executor.py        # Abstract executor interface
│   ├── paper_executor.py       # Simulated fills with slippage model
│   └── slippage_model.py       # Fill simulation from order book depth
│
├── agents/
│   ├── base_agent.py           # Abstract agent interface
│   ├── claude_agent.py         # Claude API agent (structured prompt/response)
│   ├── rl_agent.py             # RL policy wrapper (SB3/CleanRL compatible)
│   └── rule_agent.py           # Deterministic rule-based baseline
│
├── eval/
│   ├── runner.py               # Episode runner (single agent or multi-agent)
│   ├── metrics.py              # Sharpe, win rate, avg P&L, spread capture
│   ├── compare.py              # Head-to-head agent comparison harness
│   └── visualize.py            # P&L curves, position timelines
│
├── config/
│   ├── base.yaml               # Default env config
│   ├── btc.yaml                # BTC market config
│   └── elections.yaml          # Election market config
│
└── cli.py                      # Entry point: run agents, replay, compare
```

---

## Core Abstractions

### `MarketState` — the universal snapshot

Every market adapter produces this same struct at each timestep. Agents see only this.

```python
@dataclass
class MarketState:
    # Contract identity
    market_id:       str
    market_type:     str           # "btc_price" | "election" | "sports" | ...
    question:        str           # Human-readable: "BTC above $95k at 3pm UTC?"
    resolution_time: datetime

    # Polymarket prices
    yes_price:       float         # [0, 1]
    no_price:        float         # [0, 1]
    yes_bid:         float
    yes_ask:         float
    no_bid:          float
    no_ask:          float
    spread:          float
    book_depth:      float         # liquidity at best bid/ask (USDC)

    # Time
    time_elapsed:    float         # fraction of contract life elapsed [0,1]
    time_remaining:  float         # seconds until resolution
    timestamp:       datetime

    # External signals (market-type specific, keyed dict)
    signals:         dict[str, float]

    # Resolution (None until market closes)
    is_resolved:     bool
    resolution:      Optional[float]   # 1.0 = YES, 0.0 = NO
```

### `MarketAdapter` — plug-in market logic

```python
class MarketAdapter(ABC):
    @abstractmethod
    def get_markets(self) -> list[str]: ...

    @abstractmethod
    def fetch_signals(self, market_id: str, timestamp: datetime) -> dict[str, float]: ...

    @abstractmethod
    def build_agent_context(self, state: MarketState) -> str: ...

    @abstractmethod
    def select_episodes(self, filters: dict) -> list[Episode]: ...
```

**Concrete adapters:** BTCPriceAdapter, ElectionAdapter, SportsAdapter, MacroAdapter.

Adding a new market type = adding one file in `markets/`. Zero core changes.

---

## Base Environment: `core/env.py`

```python
class TradingEnv(gym.Env):
    """
    Market-agnostic, agent-agnostic Gymnasium environment.

    Parameterized entirely by:
      - adapter: which MarketAdapter to use
      - config:  position limits, step size, episode filters
    """

    def __init__(self, adapter: MarketAdapter, config: dict = None):
        self.adapter  = adapter
        self.executor = PaperExecutor(config=config)
        self.data     = HistoricalLoader(adapter, max_steps=config.get("max_steps", 50))
        self.position = None
        self.state    = None
```

---

## Agent Interface

Every agent implements this — RL policies, Claude, rule-based:

```python
class BaseAgent(ABC):
    @abstractmethod
    def act(self, state: MarketState) -> Action: ...

    def on_reset(self, state: MarketState):
        pass

    def on_episode_end(self, result: EpisodeResult):
        pass
```

---

## CLI: `cli.py`

```bash
# Run Claude agent on BTC markets
uv run python cli.py run \
  --agent claude \
  --market btc \
  --episodes 200

# Run RL training on election markets
uv run python cli.py train \
  --market elections \
  --timesteps 1000000

# Compare Claude vs random on sports markets
uv run python cli.py compare \
  --agents claude --agents random \
  --market sports \
  --episodes 100 \
  --seed 42
```

---

## Data Flow Summary

```
                    ┌─────────────────────────────────┐
                    │         MarketAdapter            │
                    │  (btc / elections / sports / ...) │
                    └────────────┬────────────────────┘
                                 │ fetch_signals()
                    ┌────────────▼────────────────────┐
                    │      HistoricalLoader            │
                    └────────────┬────────────────────┘
                                 │ MarketState
                    ┌────────────▼────────────────────┐
                    │         TradingEnv               │
                    │  (Gymnasium core, market-agnostic)│
                    └────┬──────────────┬─────────────┘
                         │              │
              MarketState│              │Action
                         │              │
          ┌──────────────▼──┐    ┌──────▼──────────────┐
          │   ClaudeAgent   │    │     RL / Rule Agent  │
          │  (language in,  │    │  (obs vector in,     │
          │   JSON out)     │    │   int/float out)     │
          └─────────────────┘    └─────────────────────┘
                         │              │
                    ┌────▼──────────────▼─────────────┐
                    │         EpisodeRunner            │
                    │    (drives loop, logs results)   │
                    └─────────────────────────────────┘
```

---

## Key Design Decisions

**Why `MarketState` instead of a numpy obs vector at the agent boundary?**
Claude agents need language, not floats. RL agents need floats, not language. By passing the
full state object and letting each agent extract what it needs, neither is compromised.

**Why conversation history within episodes for Claude?**
A memoryless Claude that sees only the current tick can't reason about its own position or
track how the market has evolved. Multi-turn history within an episode makes Claude a proper
stateful agent.

**Why adapter-level signal fetching instead of a global feature store?**
Different market types need radically different external signals. Adapter-scoped fetching
keeps each market type self-contained and independently testable.

**Why is RL optional / additive?**
The Claude agent and eval harness are fully functional without RL. RL plugs in as one more
`BaseAgent` subclass — it doesn't change the env, the runner, or anything else.
