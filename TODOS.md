# TODOs

## Roadmap: From Infrastructure to Alpha

### P1 — Must Have (blocks making money)

- [x] **Information Signal Pipeline** — `data/signal_provider.py`: `SignalProvider` ABC, `HistoricalSignalProvider` (Claude analysis, SQLite-cached), `LiveSignalProvider` (Tavily + Claude, TTL-cached). Enriched signals wired through `TradingEnv` → `HistoricalLoader` → `MarketState.signals`. Claude agent's `_build_context()` surfaces enriched signals.
  - Effort: L (human: 2w / CC: 2h)
  - Depends on: nothing

- [x] **Walk-Forward Backtesting** — `market_cache.get_market_ids_by_date_range()` and `get_earliest_latest_timestamps()` enable temporal splits. `HistoricalLoader.set_market_ids()` restricts episode pool. Wired into `pipeline` CLI command.
  - Effort: M (human: 1w / CC: 1h)
  - Depends on: nothing

- [x] **Strategy Validation & Statistical Significance** — `eval/metrics.py`: `t_test_returns()`, `bootstrap_sharpe_ci()`, `validate_strategy()`, `compare_vs_random()`, `ValidationResult` dataclass. Wired into `pipeline` CLI command.
  - Effort: S (human: 3d / CC: 30m)
  - Depends on: Walk-Forward Backtesting

### P2 — Should Have (blocks scaling)

- [x] **Claude Agent Learning Loop** — `agents/strategy_memory.py`: `StrategyMemory` class with `load()`, `update()`, `clear()`. Uses Haiku for summarization. `ClaudeAgent` loads strategy memory into system prompt. `train-with-memory` CLI command runs batched training with memory updates.
  - Effort: M (human: 1w / CC: 45m)
  - Depends on: nothing

- [x] **Live Deployment Hardening** — `py-clob-client` added as optional dep in `pyproject.toml`. `LiveExecutor.get_balance()` implemented via CLOB API `/balance` endpoint. `tavily-python` as optional signal dep.
  - Effort: S (human: 3d / CC: 1h)
  - Depends on: Strategy Validation

- [x] **Automated Strategy Pipeline** — `python main.py pipeline`: sync → temporal split → train all agents → validate (t-test + bootstrap) → compare vs random → rank → recommend (DEPLOY/SKIP).
  - Effort: M (human: 1w / CC: 1.5h)
  - Depends on: Signal Pipeline, Backtesting, Validation, Learning Loop

### P3 — Nice to Have (improves velocity)

- [x] **Episode Replay & Reasoning Viewer** — `python main.py replay`: runs an episode and displays ASCII price chart, cumulative P&L curve, trade log with reasoning, and final state summary. Supports `--market-id`, `--agent`, `--category` filters.
  - Effort: M (human: 4d / CC: 45m)
  - Depends on: nothing

### Implementation Decisions (from eng review 2026-03-17)

1. **Extend existing files, don't create new modules** — Validation goes into `eval/metrics.py`, backtesting goes into `data/historical_loader.py`, pipeline is a new CLI command in `main.py`. Only the signal pipeline (`data/signal_provider.py`) and strategy memory (`agents/strategy_memory.py`) get new files. A shared Claude helper goes into `agents/claude_helper.py`.

2. **Historical training: Claude-only analysis (no web search)** — For historical/resolved markets, use Claude to analyze the market question + price pattern. Web search signals are live-mode only (you can't search for news about events that resolved months ago and get meaningful results).

3. **Strategy memory: markdown file** — `strategy_memory.md` read into Claude's system prompt. Human-readable, inspectable, editable. Not SQLite.

4. **Model tiering: Haiku for cheap tasks, Sonnet for trading** — Signal analysis and strategy summarization use `claude-haiku-4-5` (~10x cheaper). Trading agent uses `claude-sonnet-4-6` for stronger reasoning.

### Known Bugs

- [x] `LiveExecutor.get_balance()` — implemented via CLOB API `/balance` endpoint with Bearer token auth
- [x] `py-clob-client` — added as optional dep `live = ["py-clob-client>=0.1.0"]` in `pyproject.toml`
