"""FastAPI backend for the trading playground web interface."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from polymarket_playground.core.action import Action, DIRECTION_MAP
from polymarket_playground.core.env import TradingEnv
from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import EpisodeResult
from polymarket_playground.data.historical_loader import HistoricalLoader
from polymarket_playground.data.polymarket_client import PolymarketClient
from polymarket_playground.markets.polymarket import PolymarketAdapter
from polymarket_playground.markets.btc_price import BTCPriceAdapter
from polymarket_playground.markets.elections import ElectionAdapter
from polymarket_playground.markets.sports import SportsAdapter
from polymarket_playground.markets.macro import MacroAdapter

logger = logging.getLogger(__name__)

# Shared Polymarket adapter (fetches markets once, reused across sessions)
_polymarket_client = PolymarketClient()
_polymarket_adapter: PolymarketAdapter | None = None
# Resolved market metadata from historical loader (populated at startup)
_historical_markets: dict[str, dict] = {}


def _get_polymarket_adapter() -> PolymarketAdapter | None:
    """Lazy-init the shared PolymarketAdapter."""
    global _polymarket_adapter
    if _polymarket_adapter is None:
        try:
            _polymarket_adapter = PolymarketAdapter(client=_polymarket_client)
            if not _polymarket_adapter.get_markets():
                logger.warning("No Polymarket markets available, falling back to synthetic")
                _polymarket_adapter = None
        except Exception as e:
            logger.warning("Failed to init PolymarketAdapter: %s", e)
            _polymarket_adapter = None
    return _polymarket_adapter


# Fallback synthetic adapters
SYNTHETIC_ADAPTERS = {
    "btc": BTCPriceAdapter,
    "elections": ElectionAdapter,
    "sports": SportsAdapter,
    "macro": MacroAdapter,
}

AGENT_TYPES = ["rule", "claude", "random", "rl"]


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TradingSession:
    """A single trading session (one env + one agent or manual)."""

    def __init__(self, session_id: str, market: str, agent_type: str | None = None):
        self.session_id = session_id
        self.market = market
        self.agent_type = agent_type

        # Use real Polymarket data if market is a Polymarket market ID
        poly = _get_polymarket_adapter()
        if poly and (market in poly.get_markets() or market in _historical_markets):
            self.adapter = poly
            self.is_polymarket = True
        elif market in SYNTHETIC_ADAPTERS:
            self.adapter = SYNTHETIC_ADAPTERS[market]()
            self.is_polymarket = False
        else:
            # Assume it's a Polymarket market ID we haven't cached yet
            if poly:
                self.adapter = poly
                self.is_polymarket = True
            else:
                self.adapter = BTCPriceAdapter()
                self.is_polymarket = False

        self.env = TradingEnv(
            adapter=self.adapter,
            config={"max_steps": 100},
        )

        # Share the pre-warmed price cache with this session's loader
        if self.is_polymarket and _shared_loader and isinstance(self.env.data, HistoricalLoader):
            self.env.data._price_cache = _shared_loader._price_cache
            self.env.data._market_meta = _shared_loader._market_meta
            self.env.data._token_cache = _shared_loader._token_cache

        # For Polymarket markets, pin the loader to this specific market
        if self.is_polymarket:
            if hasattr(self.env.data, 'set_market'):
                self.env.data.set_market(market)
            self.env.reset()
        else:
            self.env.reset()

        self.history: list[dict] = []
        self.step_count = 0
        self.running = False
        self._agent = None

    def get_state_dict(self) -> dict:
        """Serialize current state for the frontend."""
        state = self.env.state
        pos = self.env.position
        return {
            "session_id": self.session_id,
            "market": self.market,
            "agent_type": self.agent_type,
            "step": self.step_count,
            "running": self.running,
            "state": {
                "market_id": state.market_id,
                "market_type": state.market_type,
                "question": state.question,
                "yes_price": state.yes_price,
                "no_price": state.no_price,
                "yes_bid": state.yes_bid,
                "yes_ask": state.yes_ask,
                "no_bid": state.no_bid,
                "no_ask": state.no_ask,
                "spread": state.spread,
                "book_depth": state.book_depth,
                "time_elapsed": state.time_elapsed,
                "time_remaining": state.time_remaining,
                "signals": state.signals,
                "is_resolved": state.is_resolved,
            },
            "position": {
                "yes_shares": pos.yes_shares if pos else 0,
                "no_shares": pos.no_shares if pos else 0,
                "net_exposure": pos.net_exposure if pos else 0,
                "cost_basis": pos.cost_basis if pos else 0,
            },
            "history": self.history[-50:],
            "total_pnl": sum(h.get("reward", 0) for h in self.history),
        }

    def execute_action(self, action_dict: dict) -> dict:
        """Execute a single action and return the result."""
        action = Action.from_dict(action_dict)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        entry = {
            "step": self.step_count,
            "action": action.direction_name,
            "size": action.size,
            "reasoning": action.reasoning,
            "reward": reward,
            "yes_price": self.env.state.yes_price,
            "terminated": terminated,
            "truncated": truncated,
        }
        self.history.append(entry)

        return {
            "entry": entry,
            "done": terminated or truncated,
            **self.get_state_dict(),
        }

    def reset(self):
        """Reset the environment for a new episode."""
        if self.is_polymarket and hasattr(self.env.data, 'set_market'):
            self.env.data.set_market(self.market)
        self.env.reset()
        self.history = []
        self.step_count = 0
        self.running = False

    def get_agent(self):
        """Lazy-load the agent."""
        if self._agent is not None:
            return self._agent

        if self.agent_type == "rule":
            from polymarket_playground.agents.rule_agent import RuleAgent
            self._agent = RuleAgent()
        elif self.agent_type == "claude":
            from polymarket_playground.agents.claude_agent import ClaudeAgent
            self._agent = ClaudeAgent(adapter=self.adapter)
        elif self.agent_type == "random":
            from polymarket_playground.agents.rl_agent import RLAgent
            self._agent = RLAgent(policy=None)
        elif self.agent_type == "rl":
            from polymarket_playground.agents.rl_agent import RLAgent
            self._agent = RLAgent(policy=None)
        return self._agent


sessions: dict[str, TradingSession] = {}


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    def __init__(self):
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self.connections.setdefault(session_id, []).append(ws)

    def disconnect(self, session_id: str, ws: WebSocket):
        if session_id in self.connections:
            self.connections[session_id] = [
                c for c in self.connections[session_id] if c != ws
            ]

    async def broadcast(self, session_id: str, data: dict):
        if session_id not in self.connections:
            return
        dead = []
        for ws in self.connections[session_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections[session_id].remove(ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

# Shared historical loader (pre-warms price cache at startup, reused by sessions)
_shared_loader: HistoricalLoader | None = None


def _get_shared_loader(adapter) -> HistoricalLoader:
    """Get or create the shared HistoricalLoader with pre-fetched price data."""
    global _shared_loader, _historical_markets
    if _shared_loader is None:
        _shared_loader = HistoricalLoader(adapter, max_steps=100)
        _historical_markets = dict(_shared_loader._market_meta)
        logger.info("Pre-loaded %d markets for historical replay", len(_historical_markets))
    return _shared_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the Polymarket adapter and historical loader
    poly = _get_polymarket_adapter()
    if poly:
        _get_shared_loader(poly)
    yield

app = FastAPI(title="Trading Playground", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/markets")
async def list_markets():
    """List available markets — resolved Polymarket + synthetic fallbacks."""
    markets = []

    # Resolved Polymarket markets (from historical loader cache)
    if _historical_markets:
        for mid, m in _historical_markets.items():
            yes_price, no_price = PolymarketClient.parse_outcome_prices(m)
            markets.append({
                "id": mid,
                "name": m.get("question", mid)[:80],
                "type": "polymarket",
                "yes_price": yes_price,
                "no_price": no_price,
                "volume_24hr": m.get("volume24hr", 0),
                "liquidity": m.get("liquidityNum", 0),
                "resolved": bool(m.get("closed", False)),
                "signals": PolymarketAdapter.SIGNALS,
            })

    # Synthetic fallbacks
    markets.append({"id": "btc", "name": "BTC Price (Synthetic)", "type": "synthetic", "signals": BTCPriceAdapter.SIGNALS})
    markets.append({"id": "elections", "name": "Elections (Synthetic)", "type": "synthetic", "signals": ElectionAdapter.SIGNALS})
    markets.append({"id": "sports", "name": "Sports (Synthetic)", "type": "synthetic", "signals": SportsAdapter.SIGNALS})
    markets.append({"id": "macro", "name": "Macro (Synthetic)", "type": "synthetic", "signals": MacroAdapter.SIGNALS})

    return {"markets": markets}


@app.get("/api/agents")
async def list_agents():
    return {"agents": AGENT_TYPES}


@app.post("/api/sessions")
async def create_session(body: dict[str, Any]):
    market = body.get("market", "btc")
    agent_type = body.get("agent_type")
    session_id = f"s_{random.randint(10000, 99999)}_{int(time.time())}"

    session = TradingSession(session_id, market, agent_type)
    sessions[session_id] = session
    return session.get_state_dict()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        return {"error": "Session not found"}
    return sessions[session_id].get_state_dict()


@app.post("/api/sessions/{session_id}/action")
async def session_action(session_id: str, body: dict[str, Any]):
    if session_id not in sessions:
        return {"error": "Session not found"}

    session = sessions[session_id]
    result = session.execute_action(body)

    await manager.broadcast(session_id, {"type": "step", **result})
    return result


@app.post("/api/sessions/{session_id}/reset")
async def session_reset(session_id: str):
    if session_id not in sessions:
        return {"error": "Session not found"}

    sessions[session_id].reset()
    return sessions[session_id].get_state_dict()


@app.post("/api/sessions/{session_id}/run")
async def run_agent(session_id: str, body: dict[str, Any] | None = None):
    """Start an agent running in the background."""
    if session_id not in sessions:
        return {"error": "Session not found"}

    session = sessions[session_id]
    if session.agent_type is None:
        return {"error": "No agent configured for manual session"}

    steps = (body or {}).get("steps", 50)
    delay = (body or {}).get("delay", 0.3)
    session.running = True

    async def _run():
        agent = session.get_agent()
        agent.on_reset(session.env.state)
        for _ in range(steps):
            if not session.running:
                break
            action = agent.act(session.env.state)
            result = session.execute_action({
                "action": action.direction_name,
                "size": action.size,
                "reasoning": action.reasoning,
            })
            await manager.broadcast(session_id, {"type": "step", **result})
            if result.get("done"):
                break
            await asyncio.sleep(delay)
        session.running = False
        await manager.broadcast(session_id, {"type": "done", **session.get_state_dict()})

    asyncio.create_task(_run())
    return {"status": "started", "session_id": session_id}


@app.post("/api/sessions/{session_id}/stop")
async def stop_agent(session_id: str):
    if session_id not in sessions:
        return {"error": "Session not found"}
    sessions[session_id].running = False
    return {"status": "stopped"}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        sessions[session_id].running = False
        del sessions[session_id]
    return {"status": "deleted"}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        if session_id in sessions:
            await websocket.send_json({
                "type": "state",
                **sessions[session_id].get_state_dict(),
            })

        while True:
            data = await websocket.receive_json()
            if data.get("type") == "action" and session_id in sessions:
                result = sessions[session_id].execute_action(data)
                await manager.broadcast(session_id, {"type": "step", **result})
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)
