"""FastAPI backend for the batch training playground."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from polymarket_playground.core.env import TradingEnv
from polymarket_playground.data.market_cache import MarketCache
from polymarket_playground.training.run_manager import RunManager, RunStatus, TrainingRun

logger = logging.getLogger(__name__)

AGENT_TYPES = {
    "rule": {
        "name": "Rule Agent",
        "description": "Simple threshold-based strategy (buy YES < 0.3, buy NO > 0.7)",
        "config_params": {},
    },
    "claude": {
        "name": "Claude Agent",
        "description": "Reasons over market context using Claude API",
        "config_params": {"model": "claude-sonnet-4-6"},
    },
    "random": {
        "name": "Random Agent",
        "description": "Random actions for baseline comparison",
        "config_params": {},
    },
    "rl": {
        "name": "RL Agent",
        "description": "Reinforcement learning policy (PPO/A2C). Train first, then select a saved model.",
        "config_params": {"model_path": "", "algorithm": "PPO"},
    },
}

# Globals initialized at startup
_cache: MarketCache | None = None
_run_manager: RunManager | None = None
_sync_task: asyncio.Task | None = None
_sync_status: dict = {"running": False, "fetched": 0, "skipped": 0, "errors": 0}
_rl_train_task: asyncio.Task | None = None
_rl_train_status: dict = {"running": False, "timesteps": 0, "algorithm": "", "save_path": ""}


def _get_cache() -> MarketCache:
    global _cache
    if _cache is None:
        _cache = MarketCache()
    return _cache


def _get_run_manager() -> RunManager:
    global _run_manager
    if _run_manager is None:
        _run_manager = RunManager()
    return _run_manager


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    def __init__(self):
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, run_id: str, ws: WebSocket):
        await ws.accept()
        self.connections.setdefault(run_id, []).append(ws)

    def disconnect(self, run_id: str, ws: WebSocket):
        if run_id in self.connections:
            self.connections[run_id] = [
                c for c in self.connections[run_id] if c != ws
            ]

    async def broadcast(self, run_id: str, data: dict):
        if run_id not in self.connections:
            return
        dead = []
        for ws in self.connections[run_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections[run_id].remove(ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize cache at startup
    _get_cache()
    _get_run_manager()
    stats = _get_cache().stats()
    if stats["total_markets"] == 0:
        logger.info("Market cache is empty. Use POST /api/data/sync to fetch markets.")
    else:
        logger.info(
            "Market cache: %d markets, %d price points",
            stats["total_markets"],
            stats["total_price_points"],
        )
    yield


app = FastAPI(title="Trading Playground", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------


@app.get("/api/data/stats")
async def data_stats():
    """Return cached market stats."""
    return _get_cache().stats()


@app.post("/api/data/sync")
async def data_sync(body: dict[str, Any] | None = None):
    """Trigger a background sync of resolved markets."""
    global _sync_task, _sync_status

    if _sync_status["running"]:
        return {"status": "already_running", **_sync_status}

    body = body or {}
    limit = body.get("limit", 100)

    _sync_status = {"running": True, "fetched": 0, "skipped": 0, "errors": 0}

    async def _run_sync():
        global _sync_status
        try:
            cache = _get_cache()

            def on_progress(fetched, skipped, errors):
                _sync_status.update(
                    fetched=fetched, skipped=skipped, errors=errors
                )

            # Run sync in a thread to avoid blocking the event loop
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: cache.sync(limit=limit, on_progress=on_progress),
            )
            _sync_status.update(**result, running=False)
        except Exception as e:
            _sync_status["running"] = False
            _sync_status["error"] = str(e)
            logger.exception("Sync failed")

    _sync_task = asyncio.create_task(_run_sync())
    return {"status": "started"}


@app.get("/api/data/sync/status")
async def sync_status():
    """Return current sync progress."""
    return _sync_status


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------


@app.post("/api/runs")
async def create_run(body: dict[str, Any]):
    """Create and start a training run."""
    rm = _get_run_manager()
    cache = _get_cache()

    agent_type = body.get("agent_type", "rule")
    num_episodes = body.get("num_episodes", 50)
    agent_config = body.get("agent_config", {})

    run = rm.create_run(
        agent_type=agent_type,
        num_episodes=num_episodes,
        agent_config=agent_config,
    )

    # Create env with cache
    env = TradingEnv(cache=cache, config={"max_steps": 100})

    async def on_progress(r: TrainingRun):
        await manager.broadcast(
            r.id,
            {
                "type": "progress",
                "current_episode": r.current_episode,
                "num_episodes": r.num_episodes,
                "latest_reward": round(r.results[-1].total_reward, 6) if r.results else 0,
                "metrics": r.metrics(),
            },
        )

    await rm.start_run(run.id, env, on_progress=on_progress)
    return run.to_dict()


@app.get("/api/runs")
async def list_runs():
    """List all training runs."""
    rm = _get_run_manager()
    return {"runs": [r.to_dict() for r in rm.list_runs()]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get run details + results."""
    rm = _get_run_manager()
    run = rm.get_run(run_id)
    if not run:
        return {"error": "Run not found"}
    return run.to_dict()


@app.get("/api/runs/{run_id}/metrics")
async def get_run_metrics(run_id: str):
    """Get computed metrics for a run."""
    rm = _get_run_manager()
    run = rm.get_run(run_id)
    if not run:
        return {"error": "Run not found"}
    return run.metrics()


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Cancel and delete a run."""
    rm = _get_run_manager()
    if rm.delete_run(run_id):
        return {"status": "deleted"}
    return {"error": "Run not found"}


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------


@app.get("/api/agents")
async def list_agents():
    """List available agent types with configurable params."""
    return {"agents": AGENT_TYPES}


# ---------------------------------------------------------------------------
# RL Training endpoints
# ---------------------------------------------------------------------------


@app.post("/api/train/rl")
async def train_rl(body: dict[str, Any]):
    """Start RL training in the background."""
    global _rl_train_task, _rl_train_status

    if _rl_train_status["running"]:
        return {"status": "already_running", **_rl_train_status}

    timesteps = body.get("timesteps", 100_000)
    algorithm = body.get("algorithm", "PPO").upper()
    save_path = body.get("save_path", f"models/{algorithm.lower()}_agent")

    _rl_train_status = {
        "running": True,
        "timesteps": timesteps,
        "algorithm": algorithm,
        "save_path": save_path,
        "progress": 0,
        "error": None,
    }

    async def _run_training():
        global _rl_train_status
        try:
            from pathlib import Path
            from polymarket_playground.eval.train import RLTrainer

            cache = _get_cache()
            env = TradingEnv(cache=cache, config={"max_steps": 100})

            def do_train():
                trainer = RLTrainer(env, algorithm=algorithm)
                trainer.train(timesteps=timesteps)
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                trainer.save(save_path)

            await asyncio.get_event_loop().run_in_executor(None, do_train)
            _rl_train_status.update(running=False, progress=timesteps)
        except Exception as e:
            _rl_train_status["running"] = False
            _rl_train_status["error"] = str(e)
            logger.exception("RL training failed")

    _rl_train_task = asyncio.create_task(_run_training())
    return {"status": "started", **_rl_train_status}


@app.get("/api/train/rl/status")
async def rl_train_status():
    """Return RL training progress."""
    return _rl_train_status


@app.get("/api/models")
async def list_models():
    """List saved RL models."""
    from pathlib import Path

    models_dir = Path("models")
    if not models_dir.exists():
        return {"models": []}

    models = []
    for f in models_dir.iterdir():
        if f.suffix == ".zip" or (f.is_file() and not f.suffix):
            name = f.stem if f.suffix == ".zip" else f.name
            models.append({
                "name": name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
            })
    return {"models": models}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws/runs/{run_id}")
async def websocket_run(websocket: WebSocket, run_id: str):
    """Live progress streaming for a training run."""
    await manager.connect(run_id, websocket)
    try:
        # Send current state immediately
        rm = _get_run_manager()
        run = rm.get_run(run_id)
        if run:
            await websocket.send_json({"type": "state", **run.to_dict()})

        # Keep connection alive, listen for client messages
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "cancel":
                rm.cancel_run(run_id)
                await websocket.send_json({"type": "cancelled"})
    except WebSocketDisconnect:
        manager.disconnect(run_id, websocket)
