"""
Oceanus 3D Mission Control — FastAPI + WebSocket backend
Serves the 3D HTML frontend and streams live environment state.

Run: uvicorn dashboard.server:app --host 0.0.0.0 --port 8000 --reload
Then open: http://localhost:8000
"""
import sys, os, json, asyncio, time
from typing import Dict, List, Optional, Set
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from oceanus.models import OceanusEnv
from oceanus.adversary import AdversaryAgent
from oceanus.runner import OceanusRunner, MockASVAgent, MockPolicyAgent
from oceanus.physics import GRID_SIZE

app = FastAPI(title="Oceanus Mission Control", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Global simulation state ────────────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.env: Optional[OceanusEnv] = None
        self.adversary: Optional[AdversaryAgent] = None
        self.runner: Optional[OceanusRunner] = None
        self.obs_all: Optional[Dict] = None
        self.running: bool = False
        self.done: bool = False
        self.total_reward: float = 0.0
        self.reward_history: List[float] = []
        self.bio_history: List[float] = []
        self.net_history: List[int] = []
        self.event_log: List[Dict] = []
        self.episode_count: int = 0
        self.cfg = {"seed": 42, "max_steps": 120, "chaos_interval": 25, "chaos_enabled": True, "step_delay": 0.08}

    def reset(self, seed=42, max_steps=120, chaos_interval=25):
        self.env = OceanusEnv(seed=seed, max_steps=max_steps)
        self.adversary = AdversaryAgent(inject_interval=chaos_interval, seed=seed)
        self.runner = OceanusRunner(self.env, self.adversary, use_mock=True, verbose=False)
        self.obs_all = self.env.reset()
        self.running = False
        self.done = False
        self.total_reward = 0.0
        self.reward_history = []
        self.bio_history = []
        self.net_history = []
        self.event_log = []
        self.episode_count += 1

    def get_frame(self) -> Dict:
        """Serialize current state for WebSocket broadcast."""
        if not self.env or not self.env.state:
            return {}
        s = self.env.state
        return {
            "type": "state",
            "step": s.step_count,
            "max_steps": s.max_steps,
            "grid": s.grid.tolist(),
            "asvs": {
                aid: {"row": a.row, "col": a.col, "battery": a.battery,
                      "nets_cleaned": a.nets_cleaned,
                      "on_net": bool(s.grid[a.row, a.col] > 0),
                      "sector": s.get_sector(a.row, a.col)}
                for aid, a in s.asvs.items()
            },
            "ghost_nets": [{"row": n.row, "col": n.col, "density": n.density}
                           for n in s.ghost_nets],
            "biodiversity": round(s.biodiversity_index, 2),
            "active_nets": len(s.ghost_nets),
            "total_cleaned": s.total_cleaned,
            "treaty_status": s.treaty_status,
            "net_spawn_rate": s.net_spawn_rate,
            "wind_vector": list(s.wind_vector),
            "total_reward": round(self.total_reward, 1),
            "reward_history": self.reward_history[-60:],
            "bio_history": self.bio_history[-60:],
            "net_history": self.net_history[-60:],
            "event_log": self.event_log[-20:],
            "running": self.running,
            "done": self.done,
            "episode": self.episode_count,
            "schema_version": s.schema_version,
        }

sim = SimState()
sim.reset()

# ── WebSocket connection manager ───────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: Dict):
        msg = json.dumps(data)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.active -= dead

manager = ConnectionManager()

# ── Background simulation loop ─────────────────────────────────────────────────
async def simulation_loop():
    while True:
        if sim.running and not sim.done and sim.obs_all and sim.runner:
            # Collect actions
            actions = {}
            for agent_id, agent in sim.runner.agents.items():
                if agent_id in sim.obs_all:
                    ao = sim.obs_all[agent_id]
                    actions[agent_id] = agent.act(ao["prompt"], ao["observation"])

            # Step
            obs_all, rewards, done, info = sim.env.step(actions)
            sim.obs_all = obs_all
            sim.done = done
            step_reward = rewards.get("__total__", 0.0)
            sim.total_reward += step_reward
            sim.reward_history.append(round(step_reward, 2))
            sim.bio_history.append(round(info["biodiversity"], 1))
            sim.net_history.append(info["active_nets"])

            # Build events
            events = []
            for c in info.get("cleaned", []):
                events.append({"type": "clean", "msg": f"CLEANED: {c}"})
            for te in info.get("treaty_events", []):
                events.append({"type": "treaty", "msg": f"TREATY: {te}"})
            for b in info.get("broadcasts", []):
                events.append({"type": "broadcast", "msg": f"BROADCAST: {b}"})

            if sim.cfg["chaos_enabled"] and sim.adversary.should_inject(info["step"]):
                for c in sim.adversary.inject(sim.env.state):
                    events.append({"type": "chaos", "msg": f"CHAOS: {c}"})

            events.append({"type": "step", "msg": f"[{info['step']:3d}] Nets:{info['active_nets']} | Bio:{info['biodiversity']:.1f}% | R:{step_reward:+.1f}"})
            sim.event_log.extend(events)

            if done:
                sim.running = False
                summary = sim.env.get_episode_summary()
                sim.event_log.append({"type": "done", "msg": f"EPISODE COMPLETE | Bio:{summary['biodiversity_final']:.1f}% | Cleaned:{summary['total_cleaned']} | Treaty:{summary['treaty_status']}"})

            # Broadcast to all connected clients
            await manager.broadcast(sim.get_frame())
            await asyncio.sleep(sim.cfg["step_delay"])
        else:
            await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup():
    asyncio.create_task(simulation_loop())

# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # Send current state immediately on connect
    await websocket.send_text(json.dumps(sim.get_frame()))
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            cmd = msg.get("cmd")

            if cmd == "start":
                seed = msg.get("seed", sim.cfg["seed"])
                max_steps = msg.get("max_steps", sim.cfg["max_steps"])
                chaos_interval = msg.get("chaos_interval", sim.cfg["chaos_interval"])
                sim.cfg["chaos_enabled"] = msg.get("chaos_enabled", True)
                sim.cfg["step_delay"] = msg.get("step_delay", 0.08)
                sim.reset(seed=seed, max_steps=max_steps, chaos_interval=chaos_interval)
                sim.running = True
                await manager.broadcast({"type": "control", "status": "started"})

            elif cmd == "stop":
                sim.running = False
                await manager.broadcast({"type": "control", "status": "stopped"})

            elif cmd == "reset":
                sim.reset()
                await manager.broadcast(sim.get_frame())

            elif cmd == "get_state":
                await websocket.send_text(json.dumps(sim.get_frame()))

            elif cmd == "get_replay":
                mode = msg.get("mode", "trained")
                path = f"data/{mode}_episode.json"
                if os.path.exists(path):
                    with open(path) as f:
                        replay = json.load(f)
                    await websocket.send_text(json.dumps({"type": "replay_data", "data": replay}))
                else:
                    await websocket.send_text(json.dumps({"type": "error", "msg": f"No replay data for mode '{mode}'. Run: python oceanus/demo_recorder.py"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/state")
def api_state():
    return sim.get_frame()

@app.get("/api/replay/{mode}")
def api_replay(mode: str):
    path = f"data/{mode}_episode.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"error": f"No replay data for '{mode}'"}

@app.get("/health")
def health():
    return {"status": "ok", "environment": "Oceanus-v2", "connected_clients": len(manager.active)}

# ── Serve the 3D frontend ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found. Run build.</h1>")
