"""
Oceanus 3D Mission Control — FastAPI + WebSocket backend
Serves the 3D HTML frontend and streams live environment state.

Run: uvicorn dashboard.server:app --host 0.0.0.0 --port 8000 --reload
"""
import sys, os, json, asyncio, re
from typing import Dict, List, Optional, Set
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from oceanus.models import OceanusEnv
from oceanus.adversary import AdversaryAgent
from oceanus.runner import OceanusRunner, MockASVAgent, MockPolicyAgent

app = FastAPI(title="Oceanus Mission Control", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

# Serve static JS files (Three.js, Chart.js, OrbitControls)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── Groq LLM Agent ────────────────────────────────────────────────────────────
class GroqLLMAgent:
    """Live LLM agent powered by Groq API (llama3-8b-8192)."""

    def __init__(self, agent_id: str, api_key: str):
        self.agent_id = agent_id
        self.api_key = api_key
        self._client = None
        self._step = 0
        self._last_action = None
        self._last_reasoning = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                if not self.api_key:
                    sim.warnings.append("LLM failed: No Groq API Key provided. Falling back to mock agents.")
                    return None
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                sim.warnings.append("LLM failed: 'groq' package not installed. Falling back to mock agents.")
                return None
            except Exception as e:
                sim.warnings.append(f"LLM failed: {e}. Falling back to mock agents.")
                return None
        return self._client

    def act(self, prompt: str, obs: Dict) -> str:
        self._step += 1
        client = self._get_client()
        if client is None:
            return self._fallback(obs)
        try:
            resp = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.3,
                timeout=3.0,
            )
            raw = resp.choices[0].message.content.strip()
            self._last_action = raw
            # Extract reasoning (text before JSON)
            import re
            match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if match:
                self._last_reasoning = raw[:match.start()].strip() or "Analyzing environment..."
            return raw
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                sim.warnings.append("LLM failed: Rate limit exceeded. Falling back to mock agents.")
            else:
                sim.warnings.append(f"LLM failed: API Error. Falling back to mock agents.")
            return self._fallback(obs)

    def _fallback(self, obs: Dict) -> str:
        """Heuristic fallback if Groq unavailable."""
        if self.agent_id.startswith("ASV"):
            sonar = obs.get("sonar_3x3", [])
            for dr in range(3):
                for dc in range(3):
                    if len(sonar) > dr and len(sonar[dr]) > dc:
                        if sonar[dr][dc] == "net":
                            if dr == 1 and dc == 1:
                                return '{"intent": "clean"}'
                            dirs = {(0,0):"stay",(0,1):"east",(0,-1):"west",(1,0):"south",(-1,0):"north"}
                            d = dirs.get((dr-1, dc-1), "north")
                            return json.dumps({"intent": "move", "direction": d})
            dirs = ["north","south","east","west"]
            return json.dumps({"intent": "move", "direction": dirs[self._step % 4]})
        else:
            status = obs.get("current_policy_status", "")
            if status == "Treaty Proposed" and self.agent_id == "Fleet_Manager":
                return '{"intent": "accept_treaty", "target": "Port_Authority", "content": "Agreed. Fleet will comply."}'
            if status == "No Tagging Mandate" and self.agent_id == "Port_Authority":
                return '{"intent": "propose_treaty", "target": "Fleet_Manager", "content": "50% subsidy on tracking tags."}'
            inbox = obs.get("inbox", [])
            if inbox:
                sender = inbox[0].get("from", "Unknown")
                return json.dumps({"intent": "reply_email", "target": sender, "content": "We are addressing your concerns immediately."})
            return '{"intent": "reply_email", "target": "General_Inbox", "content": "Monitoring situation."}'


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
        self.agent_mode: str = "mock"  # "mock" or "llm"
        self.groq_key: str = ""
        self.agent_thoughts: Dict[str, Dict] = {}  # last action + reasoning per agent
        self.warnings: List[str] = []
        self.cfg = {
            "seed": 42, "max_steps": 120, "chaos_interval": 25,
            "chaos_enabled": True, "step_delay": 0.08, "difficulty": "medium"
        }

    def reset(self, seed=42, max_steps=120, chaos_interval=25, agent_mode="mock", groq_key=""):
        self.env = OceanusEnv(seed=seed, max_steps=max_steps)
        self.adversary = AdversaryAgent(inject_interval=chaos_interval, seed=seed)
        self.agent_mode = agent_mode
        self.groq_key = groq_key

        if agent_mode == "llm":
            agents = {
                aid: GroqLLMAgent(aid, groq_key)
                for aid in ["ASV-1", "ASV-2", "ASV-3", "ASV-4", "Port_Authority", "Fleet_Manager"]
            }
            self.runner = OceanusRunner(self.env, self.adversary, agents=agents, verbose=False)
        else:
            self.runner = OceanusRunner(self.env, self.adversary, use_mock=True, verbose=False)

        self.obs_all = self.env.reset()
        self.running = False
        self.done = False
        self.total_reward = 0.0
        self.reward_history = []
        self.bio_history = []
        self.net_history = []
        self.event_log = []
        self.agent_thoughts = {}
        self.episode_count += 1

    def get_frame(self) -> Dict:
        if not self.env or not self.env.state:
            return {}
        s = self.env.state
        return {
            "type": "state",
            "step": s.step_count,
            "max_steps": s.max_steps,
            "grid": s.grid.tolist(),
            "asvs": {
                aid: {
                    "row": a.row, "col": a.col, "battery": a.battery,
                    "nets_cleaned": a.nets_cleaned,
                    "on_net": bool(s.grid[a.row, a.col] > 0),
                    "sector": s.get_sector(a.row, a.col),
                }
                for aid, a in s.asvs.items()
            },
            "ghost_nets": [{"row": n.row, "col": n.col, "density": n.density} for n in s.ghost_nets],
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
            "agent_mode": self.agent_mode,
            "agent_thoughts": self.agent_thoughts,
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
        if hasattr(sim, "warnings") and sim.warnings:
            for w in sim.warnings:
                if w not in [log.get("msg") for log in sim.event_log]:
                    sim.event_log.append({"type": "chaos", "msg": w})
                await manager.broadcast({"type": "notification", "msg": w, "level": "warning"})
            sim.warnings.clear()

        if sim.running and not sim.done and sim.obs_all and sim.runner:
            actions = {}
            for agent_id, agent in sim.runner.agents.items():
                if agent_id in sim.obs_all:
                    ao = sim.obs_all[agent_id]
                    raw = agent.act(ao["prompt"], ao["observation"])
                    actions[agent_id] = raw
                    # Capture thought bubble for LLM agents
                    if isinstance(agent, GroqLLMAgent):
                        import re as _re
                        match = _re.search(r'\{[^{}]*\}', raw, _re.DOTALL)
                        action_json = match.group() if match else raw
                        reasoning = raw[:match.start()].strip() if match else "Processing..."
                        sim.agent_thoughts[agent_id] = {
                            "action": action_json,
                            "reasoning": reasoning or "Analyzing sonar data...",
                            "is_llm": True,
                        }
                    else:
                        # Parse mock agent action for display
                        import re as _re
                        match = _re.search(r'"intent"\s*:\s*"([^"]+)"', raw)
                        intent = match.group(1) if match else "scan"
                        sim.agent_thoughts[agent_id] = {
                            "action": intent.upper(),
                            "reasoning": "",
                            "is_llm": False,
                        }

            obs_all, rewards, done, info = sim.env.step(actions)
            sim.obs_all = obs_all
            sim.done = done
            step_reward = rewards.get("__total__", 0.0)
            sim.total_reward += step_reward
            sim.reward_history.append(round(step_reward, 2))
            sim.bio_history.append(round(info["biodiversity"], 1))
            sim.net_history.append(info["active_nets"])

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
    await websocket.send_text(json.dumps(sim.get_frame()))
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            cmd = msg.get("cmd")

            if cmd == "start":
                difficulty = msg.get("difficulty", "medium")
                chaos_map = {"easy": 40, "medium": 25, "hard": 12}
                chaos_interval = chaos_map.get(difficulty, 25)
                agent_mode = msg.get("agent_mode", "mock")
                groq_key = msg.get("groq_key", "")

                # Fallback to HF Space Secret if available
                env_key = os.getenv("GROQ_API_KEY")
                if not groq_key and env_key:
                    groq_key = env_key
                    agent_mode = "llm"

                sim.cfg["chaos_enabled"] = True
                sim.cfg["step_delay"] = msg.get("step_delay", 0.08)
                sim.cfg["difficulty"] = difficulty
                sim.reset(
                    seed=msg.get("seed", 42),
                    max_steps=msg.get("max_steps", 120),
                    chaos_interval=chaos_interval,
                    agent_mode=agent_mode,
                    groq_key=groq_key,
                )
                sim.running = True
                await manager.broadcast({"type": "control", "status": "started", "agent_mode": agent_mode})

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
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                path = os.path.join(base_dir, "data", f"{mode}_episode.json")
                if os.path.exists(path):
                    with open(path) as f:
                        replay = json.load(f)
                    await websocket.send_text(json.dumps({"type": "replay_data", "data": replay}))
                else:
                    await websocket.send_text(json.dumps({"type": "error", "msg": f"No replay data for mode '{mode}'."}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/state")
def api_state():
    return sim.get_frame()

@app.get("/api/replay/{mode}")
def api_replay(mode: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", f"{mode}_episode.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"error": f"No replay data for '{mode}'"}

@app.get("/health")
def health():
    return {"status": "ok", "environment": "Oceanus-v3", "connected_clients": len(manager.active), "agent_mode": sim.agent_mode}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found.</h1>")
