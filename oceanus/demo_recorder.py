"""
Demo Recorder — pre-records episodes to JSON for instant replay.
Run once: python oceanus/demo_recorder.py
Produces: data/baseline_episode.json and data/trained_episode.json
"""
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oceanus.models import OceanusEnv
from oceanus.adversary import AdversaryAgent
from oceanus.runner import MockASVAgent, MockPolicyAgent
from oceanus.physics import GRID_SIZE


def record_episode(mode: str = "trained", seed: int = 42, max_steps: int = 120) -> dict:
    """
    Record a full episode frame-by-frame.
    mode='baseline' uses dumb random agents.
    mode='trained'  uses smart heuristic agents (simulates trained behaviour).
    """
    env = OceanusEnv(seed=seed, max_steps=max_steps)
    adversary = AdversaryAgent(inject_interval=25, seed=seed)
    obs_all = env.reset()

    if mode == "baseline":
        agents = _make_baseline_agents()
    else:
        agents = _make_trained_agents()

    frames = []
    total_reward = 0.0
    done = False

    while not done:
        # Snapshot current state BEFORE step
        state = env.state
        frame = {
            "step": state.step_count,
            "grid": state.grid.tolist(),
            "asvs": {
                aid: {"row": a.row, "col": a.col, "battery": a.battery,
                      "nets_cleaned": a.nets_cleaned}
                for aid, a in state.asvs.items()
            },
            "ghost_nets": [{"row": n.row, "col": n.col, "density": n.density}
                           for n in state.ghost_nets],
            "biodiversity": round(state.biodiversity_index, 2),
            "active_nets": len(state.ghost_nets),
            "total_cleaned": state.total_cleaned,
            "treaty_status": state.treaty_status,
            "net_spawn_rate": state.net_spawn_rate,
            "wind_vector": list(state.wind_vector),
            "events": [],
            "reward": 0.0,
        }

        # Collect actions
        actions = {}
        for agent_id, agent in agents.items():
            if agent_id in obs_all:
                ao = obs_all[agent_id]
                actions[agent_id] = agent.act(ao["prompt"], ao["observation"])

        # Step
        obs_all, rewards, done, info = env.step(actions)
        step_reward = rewards.get("__total__", 0.0)
        total_reward += step_reward

        # Chaos
        chaos_events = []
        if adversary.should_inject(info["step"]):
            chaos_events = adversary.inject(env.state)

        # Fill frame events
        frame["reward"] = round(step_reward, 2)
        frame["cumulative_reward"] = round(total_reward, 2)
        frame["events"] = (
            [f"CLEANED: {c}" for c in info.get("cleaned", [])] +
            [f"TREATY: {te}" for te in info.get("treaty_events", [])] +
            [f"BROADCAST: {b}" for b in info.get("broadcasts", [])] +
            [f"CHAOS: {c}" for c in chaos_events]
        )
        frames.append(frame)

    summary = env.get_episode_summary()
    return {
        "mode": mode,
        "seed": seed,
        "total_steps": len(frames),
        "total_reward": round(total_reward, 2),
        "biodiversity_final": round(summary["biodiversity_final"], 2),
        "total_cleaned": summary["total_cleaned"],
        "treaty_status": summary["treaty_status"],
        "frames": frames,
    }


# ── Baseline: dumb agents that barely function ────────────────────────────────

class _DumbASV:
    """Baseline: just moves randomly, never cleans proactively."""
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self._step = 0
        self._dirs = ["north", "south", "east", "west", "stay", "stay"]

    def act(self, prompt, obs):
        import json
        self._step += 1
        # Only clean 20% of the time even when on a net
        sonar = obs.get("sonar_3x3", [])
        if sonar[1][1] == "net" and random.random() < 0.2:
            return json.dumps({"intent": "clean"})
        return json.dumps({"intent": "move", "direction": random.choice(self._dirs)})


class _DumbPolicy:
    """Baseline: ignores emails, never proposes treaty."""
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def act(self, prompt, obs):
        import json
        # Just scans — never negotiates
        return json.dumps({"intent": "reply_email", "target": "nobody", "content": "..."})


def _make_baseline_agents():
    return {
        "ASV-1": _DumbASV("ASV-1"),
        "ASV-2": _DumbASV("ASV-2"),
        "ASV-3": _DumbASV("ASV-3"),
        "ASV-4": _DumbASV("ASV-4"),
        "Port_Authority": _DumbPolicy("Port_Authority"),
        "Fleet_Manager": _DumbPolicy("Fleet_Manager"),
    }


class _SmartASV:
    """Trained: actively hunts nets using sonar + broadcasts coordinates."""
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self._step = 0
        self._idx = int(agent_id.split("-")[1]) - 1

    def act(self, prompt, obs):
        import json
        self._step += 1
        sonar = obs.get("sonar_3x3", [])
        pos = obs.get("position", {})
        row = pos.get("row", 0)
        col = pos.get("col", 0)
        # on_net is passed in observation if available
        on_net = obs.get("on_net", False)

        # If we're on a net, clean it
        if on_net:
            return json.dumps({"intent": "clean"})

        # Move toward nearest net in sonar
        for dr in range(3):
            for dc in range(3):
                cell = sonar[dr][dc] if len(sonar) > dr and len(sonar[dr]) > dc else "empty"
                if cell == "net":
                    rd, cd = dr - 1, dc - 1
                    if abs(rd) >= abs(cd):
                        d = "north" if rd < 0 else "south"
                    else:
                        d = "west" if cd < 0 else "east"
                    return json.dumps({"intent": "move", "direction": d})

        # Broadcast every 8 steps
        if self._step % 8 == 0:
            return json.dumps({
                "intent": "broadcast",
                "message": f"({row},{col}) Sector {pos.get('sector','?')} clear. Sweeping."
            })

        # Boustrophedon sweep per quadrant
        quadrant_targets = [
            [(r, c) for r in range(0, 10) for c in (range(0,10) if r%2==0 else range(9,-1,-1))],
            [(r, c) for r in range(0, 10) for c in (range(10,20) if r%2==0 else range(19,9,-1))],
            [(r, c) for r in range(10, 20) for c in (range(0,10) if r%2==0 else range(9,-1,-1))],
            [(r, c) for r in range(10, 20) for c in (range(10,20) if r%2==0 else range(19,9,-1))],
        ]
        targets = quadrant_targets[self._idx % 4]
        tr, tc = targets[self._step % len(targets)]
        if row < tr: return json.dumps({"intent": "move", "direction": "south"})
        if row > tr: return json.dumps({"intent": "move", "direction": "north"})
        if col < tc: return json.dumps({"intent": "move", "direction": "east"})
        if col > tc: return json.dumps({"intent": "move", "direction": "west"})
        return json.dumps({"intent": "scan"})


def _make_trained_agents():
    return {
        "ASV-1": _SmartASV("ASV-1"),
        "ASV-2": _SmartASV("ASV-2"),
        "ASV-3": _SmartASV("ASV-3"),
        "ASV-4": _SmartASV("ASV-4"),
        "Port_Authority": MockPolicyAgent("Port_Authority"),
        "Fleet_Manager": MockPolicyAgent("Fleet_Manager"),
    }


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Recording BASELINE episode...")
    baseline = record_episode(mode="baseline", seed=42, max_steps=120)
    with open("data/baseline_episode.json", "w") as f:
        json.dump(baseline, f)
    print(f"  Baseline: reward={baseline['total_reward']}, bio={baseline['biodiversity_final']}%, cleaned={baseline['total_cleaned']}")

    print("Recording TRAINED episode...")
    trained = record_episode(mode="trained", seed=42, max_steps=120)
    with open("data/trained_episode.json", "w") as f:
        json.dump(trained, f)
    print(f"  Trained:  reward={trained['total_reward']}, bio={trained['biodiversity_final']}%, cleaned={trained['total_cleaned']}")

    print("\nDone. Files saved to data/")
