"""
Oceanus OpenEnv-compliant Environment Server
Inherits from openenv-core Environment base class.
"""
from typing import Optional
from openenv.core.env_server.interfaces import Environment

from models import OceanusAction, OceanusObservation, OceanusState
from oceanus.models import OceanusEnv as _OceanusEnv
from oceanus.adversary import AdversaryAgent
from oceanus.physics import GRID_SIZE


class OceanusEnvironment(Environment[OceanusAction, OceanusObservation, OceanusState]):
    """
    OpenEnv-compliant wrapper around the Oceanus multi-agent environment.

    Agents: 4 ASV drones + Port_Authority + Fleet_Manager
    Adversary: Chaos agent with 8 event types, difficulty scaling 1.0 → 3.0
    """

    def __init__(self, seed: int = 42, max_steps: int = 120, chaos_interval: int = 25):
        self._seed = seed
        self._max_steps = max_steps
        self._chaos_interval = chaos_interval
        self._env: Optional[_OceanusEnv] = None
        self._adversary: Optional[AdversaryAgent] = None
        self._obs_all = None
        self._episode_id = 0
        self._total_reward = 0.0
        self._chaos_count = 0
        self._event_log = []

    def reset(self) -> OceanusObservation:
        """Reset environment and return initial observation."""
        self._env = _OceanusEnv(seed=self._seed, max_steps=self._max_steps)
        self._adversary = AdversaryAgent(inject_interval=self._chaos_interval, seed=self._seed)
        self._obs_all = self._env.reset()
        self._episode_id += 1
        self._total_reward = 0.0
        self._chaos_count = 0
        self._event_log = []
        return self._build_observation(done=False)

    def step(self, action: OceanusAction) -> OceanusObservation:
        """
        Execute one action for one agent.
        In multi-agent mode, collects action for the specified agent
        and steps the environment when all agents have acted.
        For simplicity in the OpenEnv interface, we accept a single
        agent action and step immediately with mock actions for others.
        """
        assert self._env is not None, "Call reset() first"
        assert self._obs_all is not None, "Call reset() first"

        # Build action dict — use provided action for specified agent,
        # mock scan for all others
        import json
        actions = {}
        for agent_id in self._obs_all:
            if agent_id == action.agent_id:
                # Serialize the typed action back to JSON string
                action_dict = {"intent": action.intent}
                if action.direction:
                    action_dict["direction"] = action.direction
                if action.message:
                    action_dict["message"] = action.message
                if action.target:
                    action_dict["target"] = action.target
                if action.content:
                    action_dict["content"] = action.content
                actions[agent_id] = json.dumps(action_dict)
            else:
                # Default scan for other agents
                actions[agent_id] = '{"intent": "scan"}'

        obs_all, rewards, done, info = self._env.step(actions)
        self._obs_all = obs_all
        step_reward = rewards.get("__total__", 0.0)
        self._total_reward += step_reward

        # Inject chaos if due
        if self._adversary.should_inject(info["step"]):
            chaos_events = self._adversary.inject(self._env.state)
            self._chaos_count += len(chaos_events)
            for ev in chaos_events:
                self._event_log.append({"type": "chaos", "msg": f"CHAOS: {ev}"})

        # Log step events
        for c in info.get("cleaned", []):
            self._event_log.append({"type": "clean", "msg": f"CLEANED: {c}"})
        for te in info.get("treaty_events", []):
            self._event_log.append({"type": "treaty", "msg": f"TREATY: {te}"})
        for b in info.get("broadcasts", []):
            self._event_log.append({"type": "broadcast", "msg": f"BROADCAST: {b}"})

        return self._build_observation(done=done, reward=step_reward)

    def state(self) -> OceanusState:
        """Return current episode state metadata."""
        s = self._env.state if self._env else None
        return OceanusState(
            episode_id=self._episode_id,
            step_count=s.step_count if s else 0,
            episode_id_field=self._episode_id,
            seed=self._seed,
            max_steps=self._max_steps,
            chaos_events_fired=self._chaos_count,
            difficulty=round(self._adversary.difficulty, 2) if self._adversary else 1.0,
            schema_version=s.schema_version if s else "v1",
        )

    def _build_observation(self, done: bool, reward: float = 0.0) -> OceanusObservation:
        """Serialize current env state into OceanusObservation."""
        s = self._env.state
        agent_obs = {}
        if self._obs_all:
            for aid, ao in self._obs_all.items():
                agent_obs[aid] = {
                    "observation": ao["observation"],
                    "prompt": ao["prompt"],
                }
        return OceanusObservation(
            done=done,
            reward=reward,
            step=s.step_count,
            biodiversity=round(s.biodiversity_index, 2),
            active_nets=len(s.ghost_nets),
            total_cleaned=s.total_cleaned,
            treaty_status=s.treaty_status,
            net_spawn_rate=s.net_spawn_rate,
            grid=s.grid.tolist(),
            asvs={
                aid: {
                    "row": a.row, "col": a.col,
                    "battery": a.battery,
                    "nets_cleaned": a.nets_cleaned,
                    "on_net": bool(s.grid[a.row, a.col] > 0),
                    "sector": s.get_sector(a.row, a.col),
                }
                for aid, a in s.asvs.items()
            },
            ghost_nets=[
                {"row": n.row, "col": n.col, "density": n.density}
                for n in s.ghost_nets
            ],
            wind_vector=list(s.wind_vector),
            total_reward=round(self._total_reward, 2),
            event_log=self._event_log[-20:],
            agent_observations=agent_obs,
        )
