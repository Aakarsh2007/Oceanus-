"""
Full episode runner — wires OceanusEnv + AdversaryAgent + agents.
"""
import json
import time
import asyncio
from typing import Dict, List, Callable, Optional
from oceanus.models import OceanusEnv
from oceanus.adversary import AdversaryAgent


class MockASVAgent:
    """Heuristic ASV agent for baseline comparison."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._directions = ["north", "south", "east", "west"]
        self._step = 0

    def act(self, prompt: str, obs: Dict) -> str:
        self._step += 1
        sonar = obs.get("sonar_3x3", [])

        # Scan sonar for nets and move toward / clean them
        net_direction = None
        for dr in range(3):
            for dc in range(3):
                if len(sonar) > dr and len(sonar[dr]) > dc and sonar[dr][dc] == "net":
                    row_diff = dr - 1
                    col_diff = dc - 1
                    if row_diff == 0 and col_diff == 0:
                        # Net is exactly at our position — clean it
                        return json.dumps({"intent": "clean"})
                    elif row_diff < 0:
                        net_direction = "north"
                    elif row_diff > 0:
                        net_direction = "south"
                    elif col_diff < 0:
                        net_direction = "west"
                    elif col_diff > 0:
                        net_direction = "east"

        if net_direction:
            return json.dumps({"intent": "move", "direction": net_direction})

        # Broadcast position every 10 steps
        if self._step % 10 == 0:
            pos = obs.get("position", {})
            return json.dumps({
                "intent": "broadcast",
                "message": f"At ({pos.get('row',0)},{pos.get('col',0)}) sector {pos.get('sector','?')}. Scanning.",
            })

        # Sweep pattern
        direction = self._directions[self._step % 4]
        return json.dumps({"intent": "move", "direction": direction})

    async def async_act(self, prompt: str, obs: Dict) -> str:
        await asyncio.sleep(0.01)  # Simulate network latency
        return self.act(prompt, obs)


class MockPolicyAgent:
    """Heuristic policy agent for baseline comparison."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._step = 0
        self._proposed = False
        self._accepted = False
        self._replied_to: set = set()

    def act(self, prompt: str, obs: Dict) -> str:
        self._step += 1
        inbox = obs.get("inbox", [])
        policy_status = obs.get("current_policy_status", "No Tagging Mandate")

        # Fleet_Manager: accept treaty if proposed
        if (
            policy_status == "Treaty Proposed"
            and self.agent_id == "Fleet_Manager"
            and not self._accepted
        ):
            self._accepted = True
            return json.dumps({
                "intent": "accept_treaty",
                "target": "Port_Authority",
                "content": "Agreed. Fleet will comply with tagging mandate under subsidy terms.",
            })

        # Reply to new emails
        for msg in inbox:
            sender = msg.get("from", "Unknown")
            if sender not in self._replied_to:
                self._replied_to.add(sender)
                mood = msg.get("mood", "Neutral")
                if mood in ("Angry", "Hostile"):
                    reply = "We hear your concerns and are working on a subsidy program to offset costs."
                elif mood in ("Urgent", "Critical"):
                    reply = "Immediate action is being taken. Emergency protocols activated."
                else:
                    reply = "Thank you for your message. We are reviewing the situation carefully."
                return json.dumps({"intent": "reply_email", "target": sender, "content": reply})

        # Port_Authority: propose treaty once
        if not self._proposed and self.agent_id == "Port_Authority":
            self._proposed = True
            return json.dumps({
                "intent": "propose_treaty",
                "target": "Fleet_Manager",
                "content": "50% subsidy on tracking tags in exchange for mandatory tagging compliance.",
            })

        # Default idle action — valid JSON so no penalty
        return json.dumps({"intent": "scan"}) if self.agent_id.startswith("ASV") else json.dumps({
            "intent": "reply_email",
            "target": "General_Inbox",
            "content": "Monitoring situation.",
        })

    async def async_act(self, prompt: str, obs: Dict) -> str:
        await asyncio.sleep(0.02)  # Policy agents have slightly higher simulated latency
        return self.act(prompt, obs)


class LLMAgent:
    """Wraps any callable LLM inference function."""

    def __init__(self, agent_id: str, llm_fn: Callable[[str], str]):
        self.agent_id = agent_id
        self.llm_fn = llm_fn

    def act(self, prompt: str, obs: Dict) -> str:
        try:
            return self.llm_fn(prompt)
        except Exception:
            if self.agent_id.startswith("ASV"):
                return json.dumps({"intent": "scan"})
            return json.dumps({
                "intent": "reply_email",
                "target": "General_Inbox",
                "content": "Processing...",
            })

    async def async_act(self, prompt: str, obs: Dict) -> str:
        # If llm_fn is actually an async function, we can await it directly.
        # For this hackathon, we simulate wrapping it:
        if asyncio.iscoroutinefunction(self.llm_fn):
            try:
                return await self.llm_fn(prompt)
            except Exception:
                return json.dumps({"intent": "scan"})
        else:
            return await asyncio.to_thread(self.act, prompt, obs)


class OceanusRunner:
    """Runs full episodes of Oceanus with any combination of agents."""

    def __init__(
        self,
        env: OceanusEnv,
        adversary: AdversaryAgent,
        agents: Optional[Dict] = None,
        use_mock: bool = True,
        llm_fn: Optional[Callable] = None,
        verbose: bool = True,
        step_delay: float = 0.0,
    ):
        self.env = env
        self.adversary = adversary
        self.verbose = verbose
        self.step_delay = step_delay

        if agents:
            self.agents = agents
        elif use_mock:
            self.agents = {
                "ASV-1": MockASVAgent("ASV-1"),
                "ASV-2": MockASVAgent("ASV-2"),
                "ASV-3": MockASVAgent("ASV-3"),
                "ASV-4": MockASVAgent("ASV-4"),
                "Port_Authority": MockPolicyAgent("Port_Authority"),
                "Fleet_Manager": MockPolicyAgent("Fleet_Manager"),
            }
        else:
            assert llm_fn is not None, "Provide llm_fn for LLM agents"
            self.agents = {
                aid: LLMAgent(aid, llm_fn)
                for aid in ["ASV-1", "ASV-2", "ASV-3", "ASV-4", "Port_Authority", "Fleet_Manager"]
            }

        self.episode_logs: List[Dict] = []

    def run_episode(self, episode_id: int = 0) -> Dict:
        """Run a full episode. Returns episode summary."""
        obs_all = self.env.reset()
        done = False
        total_reward = 0.0
        chaos_events_all: List[str] = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  OCEANUS EPISODE {episode_id} STARTING")
            print(f"  Agents: {list(self.agents.keys())}")
            print(f"{'='*60}")

        while not done:
            actions = {}
            for agent_id, agent in self.agents.items():
                if agent_id in obs_all:
                    agent_obs = obs_all[agent_id]
                    actions[agent_id] = agent.act(
                        agent_obs["prompt"], agent_obs["observation"]
                    )

            obs_all, rewards, done, info = self.env.step(actions)
            step_reward = rewards.get("__total__", 0.0)
            total_reward += step_reward

            if self.adversary.should_inject(info["step"]):
                chaos = self.adversary.inject(self.env.state)
                chaos_events_all.extend(chaos)
                if self.verbose:
                    for event in chaos:
                        print(f"  CHAOS [{info['step']}]: {event}")

            if self.verbose and info["step"] % 20 == 0:
                print(
                    f"  Step {info['step']:3d} | "
                    f"Nets: {info['active_nets']:3d} | "
                    f"Cleaned: {info['total_cleaned']:3d} | "
                    f"Bio: {info['biodiversity']:5.1f}% | "
                    f"Treaty: {info['treaty_status'][:25]} | "
                    f"R: {step_reward:+.1f}"
                )

            for c in info.get("cleaned", []):
                if self.verbose:
                    print(f"  CLEANED: {c}")

            for te in info.get("treaty_events", []):
                if self.verbose:
                    print(f"  TREATY: {te}")

            if self.step_delay > 0:
                time.sleep(self.step_delay)

        summary = self.env.get_episode_summary()
        summary["episode_id"] = episode_id
        summary["total_reward"] = total_reward
        summary["chaos_events"] = chaos_events_all
        summary["adversary_summary"] = self.adversary.get_curriculum_summary()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  EPISODE {episode_id} COMPLETE")
            print(f"  Total Reward:    {total_reward:.2f}")
            print(f"  Biodiversity:    {summary['biodiversity_final']:.1f}%")
            print(f"  Nets Cleaned:    {summary['total_cleaned']}")
            print(f"  Treaty Status:   {summary['treaty_status']}")
            print(f"  Chaos Events:    {len(chaos_events_all)}")
            print(f"{'='*60}\n")

        self.episode_logs.append(summary)
        return summary

    def run_baseline(self, n_episodes: int = 3) -> List[Dict]:
        return [self.run_episode(i) for i in range(n_episodes)]

    async def async_run_episode(self, episode_id: int = 0, on_step_callback: Optional[Callable] = None) -> Dict:
        """Run a full episode asynchronously, simulating real multi-agent latency."""
        obs_all = self.env.reset()
        done = False
        total_reward = 0.0
        chaos_events_all: List[str] = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  OCEANUS ASYNC EPISODE {episode_id} STARTING")
            print(f"{'='*60}")

        while not done:
            # Prepare async tasks
            tasks = []
            agent_ids = []
            for agent_id, agent in self.agents.items():
                if agent_id in obs_all:
                    agent_obs = obs_all[agent_id]
                    if hasattr(agent, "async_act"):
                        tasks.append(agent.async_act(agent_obs["prompt"], agent_obs["observation"]))
                    else:
                        tasks.append(asyncio.to_thread(agent.act, agent_obs["prompt"], agent_obs["observation"]))
                    agent_ids.append(agent_id)

            # Execute all agent inferences concurrently
            results = await asyncio.gather(*tasks)
            actions = dict(zip(agent_ids, results))

            obs_all, rewards, done, info = self.env.step(actions)
            step_reward = rewards.get("__total__", 0.0)
            total_reward += step_reward

            if self.adversary.should_inject(info["step"]):
                chaos = self.adversary.inject(self.env.state)
                chaos_events_all.extend(chaos)
                if self.verbose:
                    for event in chaos:
                        print(f"  CHAOS [{info['step']}]: {event}")

            if on_step_callback:
                if asyncio.iscoroutinefunction(on_step_callback):
                    await on_step_callback(self.env.state, info)
                else:
                    on_step_callback(self.env.state, info)

            if self.step_delay > 0:
                await asyncio.sleep(self.step_delay)

        summary = self.env.get_episode_summary()
        summary["episode_id"] = episode_id
        summary["total_reward"] = total_reward
        summary["chaos_events"] = chaos_events_all
        summary["adversary_summary"] = self.adversary.get_curriculum_summary()
        
        self.episode_logs.append(summary)
        return summary
