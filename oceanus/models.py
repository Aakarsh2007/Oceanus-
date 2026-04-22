"""
OpenEnv Wrapper — reset(), step(), ObservationSpace, ActionSpace.
"""
import json
import re
from typing import Dict, Any, Tuple, List, Optional
from oceanus.physics import GameState, GRID_SIZE


def parse_asv_action(raw_output: str, schema_version: str = "v1") -> Tuple[Optional[Dict], bool]:
    """Extract JSON action from LLM output. Returns (action_dict, is_valid)."""
    if not raw_output or not isinstance(raw_output, str):
        return None, False
    try:
        # Greedy match to handle nested braces
        match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
        if not match:
            return None, False
        action = json.loads(match.group())
        
        intent_key = "action_type" if schema_version == "v2" else "intent"
        intent = action.get(intent_key, "")
        
        if not intent:
            return None, False
        valid_intents = {"move", "scan", "clean", "broadcast"}
        if intent not in valid_intents:
            return None, False
        if intent == "move":
            direction = action.get("direction", "")
            if direction not in {"north", "south", "east", "west", "stay"}:
                return None, False
        if intent == "broadcast" and not action.get("message", "").strip():
            return None, False
            
        # Normalize to v1 for internal engine use
        if schema_version == "v2":
            action["intent"] = action.pop("action_type")
            
        return action, True
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None, False


def parse_policy_action(raw_output: str, schema_version: str = "v1") -> Tuple[Optional[Dict], bool]:
    """Extract JSON action from policy LLM output."""
    if not raw_output or not isinstance(raw_output, str):
        return None, False
    try:
        match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
        if not match:
            return None, False
        action = json.loads(match.group())
        
        intent_key = "action_type" if schema_version == "v2" else "intent"
        target_key = "recipient" if schema_version == "v2" else "target"
        
        intent = action.get(intent_key, "")
        if not intent:
            return None, False
        valid_intents = {"reply_email", "propose_treaty", "accept_treaty", "reject_treaty"}
        if intent not in valid_intents:
            return None, False
        # Relaxed validation — only require target for reply/propose/accept
        if intent in ("reply_email", "propose_treaty", "accept_treaty"):
            if not action.get(target_key, "").strip():
                return None, False
                
        # Normalize to v1 for internal engine use
        if schema_version == "v2":
            action["intent"] = action.pop("action_type")
            if target_key in action:
                action["target"] = action.pop("recipient")
                
        return action, True
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None, False


# ─────────────────────────────────────────────
# Observation Builders
# ─────────────────────────────────────────────

def build_asv_observation(state: GameState, asv_id: str) -> Dict:
    asv = state.asvs[asv_id]
    sonar = state.get_sonar_reading(asv)
    sector = state.get_sector(asv.row, asv.col)
    on_net = state.grid[asv.row, asv.col] > 0
    return {
        "agent_id": asv_id,
        "position": {"row": asv.row, "col": asv.col, "sector": sector},
        "battery": asv.battery,
        "sonar_3x3": sonar,
        "on_net": bool(on_net),
        "comms_inbox": list(asv.comms_inbox[-3:]),
        "step": state.step_count,
        "biodiversity_index": round(state.biodiversity_index, 1),
        "wind_hint": f"Wind direction: {state.wind_vector}",
        "schema_version": state.schema_version,
    }


def build_policy_observation(state: GameState, agent_id: str) -> Dict:
    if agent_id == "Port_Authority":
        inbox_msgs = list(state.port_authority_inbox.messages)[:4]
    else:
        inbox_msgs = list(state.fleet_manager_inbox.messages)[:4]

    return {
        "agent_id": agent_id,
        "current_policy_status": state.treaty_status,
        "net_spawn_rate": state.net_spawn_rate,
        "biodiversity_index": round(state.biodiversity_index, 1),
        "inbox": inbox_msgs,
        "step": state.step_count,
        "unanswered_emails": (
            state.port_authority_inbox.unanswered_count
            if agent_id == "Port_Authority"
            else state.fleet_manager_inbox.unanswered_count
        ),
        "schema_version": state.schema_version,
    }


def build_asv_prompt(obs: Dict) -> str:
    sonar_str = "\n".join(["  " + str(row) for row in obs["sonar_3x3"]])
    inbox_str = "\n".join([f"  - {m}" for m in obs["comms_inbox"]]) or "  (empty)"
    if obs.get("schema_version") == "v2":
        schema_instructions = """
Respond with a single JSON action using schema v2 ('action_type' instead of 'intent'):
- To move: {"action_type": "move", "direction": "north|south|east|west|stay"}
- To clean a net at your position: {"action_type": "clean"}
- To broadcast to other ASVs: {"action_type": "broadcast", "message": "your message here"}
- To scan (reveal sonar): {"action_type": "scan"}
"""
    else:
        schema_instructions = """
Respond with a single JSON action using schema v1:
- To move: {"intent": "move", "direction": "north|south|east|west|stay"}
- To clean a net at your position: {"intent": "clean"}
- To broadcast to other ASVs: {"intent": "broadcast", "message": "your message here"}
- To scan (reveal sonar): {"intent": "scan"}
"""

    return f"""You are {obs['agent_id']}, an Autonomous Surface Vehicle on a ghost-net recovery mission.

CURRENT STATE:
- Position: Row {obs['position']['row']}, Col {obs['position']['col']} (Sector {obs['position']['sector']})
- Battery: {obs['battery']}%
- Biodiversity Index: {obs['biodiversity_index']}%
- Step: {obs['step']}
- {obs['wind_hint']}

SONAR (3x3 grid around you, 'net' = ghost net detected):
{sonar_str}

COMMS INBOX:
{inbox_str}

TASK: Recover ghost nets to restore ocean biodiversity. Coordinate with other ASVs.
{schema_instructions}
Your JSON action:"""


def build_policy_prompt(obs: Dict) -> str:
    inbox_str = ""
    for msg in obs["inbox"]:
        inbox_str += f"\n  FROM: {msg.get('from','?')} [Mood: {msg.get('mood','Neutral')}]\n  MSG: {msg.get('message','')}\n"
    if not inbox_str:
        inbox_str = "  (no messages)"

    if obs.get("schema_version") == "v2":
        schema_instructions = """
Respond with a single JSON action using schema v2 ('action_type' instead of 'intent', 'recipient' instead of 'target'):
- To reply to an email: {"action_type": "reply_email", "recipient": "sender_name", "content": "your reply"}
- To propose a treaty: {"action_type": "propose_treaty", "recipient": "Fleet_Manager", "content": "treaty terms"}
- To accept a treaty: {"action_type": "accept_treaty", "recipient": "Port_Authority", "content": "acceptance message"}
"""
    else:
        schema_instructions = """
Respond with a single JSON action using schema v1:
- To reply to an email: {"intent": "reply_email", "target": "sender_name", "content": "your reply"}
- To propose a treaty: {"intent": "propose_treaty", "target": "Fleet_Manager", "content": "treaty terms"}
- To accept a treaty: {"intent": "accept_treaty", "target": "Port_Authority", "content": "acceptance message"}
"""

    return f"""You are the {obs['agent_id']} in the Oceanus Ocean Recovery Program.

CURRENT STATE:
- Policy Status: {obs['current_policy_status']}
- Net Spawn Rate: {obs['net_spawn_rate']} (lower is better)
- Biodiversity Index: {obs['biodiversity_index']}%
- Unanswered Emails: {obs['unanswered_emails']}
- Step: {obs['step']}

YOUR INBOX:
{inbox_str}

TASK: Negotiate a Tagging Mandate treaty to reduce ghost net spawning. Reply to stakeholders diplomatically.
A successful treaty reduces net spawn rate from High to Low, saving the ecosystem.
{schema_instructions}
Your JSON action:"""


# ─────────────────────────────────────────────
# Reward Engine
# ─────────────────────────────────────────────

class RewardEngine:
    """CTDE reward: R_total = α*Σ R_local_asv + β*Σ R_local_policy + γ*R_global_ecosystem"""
    ALPHA = 1.0
    BETA = 1.0
    GAMMA = 2.0

    def __init__(self):
        self.episode_rewards: Dict[str, List[float]] = {}

    def compute_asv_reward(
        self, action: Optional[Dict], valid: bool, cleaned: bool, broadcast_useful: bool
    ) -> float:
        r = 0.0
        if not valid:
            r -= 1.0
        if cleaned:
            r += 5.0
        if action and action.get("intent") == "move":
            r -= 0.1
        if broadcast_useful:
            r += 0.5
        return r

    def compute_policy_reward(
        self,
        action: Optional[Dict],
        valid: bool,
        treaty_advanced: bool,
        email_replied: bool,
        email_ignored_penalty: bool,
    ) -> float:
        r = 0.0
        if not valid:
            r -= 1.0
        if treaty_advanced:
            r += 10.0
        if email_replied:
            r += 2.0
        if email_ignored_penalty:
            r -= 5.0
        return r

    def compute_global_reward(self, state: GameState) -> float:
        if not state.done:
            return 0.0
        if state.biodiversity_index >= 75.0:
            return 100.0
        elif state.biodiversity_index >= 50.0:
            return 40.0
        elif state.biodiversity_index >= 25.0:
            return 10.0
        return -20.0

    def blend(self, asv_rewards: List[float], policy_rewards: List[float], global_r: float) -> float:
        return self.ALPHA * sum(asv_rewards) + self.BETA * sum(policy_rewards) + self.GAMMA * global_r


# ─────────────────────────────────────────────
# Main OpenEnv Environment
# ─────────────────────────────────────────────

class OceanusEnv:
    """
    OpenEnv-compatible environment for Oceanus.
    Observation space: JSON strings (text)
    Action space: Text output (parsed to JSON)
    """

    metadata = {
        "name": "Oceanus-v1",
        "description": "Multi-layer ghost-gear recovery and treaty negotiation arena",
        "themes": ["multi-agent", "long-horizon", "world-modeling", "self-improvement"],
        "num_agents": 6,
    }

    def __init__(self, seed: int = 42, max_steps: int = 200):
        self.seed = seed
        self.max_steps = max_steps
        self.state: Optional[GameState] = None
        self.reward_engine = RewardEngine()
        self._episode_asv_rewards: List[float] = []
        self._episode_policy_rewards: List[float] = []
        self._step_log: List[Dict] = []

    def reset(self) -> Dict[str, Any]:
        """Reset environment, return initial observations for all agents."""
        self.state = GameState(seed=self.seed)
        self.state.max_steps = self.max_steps
        self._episode_asv_rewards = []
        self._episode_policy_rewards = []
        self._step_log = []

        obs = {}
        for asv_id in self.state.asvs:
            raw_obs = build_asv_observation(self.state, asv_id)
            obs[asv_id] = {"observation": raw_obs, "prompt": build_asv_prompt(raw_obs)}

        for agent_id in ["Port_Authority", "Fleet_Manager"]:
            raw_obs = build_policy_observation(self.state, agent_id)
            obs[agent_id] = {"observation": raw_obs, "prompt": build_policy_prompt(raw_obs)}

        return obs

    def step(self, actions: Dict[str, str]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Process one step for all agents.
        actions: {agent_id: raw_llm_output_string}
        Returns: (observations, rewards, done, info)
        """
        assert self.state is not None, "Call reset() first"

        rewards: Dict[str, float] = {}
        info: Dict[str, Any] = {"events": [], "cleaned": [], "treaty_events": [], "broadcasts": []}

        # ── ASV actions ──
        asv_rewards = []
        for asv_id, raw_output in actions.items():
            if not asv_id.startswith("ASV"):
                continue
            asv = self.state.asvs.get(asv_id)
            if asv is None:
                continue

            action, valid = parse_asv_action(raw_output, schema_version=self.state.schema_version)
            cleaned = False
            broadcast_useful = False

            if valid and action:
                intent = action["intent"]

                if intent == "move":
                    asv.move(action.get("direction", "stay"))

                elif intent == "clean":
                    cleaned = self.state.clean_net_at(asv.row, asv.col)
                    if cleaned:
                        asv.nets_cleaned += 1
                        info["cleaned"].append(f"{asv_id} cleaned net at ({asv.row},{asv.col})")

                elif intent == "broadcast":
                    msg = action.get("message", "").strip()
                    if msg:
                        broadcast_useful = True
                        for other_id, other_asv in self.state.asvs.items():
                            if other_id != asv_id:
                                other_asv.comms_inbox.append(f"{asv_id}: {msg}")
                                if len(other_asv.comms_inbox) > 10:
                                    other_asv.comms_inbox.pop(0)
                        info["broadcasts"].append(f"{asv_id}: {msg}")

                # "scan" — sonar is always active, no extra action needed

            r = self.reward_engine.compute_asv_reward(action, valid, cleaned, broadcast_useful)
            rewards[asv_id] = r
            asv_rewards.append(r)

        # ── Policy actions ──
        policy_rewards = []
        for agent_id, raw_output in actions.items():
            if agent_id.startswith("ASV"):
                continue
            if agent_id not in ("Port_Authority", "Fleet_Manager"):
                continue

            action, valid = parse_policy_action(raw_output, schema_version=self.state.schema_version)
            treaty_advanced = False
            email_replied = False
            email_ignored_penalty = False

            if valid and action:
                intent = action["intent"]
                target = action.get("target", "")
                content = action.get("content", "")

                if intent == "reply_email":
                    inbox = (
                        self.state.port_authority_inbox
                        if agent_id == "Port_Authority"
                        else self.state.fleet_manager_inbox
                    )
                    msg = inbox.pop_message(target)
                    if msg:
                        email_replied = True
                        info["treaty_events"].append(f"{agent_id} replied to {target}")

                elif intent == "propose_treaty":
                    if self.state.treaty_status == "No Tagging Mandate":
                        self.state.treaty_status = "Treaty Proposed"
                        treaty_advanced = True
                        other_inbox = (
                            self.state.fleet_manager_inbox
                            if agent_id == "Port_Authority"
                            else self.state.port_authority_inbox
                        )
                        other_inbox.add_message(agent_id, "Formal", f"Treaty proposal: {content}")
                        info["treaty_events"].append(
                            f"{agent_id} proposed treaty: {content[:60]}..."
                        )

                elif intent == "accept_treaty":
                    if self.state.treaty_status == "Treaty Proposed":
                        self.state.treaty_status = "Tagging Mandate Active"
                        treaty_advanced = True
                        info["treaty_events"].append(
                            f"TREATY SIGNED by {agent_id}! Spawn rate dropping."
                        )

                elif intent == "reject_treaty":
                    self.state.treaty_status = "No Tagging Mandate"
                    info["treaty_events"].append(f"{agent_id} rejected treaty.")

            # Penalty for ignored emails > 5 steps old
            inbox = (
                self.state.port_authority_inbox
                if agent_id == "Port_Authority"
                else self.state.fleet_manager_inbox
            )
            if any(m["age"] > 5 for m in inbox.messages):
                email_ignored_penalty = True

            r = self.reward_engine.compute_policy_reward(
                action, valid, treaty_advanced, email_replied, email_ignored_penalty
            )
            rewards[agent_id] = r
            policy_rewards.append(r)

        # ── Tick physics ──
        self.state.tick()

        # ── Global reward at episode end ──
        global_r = self.reward_engine.compute_global_reward(self.state)
        total_r = self.reward_engine.blend(asv_rewards, policy_rewards, global_r)
        rewards["__total__"] = total_r
        rewards["__global__"] = global_r

        self._episode_asv_rewards.extend(asv_rewards)
        self._episode_policy_rewards.extend(policy_rewards)

        # ── Build next observations ──
        obs: Dict[str, Any] = {}
        for asv_id in self.state.asvs:
            raw_obs = build_asv_observation(self.state, asv_id)
            obs[asv_id] = {"observation": raw_obs, "prompt": build_asv_prompt(raw_obs)}
        for agent_id in ["Port_Authority", "Fleet_Manager"]:
            raw_obs = build_policy_observation(self.state, agent_id)
            obs[agent_id] = {"observation": raw_obs, "prompt": build_policy_prompt(raw_obs)}

        info["step"] = self.state.step_count
        info["biodiversity"] = self.state.biodiversity_index
        info["active_nets"] = len(self.state.ghost_nets)
        info["treaty_status"] = self.state.treaty_status
        info["total_cleaned"] = self.state.total_cleaned

        self._step_log.append({
            "step": self.state.step_count,
            "rewards": dict(rewards),
            "biodiversity": self.state.biodiversity_index,
            "active_nets": len(self.state.ghost_nets),
            "treaty": self.state.treaty_status,
        })

        return obs, rewards, self.state.done, info

    def get_episode_summary(self) -> Dict:
        return {
            "total_steps": self.state.step_count if self.state else 0,
            "total_cleaned": self.state.total_cleaned if self.state else 0,
            "biodiversity_final": self.state.biodiversity_index if self.state else 0.0,
            "treaty_status": self.state.treaty_status if self.state else "N/A",
            "avg_asv_reward": (
                sum(self._episode_asv_rewards) / max(1, len(self._episode_asv_rewards))
            ),
            "avg_policy_reward": (
                sum(self._episode_policy_rewards) / max(1, len(self._episode_policy_rewards))
            ),
            "step_log": self._step_log,
        }
