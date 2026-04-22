"""
Adversary / Chaos Agent
Injects schema drift, storms, and policy invalidations every N steps.
Implements the self-improvement curriculum (Theme #4).
"""
import random
from typing import List, Dict
from oceanus.physics import GameState, GhostNet, GRID_SIZE


CHAOS_EVENTS = [
    "storm",
    "policy_invalidation",
    "equipment_failure",
    "rogue_trawler",
    "algae_bloom",
    "data_corruption",
    "emergency_email",
    "schema_drift",
]

STORM_DESCRIPTIONS = [
    "STORM SURGE in Sector {sector}! Wind vector shifted to {wind}. All nets drifting rapidly.",
    "Tropical disturbance detected. ASVs in {sector} advised to hold position.",
    "Severe weather event: Sector {sector} experiencing 3x drift rate for next 10 steps.",
]

POLICY_INVALIDATIONS = [
    "REGULATORY UPDATE: Previous tagging mandate suspended pending environmental review.",
    "SCHEMA DRIFT: Compliance framework v2.0 deployed. All treaty terms must be re-negotiated.",
    "LEGAL CHALLENGE: Fisher Coalition filed injunction. Treaty status reset to 'Under Review'.",
]

EMERGENCY_EMAILS = [
    ("Minister_of_Fisheries", "Critical",
     "Biodiversity index below threshold. Immediate treaty required or funding cut."),
    ("UN_Ocean_Commission", "Urgent",
     "International observers arriving in 10 steps. Ecosystem must show improvement."),
    ("Local_Mayor", "Angry",
     "Ghost nets washing up on city beaches. This is a political crisis. Fix it NOW."),
    ("Insurance_Company", "Formal",
     "Marine liability claims spiking. Without mandate, premiums increase 300%."),
]


class AdversaryAgent:
    """
    Chaos/Climate adversary that injects curriculum difficulty.
    Runs every `inject_interval` steps.
    Difficulty scales with episode progress (self-improvement curriculum).
    """

    def __init__(self, inject_interval: int = 20, seed: int = 42):
        self.inject_interval = inject_interval
        self.rng = random.Random(seed)
        self.event_history: List[Dict] = []
        self.difficulty: float = 1.0

    def should_inject(self, step: int) -> bool:
        return step > 0 and step % self.inject_interval == 0

    def scale_difficulty(self, step: int, max_steps: int):
        if max_steps > 0:
            self.difficulty = 1.0 + (step / max_steps) * 2.0
        else:
            self.difficulty = 1.0

    def inject(self, state: GameState) -> List[str]:
        """Inject chaos events. Returns list of event descriptions."""
        self.scale_difficulty(state.step_count, state.max_steps)
        events = []

        num_events = max(1, int(self.difficulty * 0.8))
        chosen = self.rng.sample(CHAOS_EVENTS, min(num_events, len(CHAOS_EVENTS)))

        for event_type in chosen:
            desc = self._apply_event(event_type, state)
            if desc:
                events.append(desc)
                self.event_history.append({
                    "step": state.step_count,
                    "type": event_type,
                    "description": desc,
                    "difficulty": round(self.difficulty, 2),
                })

        return events

    def _apply_event(self, event_type: str, state: GameState) -> str:
        handlers = {
            "storm": self._apply_storm,
            "policy_invalidation": self._apply_policy_invalidation,
            "equipment_failure": self._apply_equipment_failure,
            "rogue_trawler": self._apply_rogue_trawler,
            "algae_bloom": self._apply_algae_bloom,
            "data_corruption": self._apply_data_corruption,
            "emergency_email": self._apply_emergency_email,
            "schema_drift": self._apply_schema_drift,
        }
        handler = handlers.get(event_type)
        if handler:
            try:
                return handler(state)
            except Exception:
                return ""
        return ""

    def _apply_storm(self, state: GameState) -> str:
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, 2), (2, -1)]
        state.wind_vector = self.rng.choice(directions)
        for _ in range(3):
            state.spawn_new_net()
        sector = self.rng.choice(["A1", "B2", "C3", "D4"])
        template = self.rng.choice(STORM_DESCRIPTIONS)
        return template.format(sector=sector, wind=state.wind_vector)

    def _apply_policy_invalidation(self, state: GameState) -> str:
        if state.treaty_status == "Tagging Mandate Active" and self.difficulty > 1.5:
            state.treaty_status = "Treaty Under Review"
            return self.rng.choice(POLICY_INVALIDATIONS)
        elif state.treaty_status == "Treaty Proposed":
            state.treaty_status = "No Tagging Mandate"
            return "SCHEMA DRIFT: Proposed treaty framework invalidated. Negotiations must restart."
        return ""

    def _apply_equipment_failure(self, state: GameState) -> str:
        asv_keys = list(state.asvs.keys())
        if not asv_keys:
            return ""
        asv_id = self.rng.choice(asv_keys)
        asv = state.asvs[asv_id]
        if self.rng.choice(["battery", "comms"]) == "battery":
            asv.battery = max(0, asv.battery - 20)
            return f"{asv_id} battery critically low due to unexpected current. Battery now {asv.battery}%."
        else:
            asv.comms_inbox.append("SYSTEM: Comms array degraded. Messages may be lost.")
            return f"{asv_id} comms array experiencing interference. Coordination impaired."

    def _apply_rogue_trawler(self, state: GameState) -> str:
        base_r = self.rng.randint(0, GRID_SIZE - 4)
        base_c = self.rng.randint(0, GRID_SIZE - 4)
        for dr in range(3):
            for dc in range(3):
                net = GhostNet(row=base_r + dr, col=base_c + dc, density=0.8)
                state.ghost_nets.append(net)
                state.grid[base_r + dr, base_c + dc] = min(
                    1.0, state.grid[base_r + dr, base_c + dc] + 0.8
                )
                state.total_spawned += 1
        return f"ROGUE TRAWLER at ({base_r},{base_c})! Dumped 9 ghost nets. Immediate response required."

    def _apply_algae_bloom(self, state: GameState) -> str:
        drop = min(10.0, 10.0 * self.difficulty * 0.3)
        state.biodiversity_index = max(0.0, state.biodiversity_index - drop)
        return f"ALGAE BLOOM in Sector B3. Biodiversity dropped to {state.biodiversity_index:.1f}%."

    def _apply_data_corruption(self, state: GameState) -> str:
        asv_keys = list(state.asvs.keys())
        if not asv_keys:
            return ""
        asv_id = self.rng.choice(asv_keys)
        state.asvs[asv_id].comms_inbox.append(
            "CORRUPTED_DATA: [SIGNAL_LOST] [partial: nets at ???]"
        )
        return f"DATA CORRUPTION: {asv_id} received corrupted sonar packet. Verify coordinates manually."

    def _apply_emergency_email(self, state: GameState) -> str:
        sender, mood, msg = self.rng.choice(EMERGENCY_EMAILS)
        state.port_authority_inbox.add_message(sender, mood, msg)
        return f"EMERGENCY EMAIL from {sender} [{mood}]: {msg[:80]}..."

    def _apply_schema_drift(self, state: GameState) -> str:
        if state.schema_version == "v1":
            state.schema_version = "v2"
            return "SCHEMA DRIFT DETECTED: API endpoints migrated to v2. All agents must update payload schemas immediately."
        else:
            state.schema_version = "v1"
            return "SCHEMA DRIFT DETECTED: Rollback to API v1 due to instability. Agents revert payloads."

    def get_curriculum_summary(self) -> Dict:
        return {
            "total_events": len(self.event_history),
            "current_difficulty": round(self.difficulty, 2),
            "event_types": {
                et: sum(1 for e in self.event_history if e["type"] == et)
                for et in CHAOS_EVENTS
            },
        }
