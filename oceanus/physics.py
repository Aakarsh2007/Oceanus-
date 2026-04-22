"""
2D Physics Engine + State Management
Ocean grid, ghost net drift mechanics, ASV movement, sonar.
"""
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque


GRID_SIZE = 20
NUM_ASVS = 4
SONAR_RADIUS = 1  # 3x3 around ASV

# Sectors for narrative (A-D rows, 1-4 cols)
SECTORS = {
    (r, c): f"{chr(65 + r // 5)}{c // 5 + 1}"
    for r in range(GRID_SIZE)
    for c in range(GRID_SIZE)
}


@dataclass
class GhostNet:
    row: int
    col: int
    density: float = 1.0
    age: int = 0

    def drift(self, wind_vector: Tuple[int, int], grid_size: int = GRID_SIZE):
        dr, dc = wind_vector
        dr += random.choice([-1, 0, 0, 1])
        dc += random.choice([-1, 0, 0, 1])
        self.row = max(0, min(grid_size - 1, self.row + dr))
        self.col = max(0, min(grid_size - 1, self.col + dc))
        self.age += 1


@dataclass
class ASVState:
    asv_id: str
    row: int
    col: int
    battery: int = 100
    nets_cleaned: int = 0
    comms_inbox: List[str] = field(default_factory=list)

    def move(self, direction: str, grid_size: int = GRID_SIZE) -> bool:
        moves = {
            "north": (-1, 0), "south": (1, 0),
            "east": (0, 1), "west": (0, -1), "stay": (0, 0),
        }
        if direction not in moves:
            return False
        dr, dc = moves[direction]
        self.row = max(0, min(grid_size - 1, self.row + dr))
        self.col = max(0, min(grid_size - 1, self.col + dc))
        self.battery = max(0, self.battery - 1)
        return True


@dataclass
class PolicyInbox:
    """Inbox queue for policy agents."""
    messages: deque = field(default_factory=deque)
    sent_messages: List[Dict] = field(default_factory=list)
    unanswered_count: int = 0

    def add_message(self, sender: str, mood: str, content: str):
        self.messages.append({"from": sender, "mood": mood, "message": content, "age": 0})
        self.unanswered_count += 1

    def age_messages(self):
        for msg in self.messages:
            msg["age"] += 1

    def pop_message(self, sender: str) -> Optional[Dict]:
        """Remove and return the first message from sender. Returns None if not found."""
        for i, msg in enumerate(list(self.messages)):
            if msg["from"] == sender:
                # Remove from deque by rebuilding without this element
                self.messages = deque(
                    m for j, m in enumerate(self.messages) if j != i
                )
                self.unanswered_count = max(0, self.unanswered_count - 1)
                return msg
        return None


class GameState:
    """Central state manager for the Oceanus environment."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self.step_count = 0
        self.max_steps = 200

        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        self.ghost_nets: List[GhostNet] = []

        self.wind_vector: Tuple[int, int] = (0, 1)
        self.wind_change_interval = 20

        self.asvs: Dict[str, ASVState] = {}
        self._init_asvs()

        self.treaty_status: str = "No Tagging Mandate"
        self.treaty_subsidy: float = 0.0
        self.net_spawn_rate: str = "High"
        self.spawn_interval: int = 5

        self.port_authority_inbox = PolicyInbox()
        self.fleet_manager_inbox = PolicyInbox()

        self.biodiversity_index: float = 100.0
        self.total_cleaned: int = 0
        self.total_spawned: int = 0
        self.chaos_events: List[str] = []
        self.done: bool = False

        # Enterprise Features
        self.schema_version: str = "v1"
        self.base_lat: float = 35.0  # Pacific Garbage Patch
        self.base_lon: float = -140.0
        self.lat_step: float = 0.05
        self.lon_step: float = 0.05

        self._spawn_initial_nets(count=15)
        self._seed_stakeholder_emails()

    def _init_asvs(self):
        starts = [(2, 2), (2, 17), (17, 2), (17, 17)]
        for i, (r, c) in enumerate(starts):
            aid = f"ASV-{i+1}"
            self.asvs[aid] = ASVState(asv_id=aid, row=r, col=c)

    def _spawn_initial_nets(self, count: int):
        for _ in range(count):
            r = int(self.rng.integers(0, GRID_SIZE))
            c = int(self.rng.integers(0, GRID_SIZE))
            net = GhostNet(row=r, col=c, density=round(float(self.rng.uniform(0.5, 1.0)), 2))
            self.ghost_nets.append(net)
            self.grid[r, c] = min(1.0, self.grid[r, c] + net.density)
            self.total_spawned += 1

    def _seed_stakeholder_emails(self):
        stakeholder_emails = [
            ("Fisher_Bob", "Angry", "I can't afford these tracker tags! This mandate will bankrupt us."),
            ("Coastal_Resident_Maria", "Concerned", "The ghost nets are destroying our beach. Please act NOW."),
            ("NGO_OceanWatch", "Urgent", "Biodiversity in Sector B is critical. We need immediate action."),
            ("Fisher_Coalition", "Hostile", "Any tagging mandate without subsidies is a non-starter. We will protest."),
        ]
        for sender, mood, msg in stakeholder_emails:
            self.port_authority_inbox.add_message(sender, mood, msg)

        self.fleet_manager_inbox.add_message(
            "Port_Authority_Agent", "Neutral",
            "Proposing a 50% subsidy for tags. Do you accept? This could reduce spawn rates significantly."
        )

    def spawn_new_net(self):
        r = int(self.rng.integers(0, GRID_SIZE))
        c = int(self.rng.integers(0, GRID_SIZE))
        net = GhostNet(row=r, col=c, density=round(float(self.rng.uniform(0.3, 1.0)), 2))
        self.ghost_nets.append(net)
        self.grid[r, c] = min(1.0, self.grid[r, c] + net.density)
        self.total_spawned += 1

    def drift_nets(self):
        """Apply wind drift to all ghost nets and rebuild grid."""
        for net in self.ghost_nets:
            net.drift(self.wind_vector)
        # Rebuild grid from scratch
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        for net in self.ghost_nets:
            self.grid[net.row, net.col] = min(1.0, self.grid[net.row, net.col] + net.density)

    def update_wind(self):
        if self.step_count % self.wind_change_interval == 0:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
            self.wind_vector = random.choice(directions)

    def update_biodiversity(self):
        active_nets = len(self.ghost_nets)
        degradation = active_nets * 0.3
        recovery = self.total_cleaned * 0.1
        self.biodiversity_index = max(0.0, min(100.0, 100.0 - degradation + recovery))

    def update_spawn_rate(self):
        if self.treaty_status == "Tagging Mandate Active":
            self.spawn_interval = 15
            self.net_spawn_rate = "Low"
        elif self.treaty_status in ("Partial Subsidy Agreed", "Treaty Under Review"):
            self.spawn_interval = 10
            self.net_spawn_rate = "Medium"
        else:
            self.spawn_interval = 5
            self.net_spawn_rate = "High"

    def get_sonar_reading(self, asv: ASVState) -> List[List[str]]:
        """Return 3x3 sonar grid centered on ASV."""
        sonar = []
        for dr in range(-SONAR_RADIUS, SONAR_RADIUS + 1):
            row_data = []
            for dc in range(-SONAR_RADIUS, SONAR_RADIUS + 1):
                r, c = asv.row + dr, asv.col + dc
                if dr == 0 and dc == 0:
                    row_data.append("self")
                elif 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    if self.grid[r, c] > 0:
                        row_data.append("net")
                    else:
                        other = any(
                            a.row == r and a.col == c
                            for aid, a in self.asvs.items()
                            if aid != asv.asv_id
                        )
                        row_data.append("asv" if other else "empty")
                else:
                    row_data.append("boundary")
            sonar.append(row_data)
        return sonar

    def clean_net_at(self, row: int, col: int) -> bool:
        """Remove ghost net(s) at given position. Returns True if any cleaned."""
        before = len(self.ghost_nets)
        self.ghost_nets = [n for n in self.ghost_nets if not (n.row == row and n.col == col)]
        cleaned_count = before - len(self.ghost_nets)
        if cleaned_count > 0:
            self.grid[row, col] = 0.0
            self.total_cleaned += cleaned_count
            return True
        return False

    def get_sector(self, row: int, col: int) -> str:
        return SECTORS.get((row, col), "Unknown")

    def tick(self):
        """Advance physics by one step."""
        self.step_count += 1
        self.update_wind()
        self.drift_nets()
        self.update_spawn_rate()

        if self.step_count % self.spawn_interval == 0:
            self.spawn_new_net()

        self.port_authority_inbox.age_messages()
        self.fleet_manager_inbox.age_messages()
        self.update_biodiversity()

        if self.step_count >= self.max_steps:
            self.done = True
