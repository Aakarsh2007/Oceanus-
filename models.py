"""
Oceanus OpenEnv Models
Typed Action, Observation, and State classes compliant with openenv-core spec.
"""
from typing import Optional, List, Dict, Any
from pydantic import Field
from openenv.core.env_server.interfaces import (
    Action as _BaseAction,
    Observation as _BaseObservation,
    State as _BaseState,
)


# ── ACTION ────────────────────────────────────────────────────────────────────

class OceanusAction(_BaseAction):
    """
    Action for any Oceanus agent (ASV or Policy).

    ASV actions:
      {"intent": "move", "direction": "north|south|east|west|stay"}
      {"intent": "clean"}
      {"intent": "broadcast", "message": "Net at sector B2"}
      {"intent": "scan"}

    Policy actions:
      {"intent": "propose_treaty", "target": "Fleet_Manager", "content": "..."}
      {"intent": "accept_treaty", "target": "Port_Authority", "content": "..."}
      {"intent": "reply_email", "target": "Fisher_Bob", "content": "..."}
    """
    agent_id: str = Field(..., description="Agent ID: ASV-1..4, Port_Authority, Fleet_Manager")
    intent: str = Field(..., description="Action intent: move|clean|broadcast|scan|propose_treaty|accept_treaty|reply_email|reject_treaty")
    direction: Optional[str] = Field(None, description="For move: north|south|east|west|stay")
    message: Optional[str] = Field(None, description="For broadcast: message text")
    target: Optional[str] = Field(None, description="For policy actions: recipient agent")
    content: Optional[str] = Field(None, description="For policy actions: message content")


# ── OBSERVATION ───────────────────────────────────────────────────────────────

class ASVObservation(_BaseObservation):
    """Observation for an ASV drone agent."""
    agent_id: str = Field(..., description="Agent identifier")
    position: Dict[str, Any] = Field(..., description="Row, col, sector")
    battery: int = Field(..., description="Battery level 0-100")
    sonar_3x3: List[List[str]] = Field(..., description="3x3 sonar grid: empty|net|asv|self|boundary")
    on_net: bool = Field(..., description="Whether agent is currently on a ghost net")
    comms_inbox: List[str] = Field(default_factory=list, description="Messages from other ASVs")
    wind_hint: str = Field(..., description="Current wind direction hint")
    biodiversity_index: float = Field(..., description="Current ecosystem biodiversity %")
    step: int = Field(..., description="Current episode step")
    prompt: str = Field(..., description="Natural language prompt for LLM agent")


class PolicyObservation(_BaseObservation):
    """Observation for a policy agent (Port_Authority or Fleet_Manager)."""
    agent_id: str = Field(..., description="Agent identifier")
    current_policy_status: str = Field(..., description="Current treaty status")
    net_spawn_rate: str = Field(..., description="High|Medium|Low")
    biodiversity_index: float = Field(..., description="Current ecosystem biodiversity %")
    inbox: List[Dict[str, Any]] = Field(default_factory=list, description="Stakeholder messages")
    unanswered_emails: int = Field(..., description="Number of unanswered emails")
    step: int = Field(..., description="Current episode step")
    prompt: str = Field(..., description="Natural language prompt for LLM agent")


class OceanusObservation(_BaseObservation):
    """Combined observation for all agents in one step."""
    step: int = Field(..., description="Current episode step")
    biodiversity: float = Field(..., description="Ecosystem biodiversity index %")
    active_nets: int = Field(..., description="Number of active ghost nets")
    total_cleaned: int = Field(..., description="Total nets cleaned this episode")
    treaty_status: str = Field(..., description="Current treaty negotiation status")
    net_spawn_rate: str = Field(..., description="Current net spawn rate")
    grid: List[List[float]] = Field(..., description="20x20 ocean grid density values")
    asvs: Dict[str, Any] = Field(..., description="ASV positions and states")
    ghost_nets: List[Dict[str, Any]] = Field(..., description="Active ghost net positions")
    wind_vector: List[int] = Field(..., description="Current wind direction vector")
    total_reward: float = Field(..., description="Cumulative episode reward")
    event_log: List[Dict[str, Any]] = Field(default_factory=list, description="Recent events")
    agent_observations: Dict[str, Any] = Field(default_factory=dict, description="Per-agent observations and prompts")


# ── STATE ─────────────────────────────────────────────────────────────────────

class OceanusState(_BaseState):
    """Episode state metadata for Oceanus."""
    episode_id: int = Field(default=0, description="Episode number")
    seed: int = Field(default=42, description="Random seed used")
    max_steps: int = Field(default=120, description="Maximum steps per episode")
    chaos_events_fired: int = Field(default=0, description="Number of chaos events fired")
    difficulty: float = Field(default=1.0, description="Current adversary difficulty")
    schema_version: str = Field(default="v1", description="Current API schema version")
