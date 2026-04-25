"""
Oceanus OpenEnv Models
Typed Action, Observation, and State classes compliant with openenv-core spec.
Falls back to plain Pydantic BaseModel if openenv-core is not installed.
"""
from typing import Optional, List, Dict, Any
from pydantic import Field, BaseModel

# Graceful fallback if openenv-core not installed
try:
    from openenv.core.env_server.interfaces import (
        Action as _BaseAction,
        Observation as _BaseObservation,
        State as _BaseState,
    )
except ImportError:
    # Fallback base classes — same interface, no openenv dependency
    class _BaseAction(BaseModel):
        pass
    class _BaseObservation(BaseModel):
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)
    class _BaseState(BaseModel):
        episode_id: int = 0
        step_count: int = 0


# ── ACTION ────────────────────────────────────────────────────────────────────

class OceanusAction(_BaseAction):
    """
    Action for any Oceanus agent (ASV or Policy).

    ASV:    {"agent_id": "ASV-1", "intent": "clean"}
            {"agent_id": "ASV-1", "intent": "move", "direction": "north"}
            {"agent_id": "ASV-1", "intent": "broadcast", "message": "Net at B2"}
    Policy: {"agent_id": "Port_Authority", "intent": "propose_treaty",
             "target": "Fleet_Manager", "content": "50% subsidy"}
    """
    agent_id: str = Field(..., description="Agent ID: ASV-1..4, Port_Authority, Fleet_Manager")
    intent: str = Field(..., description="move|clean|broadcast|scan|propose_treaty|accept_treaty|reply_email|reject_treaty")
    direction: Optional[str] = Field(None, description="For move: north|south|east|west|stay")
    message: Optional[str] = Field(None, description="For broadcast: message text")
    target: Optional[str] = Field(None, description="For policy actions: recipient")
    content: Optional[str] = Field(None, description="For policy actions: message content")


# ── OBSERVATION ───────────────────────────────────────────────────────────────

class OceanusObservation(_BaseObservation):
    """Combined observation for all agents after one step."""
    step: int = Field(default=0)
    biodiversity: float = Field(default=100.0)
    active_nets: int = Field(default=0)
    total_cleaned: int = Field(default=0)
    treaty_status: str = Field(default="No Tagging Mandate")
    net_spawn_rate: str = Field(default="High")
    grid: List[List[float]] = Field(default_factory=list)
    asvs: Dict[str, Any] = Field(default_factory=dict)
    ghost_nets: List[Dict[str, Any]] = Field(default_factory=list)
    wind_vector: List[int] = Field(default_factory=lambda: [0, 1])
    total_reward: float = Field(default=0.0)
    event_log: List[Dict[str, Any]] = Field(default_factory=list)
    agent_observations: Dict[str, Any] = Field(default_factory=dict)


# ── STATE ─────────────────────────────────────────────────────────────────────

class OceanusState(_BaseState):
    """Episode state metadata."""
    seed: int = Field(default=42)
    max_steps: int = Field(default=120)
    chaos_events_fired: int = Field(default=0)
    difficulty: float = Field(default=1.0)
    schema_version: str = Field(default="v1")
