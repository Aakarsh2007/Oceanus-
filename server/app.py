"""
Oceanus OpenEnv Server App
Exposes the environment via REST API (OpenEnv-compatible).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import OceanusAction, OceanusObservation, OceanusState
from server.oceanus_environment import OceanusEnvironment

app = FastAPI(title="Oceanus OpenEnv Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

env = OceanusEnvironment(seed=42, max_steps=120, chaos_interval=25)

@app.post("/reset", response_model=OceanusObservation)
def reset():
    return env.reset()

@app.post("/step", response_model=OceanusObservation)
def step(action: OceanusAction):
    return env.step(action)

@app.get("/state", response_model=OceanusState)
def state():
    return env.state()

@app.get("/health")
def health():
    return {"status": "ok", "environment": "Oceanus-v1"}
