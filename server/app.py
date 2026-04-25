"""
Oceanus OpenEnv Server App
Creates the FastAPI application using openenv-core's server factory.
"""
import sys
import os

# Ensure oceanus package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_app
from server.oceanus_environment import OceanusEnvironment
from models import OceanusAction, OceanusObservation

env = OceanusEnvironment(seed=42, max_steps=120, chaos_interval=25)
app = create_app(env, OceanusAction, OceanusObservation)
