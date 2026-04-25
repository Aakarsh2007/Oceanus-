"""
Oceanus OpenEnv Client
Connect to a running Oceanus environment server.
"""
from models import OceanusAction, OceanusObservation, OceanusState

try:
    from openenv.core import EnvClient

    class OceanusEnvClient(EnvClient[OceanusAction, OceanusObservation, OceanusState]):
        """
        Client for the Oceanus environment.

        Usage:
            with OceanusEnvClient(base_url="https://aakarsh2007-oceanus-ai.hf.space").sync() as client:
                obs = client.reset()
                result = client.step(OceanusAction(agent_id="ASV-1", intent="clean"))
        """
        pass

except ImportError:
    # Fallback HTTP client when openenv-core not installed
    import requests

    class OceanusEnvClient:
        def __init__(self, base_url: str):
            self.base_url = base_url.rstrip("/")

        def reset(self) -> OceanusObservation:
            r = requests.post(f"{self.base_url}/reset")
            return OceanusObservation(**r.json())

        def step(self, action: OceanusAction) -> OceanusObservation:
            r = requests.post(f"{self.base_url}/step", json=action.model_dump())
            return OceanusObservation(**r.json())

        def state(self) -> OceanusState:
            r = requests.get(f"{self.base_url}/state")
            return OceanusState(**r.json())
