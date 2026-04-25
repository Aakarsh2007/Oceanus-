"""
Oceanus OpenEnv Client
Use this to connect to a running Oceanus environment server.
"""
from openenv.core import EnvClient
from models import OceanusAction, OceanusObservation, OceanusState


class OceanusEnvClient(EnvClient[OceanusAction, OceanusObservation, OceanusState]):
    """
    Client for the Oceanus multi-agent ghost-gear recovery environment.

    Example usage:
        # Connect to HuggingFace Space
        with OceanusEnvClient(base_url="https://aakarsh2007-oceanus-ai.hf.space").sync() as client:
            obs = client.reset()
            print(f"Biodiversity: {obs.observation.biodiversity}%")
            print(f"Active nets: {obs.observation.active_nets}")

            # ASV-1 cleans a net
            result = client.step(OceanusAction(agent_id="ASV-1", intent="clean"))
            print(f"Reward: {result.reward}")

            # Port Authority proposes treaty
            result = client.step(OceanusAction(
                agent_id="Port_Authority",
                intent="propose_treaty",
                target="Fleet_Manager",
                content="50% subsidy on tracking tags"
            ))

        # Async usage
        async with OceanusEnvClient(base_url="https://aakarsh2007-oceanus-ai.hf.space") as client:
            obs = await client.reset()
            result = await client.step(OceanusAction(agent_id="ASV-1", intent="scan"))
    """
    pass
