# 🌊 Oceanus: Multi-Layer Ghost-Gear Recovery & Treaty Arena

> *The first multi-modal RL arena where LLM policy negotiations directly control the physics of ocean restoration.*

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/openenv)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)

---

## What is Oceanus?

Oceanus is a multi-agent RL environment that trains LLM agents on the real-world crisis of abandoned fishing gear (ghost nets) across transboundary ocean sectors. It operates on **three simultaneous layers**:

1. **Physical Swarm Layer** — 4 ASV drones coordinate on a 20×20 ocean grid with partial observability, sonar APIs, and inter-agent comms
2. **Policy & Stakeholder Layer** — 2 policy agents negotiate a gear-tagging treaty via email, directly reducing ghost net spawn rates
3. **Adversary Curriculum Layer** — A chaos agent injects storms, schema drift, and policy invalidations to force adaptive behavior

## Hackathon Theme Coverage

| Theme | Coverage |
|-------|----------|
| #1 Multi-Agent | ASV swarm cooperation + policy negotiation |
| #2 Long-Horizon | Treaty at step 10 pays off at step 50+ |
| #3.1 Professional Tools (Scaler Bonus) | `get_sonar_data`, `broadcast_alert` APIs |
| #3.2 Personalized Tasks (Patronus Bonus) | Stakeholder email handling + schema drift |
| #4 Self-Improvement (Snorkel Bonus) | Adversary generates escalating curriculum |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run demo episode
python main.py --max_steps 100

# Run dashboard
streamlit run dashboard/app.py

# Run API server
uvicorn dashboard.api:app --reload

# Train with GRPO
python train/train_grpo.py --model Qwen/Qwen1.5-0.5B-Instruct --steps 500

# Evaluate baseline vs trained
python train/evaluate.py --model_path ./oceanus_grpo_model
```

## Project Structure

```
oceanus/
├── physics.py       # 2D ocean grid, drift mechanics, ASV movement
├── models.py        # OpenEnv wrapper: reset(), step(), observations, rewards
├── adversary.py     # Chaos/curriculum agent
└── runner.py        # Episode runner + mock agents

dashboard/
├── app.py           # Streamlit split-screen UI
└── api.py           # FastAPI /step /reset /state endpoints

train/
├── train_grpo.py    # HuggingFace TRL + Unsloth GRPO training
└── evaluate.py      # Baseline vs trained comparison + plots

notebooks/
└── oceanus_colab_training.ipynb  # Google Colab training notebook

tests/
└── test_environment.py  # Full test suite
```

## Reward Structure

```
R_total = α * Σ R_local_asv + β * Σ R_local_policy + γ * R_global_ecosystem

R_local_asv:    +5 clean net | -0.1 move | +0.5 broadcast | -1 invalid JSON
R_local_policy: +10 treaty advance | +2 email reply | -5 ignored email | -1 invalid JSON
R_global:       +100 biodiversity ≥ 75% | +40 ≥ 50% | -20 collapse (episode end only)
```

## Observation Spaces

**ASV Agent:**
```json
{
  "agent_id": "ASV-1",
  "battery": 85,
  "sonar_3x3": [["empty", "net", "empty"], ["empty", "self", "empty"], ["empty", "empty", "empty"]],
  "comms_inbox": ["ASV-3: Found heavy nets in North-East quadrant."],
  "wind_hint": "Wind direction: (0, 1)"
}
```

**Policy Agent:**
```json
{
  "agent_id": "Port_Authority",
  "current_policy_status": "No Tagging Mandate",
  "inbox": [{"from": "Fisher_Bob", "mood": "Angry", "message": "I can't afford these tracker tags!"}],
  "net_spawn_rate": "High"
}
```

## Action Schemas

**ASV:** `{"intent": "move|scan|clean|broadcast", "direction": "north|south|east|west", "message": "string"}`

**Policy:** `{"intent": "reply_email|propose_treaty|accept_treaty|reject_treaty", "target": "string", "content": "string"}`

## Training

Uses **GRPO** (Group Relative Policy Optimization) via HuggingFace TRL + Unsloth 4-bit quantization.

Expected improvement after 500 training steps:
- Reward: -300 → +200+
- Biodiversity: ~93% → 80%+ (with active net cleaning)
- Treaty success rate: 60% → 90%+

## License

MIT
