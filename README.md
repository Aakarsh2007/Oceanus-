# 🌊 Oceanus: Multi-Layer Ghost-Gear Recovery & Treaty Arena

> *The first multi-modal RL arena where LLM policy negotiations directly control the physics of ocean restoration.*

[![HF Space](https://img.shields.io/badge/🤗_Space-Live_Demo-blue)](https://aakarsh2007-oceanus-ai.hf.space)
[![Colab](https://img.shields.io/badge/Colab-Training_Notebook-orange)](https://colab.research.google.com/drive/1lJ_EKDEZptQorExyGS6jiMMX98Qw-uOs)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Aakarsh2007/Oceanus-)
[![Video](https://img.shields.io/badge/YouTube-Demo_Video-red)](https://youtube.com/YOUR_VIDEO_ID)

---

## 🔗 Submission Links

| Resource | Link |
|----------|------|
| 🤗 HuggingFace Space (Live Demo) | https://aakarsh2007-oceanus-ai.hf.space |
| 📓 Colab Training Notebook | https://colab.research.google.com/drive/1lJ_EKDEZptQorExyGS6jiMMX98Qw-uOs |
| 💻 GitHub Repository | https://github.com/Aakarsh2007/Oceanus- |
| 🎥 Demo Video / Blog | https://youtube.com/YOUR_VIDEO_ID |

> **Update the YouTube link before submission deadline.**

---

## 🌊 What is Oceanus?

Oceanus is a multi-agent RL environment built on the real-world crisis of **abandoned fishing gear (ghost nets)** — 640,000 tonnes of gear lost annually, killing marine life for decades. No single agent can solve it. It requires coordination, negotiation, and long-horizon planning.

The environment operates on **three simultaneous layers**:

1. **Physical Swarm Layer** — 4 ASV drones coordinate on a 20×20 ocean grid under partial observability. Each agent sees only a 3×3 sonar window. They must broadcast coordinates, sweep sectors, and clean nets.

2. **Policy & Stakeholder Layer** — 2 policy agents (Port Authority + Fleet Manager) negotiate a gear-tagging treaty via stakeholder emails. A successful treaty reduces ghost net spawn rate from High → Low, directly helping the ASV layer.

3. **Adversary Curriculum Layer** — A chaos agent injects storms, schema drift, rogue trawlers, and policy invalidations. Difficulty scales from 1.0 → 3.0 over the episode, forcing agents to adapt.

**The key insight**: a semantic treaty negotiated in text directly alters the physics of a 2D grid. This cross-layer causality is what makes Oceanus novel.

---

## 🏆 Hackathon Theme Coverage

| Theme | How Oceanus Covers It |
|-------|----------------------|
| **#1 Multi-Agent** | 4 ASV drones cooperate + 2 policy agents negotiate. Heterogeneous agents with different obs/action spaces. |
| **#2 Long-Horizon** | Treaty at step 7 pays off at step 50+. Global reward only at episode end. Sparse reward forces planning. |
| **#3.1 Professional Tools** | Sonar API, broadcast API, schema versioning v1↔v2, stakeholder email system. |
| **#3.2 Personalized Tasks** | Policy agents handle mood-aware emails from angry fishers, NGOs, government officials. |
| **#4 Self-Improvement** | Adversary curriculum scales difficulty with episode progress. 8 chaos event types. |

---

## 📊 Training Results

**Algorithm**: GRPO (Group Relative Policy Optimization) — same algorithm as DeepSeek-R1
**Model**: `unsloth/Qwen2-0.5B-Instruct` with 4-bit LoRA (r=16), 2.1M trainable parameters
**Training**: 600 steps on Google Colab T4 GPU

![Training Curve](https://raw.githubusercontent.com/Aakarsh2007/Oceanus-/main/notebooks/training_curve.png)

> *Trained agent (cyan) climbs from -1.0 → +5.5 over 600 GRPO steps. Red dashed = untrained baseline (random text). Gap = +6.5 reward points.*

### Before vs After Training

| Metric | Untrained Baseline | GRPO Trained |
|--------|-------------------|--------------|
| Output format | Random text | Valid JSON ✅ |
| Correct intent | Never | Consistently ✅ |
| Step reward | -1.0 | +5.0 to +8.0 |
| Improvement | — | **+9.0 reward points/step** |

**Live inference proof** — given sonar showing a net, trained model outputs:
```json
{"intent": "clean", "direction": "south"}
```
Untrained model outputs: *"I think I should navigate toward the detected anomaly..."* → reward: -1.0

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run demo episode (mock agents)
python main.py --max_steps 100

# Launch 3D Mission Control dashboard
python run.py
# Opens at http://localhost:8000

# Run test suite (59 tests)
python -m pytest tests/test_environment.py -q
```

---

## 🏗️ Architecture

```
oceanus/
├── physics.py        # 20×20 grid, ghost net drift, ASV movement, sonar
├── models.py         # OpenEnv wrapper: reset(), step(), CTDE reward engine
├── adversary.py      # 8-type chaos agent, difficulty curriculum
├── runner.py         # Episode runner, MockASVAgent, MockPolicyAgent, LLMAgent
└── demo_recorder.py  # Records baseline + trained episodes to JSON

dashboard/
├── server.py         # FastAPI + WebSocket backend
└── index.html        # Three.js 3D frontend (single-page app)

train/
├── train_grpo.py     # GRPO training (HuggingFace TRL + Unsloth)
└── evaluate.py       # Baseline vs trained evaluation

data/
├── baseline_episode.json   # Pre-recorded baseline (reward: -988)
└── trained_episode.json    # Pre-recorded trained (reward: +270, 13 nets cleaned)

notebooks/
└── oceanus_final_submission.ipynb  # Full Colab training walkthrough
```

---

## 🎯 Reward Structure (CTDE)

Centralized Training, Decentralized Execution:

```
R_total = α·Σ R_local_asv + β·Σ R_local_policy + γ·R_global_ecosystem

R_local_asv:    +5.0  clean net
                +0.5  useful broadcast
                -0.1  move (energy cost)
                -1.0  invalid JSON

R_local_policy: +10.0 treaty advance
                +2.0  email reply
                -5.0  ignored email (>5 steps old)
                -1.0  invalid JSON

R_global:       +100  biodiversity ≥ 75%  (full recovery)
                +40   biodiversity ≥ 50%
                +10   biodiversity ≥ 25%
                -20   ecosystem collapse
                ↑ awarded at episode end ONLY — forces long-horizon planning
```

---

## 👁️ Observation Spaces

**ASV Agent** (partial observability — 3×3 sonar only):
```json
{
  "agent_id": "ASV-1",
  "position": {"row": 5, "col": 8, "sector": "B2"},
  "battery": 85,
  "sonar_3x3": [["empty", "net", "empty"], ["empty", "self", "empty"], ["empty", "empty", "empty"]],
  "comms_inbox": ["ASV-3: Found heavy nets in North-East quadrant."],
  "wind_hint": "Wind direction: (0, 1)",
  "biodiversity_index": 72.4
}
```

**Policy Agent**:
```json
{
  "agent_id": "Port_Authority",
  "current_policy_status": "No Tagging Mandate",
  "net_spawn_rate": "High",
  "inbox": [{"from": "Fisher_Bob", "mood": "Angry", "message": "I can't afford these tracker tags!"}],
  "unanswered_emails": 3
}
```

---

## ⚡ Adversary Chaos Events

| Event | Effect | Tests |
|-------|--------|-------|
| `storm` | Shifts wind, spawns 3 nets | Physical adaptation |
| `policy_invalidation` | Resets treaty mid-episode | Long-horizon recovery |
| `equipment_failure` | Drains battery / corrupts comms | Resource management |
| `rogue_trawler` | Dumps 9 nets in 3×3 cluster | Emergency response |
| `algae_bloom` | Directly drops biodiversity | Urgency prioritization |
| `data_corruption` | Injects garbage into comms | Noise robustness |
| `emergency_email` | Urgent stakeholder message | Policy responsiveness |
| `schema_drift` | Flips API v1 ↔ v2 | API adaptation |

Difficulty scales 1.0 → 3.0 over episode duration.

---

## 🧠 Why This Environment Matters

Most RL environments train agents on fixed, isolated tasks. Oceanus trains agents on **emergent cross-layer causality** — a text negotiation changes physical spawn mechanics. This is the kind of reasoning that matters in real enterprise systems, policy decisions, and multi-stakeholder coordination problems.

The environment is designed to be hard to game: an agent that just outputs `{"intent": "clean"}` every step without moving toward nets will score poorly. An agent that proposes a treaty but ignores stakeholder emails will get penalized. The reward structure requires genuinely intelligent behavior.

---

## 📋 Environment API

```python
from oceanus.models import OceanusEnv
from oceanus.adversary import AdversaryAgent

env = OceanusEnv(seed=42, max_steps=120)
adversary = AdversaryAgent(inject_interval=20)

# Standard Gym-style API
obs = env.reset()
# obs = {agent_id: {"observation": dict, "prompt": str}, ...}

actions = {"ASV-1": '{"intent": "clean"}', "Port_Authority": '{"intent": "propose_treaty", ...}'}
obs, rewards, done, info = env.step(actions)
# rewards = {"ASV-1": 5.0, "Port_Authority": 10.0, "__total__": 18.0, ...}
```

---

## License

MIT
https://aakarsh2007-oceanus-ai.hf.spacePlease