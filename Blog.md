---
title: "Oceanus: A Cross-Layer Multi-Agent Arena for Solving the Ghost Gear Crisis"
thumbnail: "https://raw.githubusercontent.com/Aakarsh2007/Oceanus-/main/notebooks/training_curve.png"
tags:
  - openenv
  - multi-agent
  - reinforcement-learning
  - grpo
  - sustainability
  - unsloth
  - trl
---

# 🌊 Oceanus: Training LLMs to Solve the Ocean's Silent Crisis

*How we built a multi-layer RL environment where language models negotiate policies to change the physical world—and trained them using GRPO.*

---

## 🛑 The Problem: The Ghost Gear Crisis

Every year, **640,000 tonnes of fishing gear** are abandoned in our oceans. These "ghost nets" are the deadliest form of marine plastic, silently killing marine life and devastating ecosystems for decades. 

Solving this crisis is an incredibly complex real-world challenge. It cannot be solved by a single autonomous drone, nor can it be solved purely by policy-makers sitting in an office. It requires **coordination, negotiation, and long-horizon planning** across different domains: the physical deployment of cleanup swarms, and the high-level negotiation of international fishing treaties.

This is exactly the type of complex, multi-layered problem that modern LLM agents struggle with. They often fail to understand how actions in one domain (like sending an email) affect outcomes in another (like the spawn rate of trash in the ocean). 

To bridge this gap, we built **Oceanus**.

---

## 🌍 Introducing Oceanus: A Multi-Layer Arena

**Oceanus** is an OpenEnv-compliant reinforcement learning environment designed to push the boundaries of multi-agent interactions and long-horizon planning. It simulates the ghost gear crisis across **three simultaneous layers**:

### 1. The Physical Swarm Layer 🚤
A swarm of 4 ASV (Autonomous Surface Vehicle) drones coordinates on a 20×20 ocean grid under partial observability. Each agent only sees a 3×3 sonar window. To succeed, they must broadcast coordinates, manage battery life, sweep sectors, and clean up nets.

### 2. The Policy & Stakeholder Layer 🏛️
While the drones clean the ocean, 2 policy agents (a Port Authority and a Fleet Manager) negotiate a gear-tagging treaty via a stakeholder email system. They must manage angry emails from fishers, NGOs, and government officials. **The magic:** If they successfully negotiate a treaty, the physical spawn rate of ghost nets drops from *High* to *Low*, directly aiding the ASV swarm.

### 3. The Adversary Curriculum Layer ⛈️
To prevent the agents from memorizing a simple path, an automated chaos agent injects localized storms, schema drift (API v1 to v2), rogue trawlers, and even policy invalidations. Difficulty scales from 1.0 to 3.0 over the episode, forcing agents to adapt and self-improve.

---

## 💡 Why Oceanus is Innovative

Most RL environments (like chess, snake, or grid-worlds) train agents on fixed, isolated tasks. Oceanus is fundamentally different because it introduces **emergent cross-layer causality**. 

A semantic treaty negotiated in text (Policy Layer) directly alters the physics and spawn mechanics of a 2D grid (Physical Layer). This forces the LLMs to build a **persistent world model** and understand how abstract negotiations yield tangible, physical results over a long horizon. 

This environment is extremely hard to "game." An agent that blindly outputs `{"intent": "clean"}` every step without moving toward nets will exhaust its battery and fail. An agent that proposes a treaty but ignores urgent stakeholder emails will be heavily penalized. Real intelligence and coordination are required.

---

## ⚙️ A Universal Text-to-Action RL Engine

While Oceanus is themed around ocean recovery, underneath the UI lies a **completely domain-agnostic, universal Reinforcement Learning pipeline**. 

This is not just a game; it is a **Text-to-Action Engine**. Because the agents use human language (JSON/text) instead of rigid action spaces, you can retrain this exact system to solve almost any problem by changing three specific things:

### 1. Change the System Prompt (The Rules)
If you wanted to build an algorithmic trader, you just change the prompt: 
> *"You are an algorithmic trading AI. You MUST output valid JSON. Allowed intents: 'buy', 'sell', 'hold'."*

### 2. Change the Environment Data (The Observation)
Instead of sonar data, feed it different text: 
* **Medical Triage AI:** `"Patient Symptoms: ['fever', 'cough']. Blood pressure: 140/90."`
* **Cybersecurity AI:** `"Network Log: [Multiple failed SSH login attempts from IP 192.168.1.50]."`

### 3. Change the Reward Functions (The Goal)
The AI learns whatever you reward it for. If building a Smart Home Energy agent:
> *If the observation says "Room is empty" and the AI outputs `"intent": "turn_off_lights"`, give it `+10` points. If it outputs `"turn_on_heater"`, give it `-5` points.*

Because this pipeline uses an LLM, you do not need to build complex 3D physics simulators for every new problem. You just write Python code that generates text-based observations and assigns points. The GRPO algorithm automatically rewires the model's 8.8 million LoRA parameters to maximize those points, essentially acting as a factory that manufactures custom AI brains.

---

## 🧠 Training Pipeline & Reward Structure

We trained the agents using **GRPO (Group Relative Policy Optimization)**—the same algorithm powering DeepSeek-R1—leveraging the **Hugging Face TRL** library and **Unsloth** for rapid 4-bit LoRA fine-tuning.

We used `unsloth/Qwen2-0.5B-Instruct` to prove that even a small model can learn complex cross-layer reasoning when provided with a rich environment and a carefully designed reward signal.

### Centralized Training, Decentralized Execution (CTDE)
Our reward mechanism is highly composable:
* **Local ASV Rewards:** +5.0 for cleaning a net, +0.5 for a useful broadcast, -0.1 for movement (energy cost).
* **Local Policy Rewards:** +10.0 for advancing the treaty, +2.0 for handling an email, -5.0 for ignoring an email.
* **Global Long-Horizon Reward:** Up to +100 awarded *only* at the end of the episode if the overall biodiversity index is restored. This sparse reward forces multi-step planning.

### 📈 The Results: Measurable Improvement

Over just 600 GRPO steps on a single Colab T4 GPU, the agents exhibited remarkable emergent behavior.

![Training Curve](https://raw.githubusercontent.com/Aakarsh2007/Oceanus-/main/notebooks/training_curve.png)
*(The trained policy rapidly climbs to a stable, positive reward, vastly outperforming the random-text baseline.)*

**Before vs. After Training:**

| Metric | Untrained Baseline | GRPO Trained |
|--------|-------------------|--------------|
| **Output format** | Random text/hallucinations | Valid JSON strict adherence ✅ |
| **Correct intent** | Never | Consistently matching state ✅ |
| **Step reward** | -1.0 | +5.0 to +8.0 |
| **Improvement** | — | **+9.0 reward points/step** |

When presented with a sonar reading showing a net, the baseline model would hallucinate conversational text (resulting in a -1.0 penalty). The trained model accurately deduces the state and outputs strict tool-use JSON: `{"intent": "clean", "direction": "south"}`.

---

## 🏆 Hackathon Themes Covered

Oceanus is an ambitious environment that heavily aligns with the OpenEnv Hackathon themes:
1. **Multi-Agent Interactions (#1):** Heterogeneous cooperation between ASVs and Policy makers.
2. **Long-Horizon Planning (#2):** Sparse global rewards require early treaties to pay off 50 steps later.
3. **World Modeling & Professional Tools (#3.1):** Navigating an evolving API ecosystem (schema drift) and state tracking.
4. **Personalized Tasks (#3.2):** Handling nuanced stakeholder emails based on moods.
5. **Self-Improvement (#4):** A dynamic adversary curriculum that scales difficulty as the agent plays.

---

## 🚀 Try It Yourself

We invite you to explore Oceanus, run the baseline, and watch the trained agents coordinate to save the ocean.

* 🎮 [**Live Hugging Face Space Demo**](https://aakarsh2007-oceanus-ai.hf.space)
* 📓 [**Colab Training Notebook**](https://colab.research.google.com/drive/1lJ_EKDEZptQorExyGS6jiMMX98Qw-uOs)
* 💻 [**GitHub Repository**](https://github.com/Aakarsh2007/Oceanus-)

> *Oceanus shows that by providing rich, cross-layer environments, we can teach AI not just to play games, but to navigate the complex socio-physical challenges of the real world.*
