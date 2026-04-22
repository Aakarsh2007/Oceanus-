# 🌊 Oceanus: Multi-Layer Ghost-Gear Recovery & Treaty Arena

## 1. Project Description

**Oceanus** is a groundbreaking, multi-layered RL environment built on **OpenEnv**. It is designed to train LLM agents in both **physical spatial reasoning** and **complex semantic negotiation** by simulating the real-world crisis of abandoned fishing gear (Ghost Nets) across transboundary ocean sectors. 

Unlike standard RL environments that just move continuous points on a grid, Oceanus forces LLMs to utilize their native strengths: reasoning, tool-use (APIs), text-based negotiation, and handling personalized conflicts. 

The environment operates on three simultaneous layers:
1.  **The Physical Swarm Layer:** 4 LLM-powered Autonomous Surface Vehicles (ASVs) coordinate on a 2D ocean grid. They cannot see the whole map (Partial Observability). They must output structured JSON to query Sonar APIs, move, and broadcast text messages to other drones to form search patterns.
2.  **The Policy & Stakeholder Layer:** 2 LLM agents act as a "Port Authority" and a "Fishing Fleet Manager". They operate entirely in text, handling a queue of personalized emails (e.g., angry coastal residents, fishers complaining about costs). They must negotiate a gear-tagging mandate. If they succeed, the rate at which new ghost nets spawn in the physical layer decreases.
3.  **The Adversary Curriculum Layer:** A single "Climate/Chaos" agent dynamically injects schema drift (e.g., sudden storms that drift nets, or sudden changes in policy compliance) to force the other agents to adapt via recursive self-improvement.

---

## 2. Why This Dominates the 1000-Team Competition

This project is surgically engineered to hit the maximum possible points on the Meta x Scaler judging rubric.

### 🏆 Hitting the Judging Criteria
*   **Environment Innovation (40%):** 99% of teams will build either a basic text-adventure or a simple physical grid where LLMs output raw (x,y) coordinates (which standard PPO does better anyway). Oceanus is a *multi-modal world model*. A semantic treaty negotiated via email directly alters the physical spawn mechanics of a 2D grid. This is entirely novel.
*   **Storytelling (30%):** The visual demo will be a stunning split-screen UI. Left side: A live 2D heatmap showing ghost nets drifting and drones sweeping. Right side: A live terminal feed of LLMs negotiating policies and replying to angry emails. The narrative goes from "untrained ocean collapse" to "trained ecological recovery."
*   **Improvement in Rewards (20%):** The reward curve will be undeniably clear. Baseline models will wander blindly and fail negotiations (crashing the ecosystem). GRPO-trained models will learn to broadcast coordinates and compromise on treaties, resulting in an 80%+ ecosystem recovery score.
*   **Pipeline Setup (10%):** Fully integrated with OpenEnv standards, Dockerized, and trained using Hugging Face TRL and Unsloth in a Colab notebook.

### 🎯 Theme Alignment
*   **Theme #1 (Multi-Agent):** Cooperation (ASV Swarms) and Negotiation (Policy Agents).
*   **Theme #2 (Long-Horizon):** Sparse rewards. Ghost nets drift for dozens of steps. A treaty signed at step 10 only pays off at step 50.
*   **Theme #3.1 (Professional Tools - Scaler AI Bonus):** ASVs query explicit `get_sonar_data` and `broadcast_alert` APIs.
*   **Theme #3.2 (Personalized Tasks - Patronus Bonus):** Policymakers must reply to highly personalized, emotionally charged stakeholder emails while navigating schema drift (e.g., sudden policy invalidations).
*   **Theme #4 (Self-Improvement - Snorkel Bonus):** The Adversary agent creates a dynamic curriculum, constantly generating new edge cases to test the primary agents.

---

## 3. The 5-Day Implementation Plan

With a 5-day runway, we will break the work into strictly isolated modules to ensure a flawless, bug-free demo.

### Day 1: Core Simulation & OpenEnv Scaffolding
**Goal:** Build the underlying physics engine and the OpenEnv wrapper. No LLMs yet.
*   **Task 1.1: 2D Physics Engine.** 
    *   Create a 20x20 NumPy grid representing the ocean.
    *   Implement drift mechanics (nets move 1 cell randomly every N steps based on a global "wind" vector).
    *   Implement ASV movement logic and localized sonar (reveals a 3x3 grid around the ASV).
*   **Task 1.2: State Management.** 
    *   Create the `GameState` class that holds the 2D grid, the ASV positions, and the `Inbox` queues for the Policy agents.
*   **Task 1.3: OpenEnv Wrapper (`models.py`).** 
    *   Implement `reset()`, `step()`, and define the `ObservationSpace` (JSON strings) and `ActionSpace` (Text output).

### Day 2: Multi-Agent LLM Interfaces & JSON Schemas
**Goal:** Connect LLMs to the environment by forcing strict JSON inputs/outputs.
*   **Task 2.1: Observation Prompts.** 
    *   Write the system prompts that translate the NumPy grid into text for the ASVs (e.g., *"You are ASV-1. Sonar detects nets at relative coords [0, 1]. Inbox: 'ASV-2 found nets at C4'."*).
*   **Task 2.2: Action Parsers.** 
    *   Create Regex/JSON parsers in the `step()` function to extract actions.
    *   *ASV Schema:* `{"intent": "move|scan|clean|broadcast", "target": "coords", "message": "string"}`
    *   *Policy Schema:* `{"intent": "reply_email|propose_treaty|accept_treaty", "target": "recipient", "content": "string"}`
*   **Task 2.3: The Adversary Agent.** 
    *   Implement a lightweight script that injects "Chaos Events" into the observation space every 20 steps (e.g., *"Weather updated: Storm surge in Sector B."*).

### Day 3: The RL Pipeline (TRL + GRPO + Unsloth)
**Goal:** Prove the agents can learn. Build the training loop in Google Colab.
*   **Task 3.1: Reward Engineering (The Secret Sauce).** 
    *   *ASV Local Reward:* +5 for cleaning a net, -1 for invalid JSON, -0.1 for moving (energy cost).
    *   *Policy Local Reward:* +10 for a successfully parsed treaty, -5 for ignoring an email > 5 steps.
    *   *Global Horizon Reward:* +100 at the end of the episode if Biodiversity Index > 75%.
*   **Task 3.2: GRPO Integration.** 
    *   Set up a Hugging Face TRL script. Use `GRPOTrainer`.
    *   Use a fast, capable model like `Qwen-1.5-0.5B` or `Llama-3-8B-Instruct` (Unsloth quantized for speed).
*   **Task 3.3: Baseline vs. Trained Rollouts.** 
    *   Run the untrained model (Baseline) and save the logs.
    *   Train for 500-1000 steps. Save the logs. Graph the massive difference in rewards.

### Day 4: Storytelling Dashboard & UI
**Goal:** Build the visual wow-factor for the judges.
*   **Task 4.1: Streamlit / FastAPI Dashboard.** 
    *   Create a local web app.
    *   **Left Panel:** A matplotlib/plotly updating heatmap of the 20x20 grid showing ASVs hunting nets.
    *   **Right Panel:** A scrolling chat-like interface showing the Policy Agents negotiating and the ASVs broadcasting messages.
*   **Task 4.2: Game Loop API.** 
    *   Expose a FastAPI endpoint `/step` that the UI polls to update the visuals in real-time.

### Day 5: Polish, Pitch, & OpenEnv Hub Submission
**Goal:** Finalize deliverables and practice the 3-minute pitch.
*   **Task 5.1: Dockerization.** 
    *   Ensure the OpenEnv environment builds flawlessly in the provided Dockerfile.
*   **Task 5.2: Hugging Face Submission.** 
    *   Push the environment code to the HF OpenEnv collection.
    *   Deploy the Streamlit UI as a Hugging Face Space for judges to click and play.
*   **Task 5.3: Pitch Script & Video.** 
    *   Record a <2 min Loom video showing the "Before & After" training curves and the visual dashboard.
    *   *Pitch Hook:* "While others trained agents to play 2D games, we built Oceanus: the first multi-modal arena where LLM policy negotiations directly control the physics of ocean restoration."

---

## 4. Technical Specifications & Details

### Observation Spaces (What the LLM sees)
To avoid context limit blowouts, observations are localized.
**Example ASV Observation:**
```json
{
  "agent_id": "ASV-1",
  "battery": 85,
  "sonar_3x3": [["empty", "net", "empty"], ["empty", "self", "empty"], ["empty", "empty", "empty"]],
  "comms_inbox": ["ASV-3: Found heavy nets in North-East quadrant."]
}
```

**Example Policy Agent Observation:**
```json
{
  "agent_id": "Port_Authority",
  "current_policy_status": "No Tagging Mandate",
  "inbox": [
    {"from": "Fisher_Bob", "mood": "Angry", "message": "I can't afford these tracker tags!"},
    {"from": "Fleet_Manager_Agent", "message": "Proposing a 50% subsidy for tags. Do you accept?"}
  ],
  "net_spawn_rate": "High"
}
```

### Reward Function Equations
The system uses a blended CTDE (Centralized Training, Decentralized Execution) reward structure.
*   $R_{total} = \alpha \sum R_{local\_asv} + \beta \sum R_{local\_policy} + \gamma R_{global\_ecosystem}$
*   Where $R_{global\_ecosystem}$ is only awarded at $Step_{max}$, forcing agents to prioritize long-horizon planning over greedy short-term actions.
