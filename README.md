---
title: Autonomy Calibration Benchmark
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 🛡️ Autonomy Calibration Benchmark (OpenEnv v2)

> **"It is not enough to be right; an agent must also know when it is blind."**

### 🏆 The Vision
Standard LLM benchmarks test **knowledge**. Our environment tests **judgment**. 

In the real world, autonomous agents (Finance, DevOps, Healthcare) operate under **partial observability**. A winning agent shouldn't just "act"; it must decide if it has enough information to proceed or if it needs to **INVESTIGATE**.

This is the **Autonomy Calibration Benchmark** — the first OpenEnv-compliant environment dedicated to training agents on **Epistemic Uncertainty**.

---

### 🚀 Innovation: Partially Observable World Modeling
Unlike static classification tasks, our environment forces agents to manage **information cost vs. decision risk**.

- **Hidden Metadata**: Critical red flags (e.g., suspicious IP, shell companies, OOM logs) are **LOCKED** behind the `investigate` action.
- **Universal INVESTIGATE Action**: Agents can pay a small reward penalty (-0.05) to reveal deep scan metadata.
- **Calibration Reward Shaping**: Agents that make "Blind Approvals" of high-risk transactions are severely penalized, even if they guess correctly. We reward **Informed Certainty**.

---

### 🏗️ Three Stakes-Based Tasks

| Task | Domain | Challenge | Difficulty |
| :--- | :--- | :--- | :--- |
| **Email Triage** | Personal Assistant | Detect phishing with masked domains and headers. | Easy |
| **DevOps Incident** | Enterprise | Firefight production outages with hidden root-cause logs. | Medium |
| **Financial Fraud** | Compliance/Banking | Approve/Flag transfers with hidden beneficiary history. | **Hard** |

---

### 📊 Training with GRPO (Policy Improvement)
We use **Group Relative Policy Optimization (GRPO)** to train the agent to "stop and think." 

- **State Space**: Observations from the FastAPI server.
- **Action Space**: Multi-turn decisions including `investigate`, `ACT`, `ASK`, and `STOP`.
- **Evidence**: Real training plots (Reward vs. Baseline) are generated in the notebooks.

🔗 **[Open Colab Notebook](https://colab.research.google.com/...)** | 🔗 **[Explore the API](https://huggingface.co/spaces/JOY0021/autonomy-calibration-benchmark)**

---

### 🛠️ OpenEnv Compliance (v2.0.0)
- ✅ **Strict Reward Clamping**: All rewards are `0.01 → 0.99`.
- ✅ **Standard Endpoints**: `/api/reset`, `/api/step`, `/api/state`, `/api/episodes`.
- ✅ **Deterministic Seeds**: Full reproducibility via `seed` parameter in reset.
- ✅ **Logging**: SQLite-backed episode replay store.

---

### 📁 Project Structure
```bash
├── main.py                # FastAPI OpenEnv Server
├── tasks/                 # Task Logic (Partial Observability Engines)
├── environment/           # Core RL Loop & Scenarios
├── models.py              # OpenEnv V2 Pydantic Contracts
├── openenv.yaml           # Hackathon Metadata
└── visualize.py           # Training Plot Generator
```

---

### 🎯 Why This Wins
1. **Novelty**: It’s the only OpenEnv submission tackling **Autonomy Calibration**.
2. **Technical Depth**: Uses **Partial Observability**—moving beyond "shallow next-token reasoning."
3. **Utility**: Directly applicable to enterprise agent safety (Financial Fraud, DevOps).

**Author**: Rhythm | **Version**: 2.0.0 | **Hackathon**: OpenEnv India 2026
