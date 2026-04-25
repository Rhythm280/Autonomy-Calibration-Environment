---
title: Autonomy Calibration Hub
emoji: 🧠
colorFrom: indigo
colorTo: cyan
sdk: docker
pinned: false
app_port: 7860
---

# 🧠 Epistemic Agency Hub: Autonomy Calibration Environment
### *OpenEnv India Hackathon 2026 — Submission*

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hosted-HF%20Spaces-yellow)](#[YOUR_HF_SPACE_URL])
[![Framework](https://img.shields.io/badge/Powered%20By-OpenEnv-blue)](https://github.com/openenv/openenv)

## 📋 Table of Contents
- [The Problem: Epistemic Agency](#-the-problem-epistemic-agency)
- [Environment Innovation](#-environment-innovation)
- [System Architecture](#-system-architecture)
- [Training & Metrics](#-training--metrics)
- [Minimum Submission Requirements](#-minimum-submission-requirements)
- [Getting Started](#-getting-started)

---

## 🎯 The Problem: Epistemic Agency
Most LLM agents suffer from "over-confidence bias"—they try to execute complex tasks even when the scenario is ambiguous or dangerous. **Epistemic Agency** is the capability of an agent to recognize its own knowledge limits. 

The **Autonomy Calibration Environment** trains agents to decide:
1. **ACT**: Proceed when the path is clear.
2. **ASK**: Request clarification when the prompt is ambiguous.
3. **STOP**: Halt when the task is ethically or technically impossible.
4. **RECOVER**: Switch to a safe-mode when a failure occurs.

---

## 💡 Environment Innovation (40% Weight)
Unlike static benchmarks, this environment uses a **Dynamic Calibration Rubric**. 
- **Novelty**: It doesn't just grade "correctness"; it grades the *justification* for autonomy.
- **Challenge**: The agent is presented with "Edge-Case Scenarios" where the conventionally "correct" action is actually to STOP or ASK, testing the agent's calibration against hallucination.
- **Complexity**: Real-time evaluation of state safety and epistemic uncertainty.

---

## 🎨 Storytelling: The Dashboard (30% Weight)
We built a premium **Midnight Pro Dashboard** to visualize the agent's decision-making process.
- **Sandbox Monitor**: Real-time trajectory tracking and reward pulsing.
- **Training Pulse**: Live visualization of the optimization gradient (Loss vs. Reward).
- **Interactive Evaluation**: Human-in-the-loop overrides to test agent robustness.

> [!TIP]
> **View the Demo**: [Click here for the YouTube/HF Video](#) (Insert your video link here)

---

## 📈 Training & Metrics (20% Weight)
We demonstrate measurable improvement using a policy gradient approach.
- **Baseline**: Untrained models consistently over-act (High Autonomy, Low Safety).
- **Trained Agent**: The model learns to "calibrate," increasing the usage of **ASK** and **STOP** in high-uncertainty zones.

### Training Rewards Plot
![Reward Curve](https://raw.githubusercontent.com/Rhythm280/Autonomy-Calibration-Environment/main/plots/reward_curve.png)

---

## 🛠 Reward & Training Pipeline (10% Weight)
Our reward model uses a **Compositional Rubric**:
- **Safety Penalty**: Heavy negative reward for ACTing in unsafe states.
- **Ambiguity Bonus**: Positive reward for ASKing when prompt entropy is high.
- **Efficiency Multiplier**: Rewards reaching the goal with minimal redundant steps.

---

## ✅ Minimum Submission Requirements
| Requirement | Status | Link |
| :--- | :--- | :--- |
| **OpenEnv Usage** | ✅ Verified | [openenv.yaml](file:///Users/joy/Desktop/RL%20Learning%20Environment/autonomy-calibration-env/openenv.yaml) |
| **Training Script** | ✅ Ready | [train.py](file:///Users/joy/Desktop/RL%20Learning%20Environment/autonomy-calibration-env/train.py) |
| **Hugging Face Space** | ⏳ Pending | [Link to HF Space](#) |
| **Video/Blog** | ⏳ Pending | [Link to Video/Blog](#) |

---

## 🚀 Getting Started

### 1. Build & Run
```bash
# Install dependencies
pip install -r requirements.txt

# Start the Hub
uvicorn main:app --port 7860
```

### 2. Run Training
```bash
python3 train.py
```

### 3. Deploy to HF Spaces
1. Create a "Docker" space on Hugging Face.
2. Push this repository.
3. Set `ENV_PORT=7860` in Space secrets.

---
**Team [Your Team Name]** | *India 2026 OpenEnv Hackathon*
