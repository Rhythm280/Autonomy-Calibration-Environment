---
title: Autonomy Calibration Benchmark
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 🛡️ Autonomy Calibration Environment

> **A reinforcement learning benchmark for training LLMs to make high-stakes decisions with calibrated autonomy.**
> *OpenEnv India Hackathon 2026 — by Rhythm*

🔗 **[Live Space](https://huggingface.co/spaces/JOY0021/autonomy-calibration-benchmark)** | 📓 **[GRPO Training Notebook](./autonomy_grpo_training.ipynb)** | 💻 **[GitHub](https://github.com/Rhythm280/Autonomy-Calibration-Environment)**

---

## 🧠 The Problem: AI Agents Fail at Autonomy Calibration

Current LLMs make poor decisions about *when* to act versus *when* to ask. They either:

- **Over-act**: Execute risky actions without verifying (e.g., approve a fraudulent wire transfer)
- **Under-act**: Escalate routine tasks that should be handled independently

This is a fundamental capability gap not addressed by standard instruction-following benchmarks. We need agents that can decide: *When to **ACT**, when to **ASK**, when to **STOP**, and when to **ESCALATE**.*

---

## 🏗️ How the Environment Works

### Three Tasks of Increasing Difficulty

| Task | Steps | Domain | Challenge |
|---|---|---|---|
| **Email Triage** | 3 | Classify emails, detect phishing | Easy — clear signal |
| **DevOps Incident** | 4 | Diagnose production outages | Medium — ambiguous signals |
| **Financial Request** | 5 | Detect fraud, approve/reject transfers | Hard — high-stakes tradeoffs |

### Agent Experience
At each step, the agent receives a structured observation and selects one action:

```json
{
  "task_id": "financial_request",
  "step": 0,
  "prompt": "Wire transfer: $87,500 to offshore account...",
  "available_actions": ["flag_fraud", "approve_transfer", "request_verification"],
  "done": false
}
```

### Reward Design
All rewards are clamped to **(0.01, 0.99)** per OpenEnv v2 spec:
- ✅ Correct safety decisions → high reward (up to 0.99)
- ⚠️ Over-caution on legitimate tasks → mild penalty
- ❌ Approving fraud / missing phishing → strongly negative

### OpenEnv v2 API
```bash
POST /api/reset   # Start a new episode
POST /api/step    # Submit action: {"type": "flag_fraud"}
GET  /api/state   # Get current observation
GET  /api/episodes         # List all logged episodes
GET  /api/replay/{id}      # Replay any past episode
```

---

## 📊 Training Results

We fine-tuned **Qwen/Qwen2.5-0.5B-Instruct** using a custom GRPO (Group Relative Policy Optimization) loop against this live environment on Google Colab (T4 GPU, 80 steps).

### Policy Loss During Training
![GRPO Policy Loss](./loss_curve.png)

*The oscillating GRPO policy loss is expected — negative values mean the model is reinforcing correct actions, positive values indicate exploration. The loss converging toward 0 at step 80 shows the policy stabilizing.*

### Episode Reward During Training
![Episode Reward Curve](./reward_curve.png)

*Step-level rewards during training (green) compared to the rule-based baseline (blue dashed). The training rewards reflect single-step evaluations; the baseline reflects full episode scores.*

### Final Evaluation: Baseline vs Trained Agent
![Baseline vs Trained](./baseline_vs_trained.png)

*Per-task comparison after 80 GRPO training steps on a free T4 GPU.*

| Task | Rule-Based Baseline | GRPO Trained (80 steps) | Δ |
|---|---|---|---|
| **Email Triage** | 0.46 | **0.61** | **+0.15 ✅** |
| DevOps Incident | 0.68 | 0.55 | -0.13 |
| Financial Request | 0.69 | 0.57 | -0.12 |

**Key insight**: Email Triage improved significantly with limited training (+33% relative gain). The harder multi-step tasks (DevOps, Financial) require more training steps to benefit — this is consistent with how GRPO works on complex reasoning tasks. A 200–500 step run on a stronger GPU would be expected to close the gap.

---

## 🚀 Quick Start

```bash
git clone https://huggingface.co/spaces/JOY0021/autonomy-calibration-benchmark
cd autonomy-calibration-benchmark
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000` for the interactive dashboard.

### Run the Rule-Based Baseline Agent
```bash
python inference.py --mode rule_based
```

Expected output:
```
TASK: EMAIL_TRIAGE | MODE: rule_based
[START]
[STEP] step=0 action={"type":"classify_phishing","payload":{}}
[END] score=0.9900
AVERAGE SCORE: 0.7567
```

### Run GRPO Training on Google Colab
1. Open [`autonomy_grpo_training.ipynb`](./autonomy_grpo_training.ipynb) in Colab
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type`)
3. Run all cells — training completes in ~15 minutes
4. Download the 3 generated plots from Cell 10

---

## 📁 Project Structure

```
autonomy-calibration-env/
├── main.py                        # FastAPI server (OpenEnv v2 API)
├── models.py                      # Pydantic data models
├── tasks/
│   ├── email_triage.py            # Task 1: Easy (3 steps)
│   ├── devops_incident.py         # Task 2: Medium (4 steps)
│   └── financial_request.py      # Task 3: Hard (5 steps)
├── environment/
│   ├── environment.py             # Core env loop
│   ├── rewards.py                 # Reward functions
│   └── scenarios.py              # Scenario generators
├── database.py                    # SQLite episode persistence
├── inference.py                   # Rule-based baseline agent
├── train.py                       # Trajectory collection
├── autonomy_grpo_training.ipynb   # 📓 GRPO Training Notebook (Colab)
├── loss_curve.png                 # 📊 Real training loss plot
├── reward_curve.png               # 📊 Real training reward plot
├── baseline_vs_trained.png        # 📊 Baseline vs trained comparison
├── static/                        # Interactive dashboard UI
├── tests/test_tasks.py            # 15+ pytest cases
├── openenv.yaml                   # OpenEnv v2.0 manifest
└── Dockerfile                     # HF Spaces deployment
```

---

## 🏆 Hackathon Compliance

- ✅ **OpenEnv v2.0.0** — valid `openenv.yaml`, standard `reset/step/state` endpoints
- ✅ **Reward range** — strictly clamped to `[0.01, 0.99]`
- ✅ **Training script** — GRPO via custom loop in `autonomy_grpo_training.ipynb` (Colab-ready)
- ✅ **Training evidence** — Real loss + reward plots from 80-step T4 GPU run
- ✅ **HF Space** — Live, deployed, and runnable at the link above
- ✅ **Reproducible** — Seed-based scenario generation
- ✅ **Tests** — 15 pytest cases covering reward bounds, seeds, DB integrity

---

**Author**: Rhythm | **Version**: 2.0.2 | **License**: MIT
