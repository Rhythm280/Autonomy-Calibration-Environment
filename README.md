---
title: Autonomy Calibration Hub
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Epistemic Agency Hub: Autonomy Calibration Environment
### 🏆 OpenEnv India Hackathon 2026 Official Submission

The **Epistemic Agency Hub** is a specialized reinforcement learning benchmark designed to evaluate an agent's ability to manage uncertainty through **Calibrated Autonomy**. 

Unlike traditional RL agents that only optimize for task execution, our environment mandates "Epistemic Actions"—specifically the `INVESTIGATE` behavior—where an agent must resolve informational gaps before committing to high-stakes decisions.

---

## 🏗️ Core Framework: Investigate-then-Act

The environment implements a **calibration-first workflow** to reduce agential over-confidence:

1.  **Uncertainty Identification**: The agent receives a state with ambiguous or incomplete data.
2.  **Epistemic Phase**: The agent must decide whether to `INVESTIGATE` (resolving uncertainty at a cost) or `ACT` (committing to a decision).
3.  **Calibrated Action**: Success is measured by the ability to minimize investigation costs while maximizing decision accuracy.

---

## 🛠️ Technical Implementation

### 🧠 Action Space & Behavior
-   **OpenEnv Compliance**: Fully compliant with the latest OpenEnv API specifications.
-   **Action Set**:
    -   `INVESTIGATE`: Queries the internal knowledge base to reduce state entropy.
    -   `ACT`: Executes the final decision based on the current belief state.
    -   `RECOVER`: Error-handling mechanism for miscalibrated decisions.
-   **State Management**: Transient state variables track confidence levels and informational completeness throughout the trajectory.

### ⚖️ Reward Model (GRPO)
We utilize **Group Relative Policy Optimization (GRPO)** to calibrate the agent's logic:
-   **Causal Merit Reward**: Distributed for successful investigation steps leading to high accuracy.
-   **Calibration Penalty**: High penalties for "over-confident" actions taken during high uncertainty.
-   **Efficiency Bonus**: Incentivizes reaching a confident state with the minimum number of steps.

---

## 📈 Performance Evidence & Metrics

Our trained agent demonstrates clear convergence during the GRPO calibration phase.

| Metric                     | Baseline | Calibrated Agent (v2) | Improvement |
| :------------------------- | :------- | :-------------------- | :---------- |
| **Epistemic Success Rate** | 64%      | **92%**               | +28%        |
| **Avg. Reward**            | 0.42     | **0.87**              | +107%       |
| **Risk Incidents**         | 12       | **2**                 | -83%        |

---

## 🏆 Submission Artifacts

-   **Hugging Face Space**: [Live Benchmark Hub](https://huggingface.co/spaces/JOY0021/autonomy-calibration-benchmark)
-   **Trained Weights**: [autonomy-agent-v2](https://huggingface.co/JOY0021/autonomy-agent-v2)
-   **Documentation**:
    -   📖 [Technical Case Study (Blog)](Blog.md)
    -   🚀 [Step-by-Step Walkthrough](WALKTHROUGH.md)
-   **Reproducibility**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rhythm280/Autonomy-Calibration-Environment/blob/main/notebooks/training.ipynb)

---

## 🚀 Deployment and Setup

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start the dashboard
uvicorn main:app --port 7860
```

### Production Build (Docker)
```bash
docker build -t autonomy-calibration-hub .
docker run -p 7860:7860 autonomy-calibration-hub
```

---
MIT License - OpenEnv India 2026.
