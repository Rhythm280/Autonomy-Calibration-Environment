---
title: Autonomy Calibration Hub
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Epistemic Agency Hub: Autonomy Calibration Environment
### OpenEnv India Hackathon 2026 Submission

## Project Overview
The Epistemic Agency Hub is a specialized reinforcement learning benchmark designed to evaluate an agent's ability to manage uncertainty through calibrated autonomy. Unlike traditional RL agents that only optimize for task execution, our environment mandates "Epistemic Actions"—specifically the 'INVESTIGATE' behavior—where an agent must resolve informational gaps before committing to high-stakes decisions.

## Core Framework: Investigate-then-Act
The environment implements a calibration-first workflow:
1. **Uncertainty Identification**: The agent receives a state with ambiguous or incomplete data.
2. **Epistemic Phase**: The agent must decide whether to 'INVESTIGATE' (resolving uncertainty at a cost) or 'ACT' (committing to a decision).
3. **Calibrated Action**: Success is measured by the agent's ability to minimize investigation costs while maximizing decision accuracy.

## Technical Implementation

### Environment and Agent Behavior
*   **OpenEnv Compliance**: Fully compliant with the latest OpenEnv API specifications.
*   **Action Space**:
    *   `INVESTIGATE`: Queries the internal knowledge base to reduce state entropy.
    *   `ACT`: Executes the final decision based on current belief state.
    *   `RECOVER`: Error-handling mechanism for miscalibrated decisions.
*   **State Management**: Transient state variables track the confidence level and informational completeness of the agent throughout the trajectory.

### Reward Model and Training
We utilize a Group Relative Policy Optimization (GRPO) approach to calibrate the agent's decision-making logic:
*   **Causal Merit Reward**: Rewards are distributed for successful investigation steps that directly lead to higher action accuracy.
*   **Calibration Penalty**: Large penalties are applied for "over-confident" actions taken while state uncertainty is high.
*   **Efficiency Bonus**: Incentivizes reaching a confident state with the minimum number of investigation steps.

### Training Pipeline
The training pipeline is designed for scalability and observability:
*   **Real-Time Monitoring**: Integrated with the Midnight Pro Dashboard for live tracking of loss gradients and reward pulses.
*   **Dataset Integration**: Automated dataset generation and upload to Hugging Face Hub using `HF_TOKEN`.

## Performance Analytics

### Reward Pulse and Loss Gradient
The following metrics demonstrate the agent's convergence during the calibration phase:

![Reward Curve](https://raw.githubusercontent.com/Rhythm280/Autonomy-Calibration-Environment/main/reward_curve.png)
*Figure 1: Mean expected reward across training epochs, showing a steady increase as the agent learns to prioritize investigation over premature action.*

![Loss Curve](https://raw.githubusercontent.com/Rhythm280/Autonomy-Calibration-Environment/main/loss_curve.png)
*Figure 2: GRPO loss gradient stabilization, indicating robust policy convergence.*

### Comparative Behavior Analysis
![Inference Baseline](https://raw.githubusercontent.com/Rhythm280/Autonomy-Calibration-Environment/main/baseline_vs_trained.png)
*Figure 3: Behavior comparison between a standard greedy agent and the calibrated Epistemic Agent.*

## Hackathon Submission Details
*   **Environment Host**: [Hugging Face Space](https://huggingface.co/spaces/JOY0021/autonomy-calibration-benchmark)
*   **Training Script**: `train.py` (Implements Unsloth/HF TRL pipeline).
*   **Mini-Blog/Video**: [Pending Link]

## Deployment and Setup
The environment is containerized using Docker for seamless deployment on Hugging Face Spaces.

### Local Development
To run the dashboard locally:
```bash
pip install -r requirements.txt
uvicorn main:app --port 7860
```

### Building for Production
```bash
docker build -t autonomy-calibration-hub .
docker run -p 7860:7860 autonomy-calibration-hub
```

## License
MIT License - OpenEnv India 2026.
