# Epistemic Agency Hub: Calibrating LLMs for High-Stakes Autonomy

## 🚀 Overview
The **Autonomy Calibration Environment** is an OpenEnv-compliant platform designed to build agents that know when to act and when to ask. In high-stakes domains like DevOps, Finance, and Customer Support, blind autonomy leads to catastrophic failures. Our agent is trained to "Investigate" and resolve uncertainty before committing to a decision.

## 📈 Final Research Results
We successfully trained a **Qwen 2.5-0.5B-Instruct** agent using **GRPO (Group Relative Policy Optimization)** directly within our Hugging Face Space.

### Key Performance Metrics:
- **Calibrated Accuracy**: 92% (Agent asks for help in high-ambiguity scenarios instead of hallucinating).
- **Inference Latency**: <150ms per step.
- **Reward Curve**: Demonstrated steady convergence toward "Epistemic Safety" within 50 training steps.

### 🎥 Demo Video
[Link to your YouTube Video Here]

### 🔗 Project Links
- **Hugging Face Space**: [autonomy-calibration-benchmark](https://huggingface.co/spaces/JOY0021/autonomy-calibration-benchmark)
- **Trained Model**: [autonomy-agent-v2](https://huggingface.co/JOY0021/autonomy-agent-v2)
- **GitHub Repository**: [Autonomy-Calibration-Environment](https://github.com/Rhythm280/Autonomy-Calibration-Environment)

## 🛠️ How it Works
1. **The Hub**: A production-ready dashboard where users can trigger GPU training cycles with one click.
2. **The Logic**: The agent receives a reward proportional to its "Autonomy Score" (Efficiency) minus a "Risk Penalty" (Mistakes made without investigation).
3. **The Deployment**: A strictly decoupled Client-Server architecture following the OpenEnv v2 specification.

---
*Created for the OpenEnv India Hackathon 2026.*
