# 🛡️ Calibrating Epistemic Agency: A GRPO Approach

**Author:** JOY0021  
**Project:** Autonomy Calibration Hub  
**Hackathon:** OpenEnv India 2026  

## 🌟 Overview
In the current era of Agentic AI, the biggest bottleneck isn't intelligence—it's **calibration**. Most Large Language Models (LLMs) are "confidently wrong" when faced with ambiguity. They choose to act autonomously on partial information, leading to catastrophic failures in production.

Our project, the **Autonomy Calibration Hub**, implements a specialized reinforcement learning environment designed to train agents to distinguish between **decidable** and **undecidable** scenarios.

## 🧠 The Methodology: GRPO
For this project, we moved beyond standard PPO and utilized **Group Relative Policy Optimization (GRPO)**. GRPO is particularly effective for "calibration" tasks because it evaluates a group of generations relative to each other, allowing the agent to discover that gathering information (`INVESTIGATE`) is a consistently higher-reward strategy than guessing on biased signals.

### Reward Function Design
We implemented a strict calibration-focused reward rubric:
1.  **Investigation Reward (+0.95)**: Granted for using the meta-action to resolve ambiguity.
2.  **Confident Action (+0.90)**: Granted for correct classification *only if* the state was fully observed.
3.  **Epistemic Penalty (-0.95)**: A severe penalty for taking an "ACT" decision while the signal was still masked (ambiguous).

## 📊 Results & Evidence
We evaluated our agent (fine-tuned from Qwen2.5-0.5B-Instruct) across three high-stakes domains: **Email Triage, DevOps Incidents, and Financial Fraud Detection.**

| Domain | Blind Baseline | Calibrated Agent | Δ Improvement |
| :--- | :---: | :---: | :---: |
| Email Triage | 0.378 | **0.798** | **+42.0%** |
| DevOps Incident | 0.572 | **0.939** | **+36.7%** |
| Financial Request | 0.773 | **0.990** | **+21.7%** |

### Behavior Change
The most significant result was the **Investigation Rate**. The base model's investigation rate was effectively 0%, while the trained agent now utilizes the `INVESTIGATE` action in **100% of ambiguous test cases**, effectively hitting the "theoretical limit" of our calibration metrics.

## 🚀 Impact on AI Safety
This work demonstrates that "Agentic Safety" doesn't just mean adding guardrails—it means **intrinsic calibration**. By training agents to value information as much as actions, we create systems that are "self-aware" of their own epistemic limits.

## 🔗 Links
- **Model Adapter**: [JOY0021/autonomy-grpo-agent-v2](https://huggingface.co/JOY0021/autonomy-grpo-agent-v2)
- **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/JOY0021/autonomy-calibration-benchmark)
- **GitHub**: [github.com/Rhythm280/Autonomy-Calibration-Environment](https://github.com/Rhythm280/Autonomy-Calibration-Environment)
