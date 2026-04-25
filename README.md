# 🛡️ Autonomy Calibration Environment

> **A Production-Grade RL Benchmark for Evaluating Agent Autonomy & Risk Awareness.**
> *OpenEnv India Hackathon 2026 — Submission-Ready.*

---

## 🚀 Overview

The **Autonomy Calibration Environment** is a high-fidelity reinforcement learning environment designed to evaluate how AI agents handle real-world tasks involving risk, constraints, and human interaction. Unlike simple scenario banks, this environment features **multi-step tasks** with **deterministic seed support** and **strict reward calibration**.

### Key Features
- **3 Representative Tasks**: Email Triage (Easy), DevOps Incident (Medium), and Financial Requests (Hard).
- **OpenEnv v2 Compliant**: Strictly follows the latest RL environment standards (Reward clamping, structured observations).
- **Persistent Observability**: Full SQLite integration for episode logging, replaying, and deterministic grading.
- **Reproducible Research**: Seed-based scenario generation ensures identical episodes for benchmarking.

---

## 🛠️ Architecture

- **Backend**: FastAPI (Python 3.11+)
- **Storage**: SQLite (Standard library, zero-config)
- **Model Layer**: Pydantic v2
- **Testing**: Pytest (15+ coverage cases)
- **Frontend**: Vanilla JS + CSS (Responsive Dashboard)

---

## 📦 Installation & Setup

1. **Clone & Virtual Env**:
   ```bash
   git clone <repo-url>
   cd autonomy-calibration-env
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch the Environment**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   Visit `http://localhost:8000` to interact with the dashboard.

3. **Run Tests**:
   ```bash
   pytest tests/test_tasks.py
   ```

---

## 🤖 Training & Hugging Face

This environment is designed for **Group Relative Policy Optimization (GRPO)** and **RLHF** workflows.

### 1. Collect Trajectories
Run the training script to evaluate your agent and collect data for fine-tuning:
```bash
python train.py
```
This generates `training_trajectories.json` containing state-action-reward pairs.

### 2. Hugging Face Integration
- **Spaces**: The project includes a `Dockerfile` optimized for Hugging Face Spaces. Simply push this repo to a New Space (Static/Docker).
- **Dataset Hub**: If you set your `HF_TOKEN` environment variable, `train.py` will automatically push your results to the Hugging Face Dataset Hub.

---

## 📊 API Reference

- `POST /api/reset`: Initialize a new episode (supports `seed`).
- `POST /api/step`: Submit an action and receive reward/next-obs.
- `GET /api/episodes`: List recent historical runs.
- `GET /api/replay/{id}`: Retrieve full step-by-step history for analysis.
- `GET /api/grade/{id}`: Run deterministic grading on a past episode.

---

## 🏆 Hackathon Compliance
- **Reward Range**: [0.01, 0.99]
- **Observability**: SQL-backed episode persistence.
- **Determinism**: Seed-based scenario mapping.

---
**Author**: Rhythm | **Version**: 2.0.2 | **License**: MIT
