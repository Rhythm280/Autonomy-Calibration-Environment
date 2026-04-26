# main.py — OpenEnv-Compliant FastAPI Server
# Autonomy Calibration Environment v1.0
# OpenEnv India Hackathon 2026 — by Rhythm

from __future__ import annotations
import os
from collections import Counter
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import subprocess
import sys
from pydantic import BaseModel

from models import Action, Observation, Reward, StepResult, ResetRequest
from environment.scenarios import SCENARIOS

def _build_registry():
    from tasks.email_triage import EmailTriageTask
    from tasks.devops_incident import DevOpsIncidentTask
    from tasks.financial_request import FinancialRequestTask
    return {
        "email_triage": EmailTriageTask,
        "devops_incident": DevOpsIncidentTask,
        "financial_request": FinancialRequestTask,
    }

TASK_REGISTRY = _build_registry()

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Autonomy Calibration Environment",
    description=(
        "OpenEnv-compliant RL environment training agents to calibrate autonomy "
        "across Email Triage, DevOps Incident Response, and Financial Request Handling."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Session State ────────────────────────────────────────────────────────────
# Single-session in-memory state. Sufficient for hackathon + HF Spaces.

_session: dict = {
    "task_name": None,
    "task": None,
    "step": 0,
    "history": [],
    "done": True,
    "seed": None,
    "episode_id": None,
}

# Global episode log for /api/history
_episode_log: list[dict] = []


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_task(name: str):
    if name not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{name}'. Valid: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name]()


# ─── API: Reset ───────────────────────────────────────────────────────────────

@app.post("/api/reset", response_model=Observation)
def reset(body: ResetRequest = ResetRequest()):
    try:
        task = _get_task(body.task)
        obs = task.reset(seed=body.seed)
        # Store seed in session and observation
        obs.seed = body.seed
        _session["task_name"] = body.task
        _session["task"] = task
        _session["step"] = 0
        _session["history"] = []
        _session["done"] = False
        _session["seed"] = body.seed
        # Create DB episode and store ID
        import database as db
        _session["episode_id"] = db.create_episode(body.task, body.seed)
        return obs
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── API: Step ───────────────────────────────────────────────────────────────

@app.post("/api/step", response_model=StepResult)
def step(action: Action):
    task = _session.get("task")
    if task is None or _session.get("done"):
        raise HTTPException(status_code=400, detail="No active episode. Call /api/reset first.")
    try:
        obs, reward, done, info = task.step(action)
        step_idx = _session["step"]
        _session["step"] += 1
        _session["done"] = done
        step_entry = {
            "step": step_idx,
            "action": action.type,
            "reward": reward.value,
            "done": done,
        }
        _session["history"].append(step_entry)
        # Persist step to SQLite
        import database as db
        db.log_step(
            episode_id=_session["episode_id"],
            step_index=step_idx,
            decision=action.type,
            reward=reward.value,
            done=done,
        )
        if done:
            episode_score = info.get("episode_score")
            db.close_episode(_session["episode_id"], episode_score or 0.0)
            _episode_log.append({
                "episode_id": _session["episode_id"],
                "task": _session["task_name"],
                "seed": _session["seed"],
                "episode_score": episode_score,
                "steps": _session["step"],
                "history": list(_session["history"]),
            })
        return StepResult(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── API: State ───────────────────────────────────────────────────────────────

@app.get("/api/state")
def state():
    task = _session.get("task")
    if task is None:
        return {"status": "not_started"}
    return {
        "status": "done" if _session["done"] else "active",
        "task": _session["task_name"],
        "step": _session["step"],
        **task.state(),
    }

# ─── API: Training ────────────────────────────────────────────────────────────

def run_training():
    """Runs the training script in a separate process and pipes output to logs."""
    try:
        print("🚀 GRPO TRAINING CORE: Initializing...")
        # Redirect stdout and stderr to the main process ones so they appear in HF Logs
        process = subprocess.Popen(
            [sys.executable, "train_rl.py"],
            stdout=sys.stdout, 
            stderr=sys.stderr,
            bufsize=1, # Line buffered
            universal_newlines=True
        )
        print(f"✅ Background process PID {process.pid} spawned.")
    except Exception as e:
        print(f"❌ Error during background training: {e}")

@app.post("/api/train")
def start_training(background_tasks: BackgroundTasks):
    # Basic check for GPU presence (useful for logs)
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
    except ImportError:
        has_gpu = False
        device_name = "CPU"
    
    background_tasks.add_task(run_training)
    
    return {
        "status": "started",
        "message": "GRPO Training started in background.",
        "using_gpu": has_gpu,
        "device": device_name
    }

@app.post("/api/upload")
def upload_to_hub(repo_id: str = "JOY0021/autonomy-agent-v2"):
    """Pushes the trained folder to the HF Hub model repo, creating it if needed."""
    try:
        import os
        from huggingface_hub import HfApi, create_repo
        
        token = os.getenv("HF_TOKEN")
        api = HfApi(token=token)
        
        # 1. Create repo if it doesn't exist
        print(f"📦 Ensuring repo {repo_id} exists...")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
        
        # 2. Upload the folder
        print(f"📡 Uploading autonomy-agent-v2 to {repo_id}...")
        api.upload_folder(
            folder_path="autonomy-agent-v2",
            repo_id=repo_id,
            repo_type="model",
        )
        return {"status": "success", "message": f"Model live at https://huggingface.co/{repo_id}"}
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        return {"status": "error", "message": str(e)}


# ─── API: Health ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    difficulty_dist = Counter(s["difficulty"] for s in SCENARIOS)
    decision_dist = Counter(s["best_decision"] for s in SCENARIOS)
    return {
        "status": "ok",
        "environment": "autonomy-calibration-env",
        "version": "2.0.0",
        "tasks": list(TASK_REGISTRY.keys()),
        "autonomy_action_space": ["ACT", "ASK", "STOP", "RECOVER"],
        "autonomy_scenarios": len(SCENARIOS),
        "autonomy_difficulty_distribution": dict(difficulty_dist),
        "autonomy_decision_distribution": dict(decision_dist),
        "reward_range": [0.01, 0.99],
    }


# ─── API: History ─────────────────────────────────────────────────────────────

@app.get("/api/history")
def history():
    total = len(_episode_log)
    scores = [e["episode_score"] for e in _episode_log if e.get("episode_score") is not None]
    return {
        "total_episodes": total,
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "episodes": _episode_log,
    }


@app.delete("/api/history")
def clear_history():
    _episode_log.clear()
    return {"status": "cleared"}


# ─── API: Observability (Step 4) ─────────────────────────────────────────────

@app.get("/api/episodes")
def episodes(limit: int = 20):
    """List recent episodes from SQLite with metadata."""
    import database as db
    try:
        rows = db.list_episodes(limit=limit)
        return {"episodes": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/replay/{episode_id}")
def replay(episode_id: int):
    """
    Return the full step history for a past episode.
    Can be used to reproduce the episode by feeding steps back into reset + step.
    """
    import database as db
    try:
        data = db.get_episode(episode_id)
        return {
            "episode_id": episode_id,
            "episode": data["episode"],
            "steps": data["steps"],
            "total_steps": len(data["steps"]),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/grade")
def grade_current():
    """Run deterministic grader on the current session's episode history."""
    task = _session.get("task")
    if task is None:
        raise HTTPException(status_code=400, detail="No active episode.")
    score = task.grade_episode(_session["history"])
    return {
        "task": _session["task_name"],
        "seed": _session["seed"],
        "episode_id": _session["episode_id"],
        "score": score,
        "steps_completed": _session["step"],
        "done": _session["done"],
    }


@app.get("/api/grade/{episode_id}")
def grade_episode(episode_id: int):
    """Run deterministic grader on a completed historical episode."""
    import database as db
    try:
        data = db.get_episode(episode_id)
        ep = data["episode"]
        steps = data["steps"]
        # Reconstruct history format expected by grade_episode()
        history = [
            {"step": s["step_index"], "action": s["decision"],
             "reward": {"value": s["reward"]}}
            for s in steps
        ]
        total_reward = sum(s["reward"] for s in steps)
        from utils import clamp
        score = clamp(total_reward)
        return {
            "episode_id": episode_id,
            "task": ep["task"],
            "seed": ep["seed"],
            "score": score,
            "total_steps": len(steps),
            "started_at": ep["started_at"],
            "ended_at": ep["ended_at"],
            "steps": steps,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ─── Static UI ────────────────────────────────────────────────────────────────

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse("static/index.html")
else:
    @app.get("/")
    def serve_fallback():
        return {
            "message": "Autonomy Calibration Environment API v1.0",
            "docs": "/docs",
            "tasks": list(TASK_REGISTRY.keys()),
        }
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
