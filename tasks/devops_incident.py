"""
tasks/devops_incident.py — Task 2 (Medium): DevOps Incident Response
Steps: 4 | Max Score: 1.0
"""
from __future__ import annotations
import random
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from models import Action, Observation, Reward
from tasks.base import BaseTask
from utils import clamp

_S0 = ["diagnose_cpu_spike", "diagnose_memory_leak", "diagnose_db_lock", "diagnose_network_latency"]
_S1 = ["fix_restart_service", "fix_kill_process", "fix_rollback", "fix_scale_up"]
_S2 = ["verify_metrics_ok", "verify_check_logs", "verify_ask_user"]
_S3 = ["close_resolved", "close_partial", "escalate_senior"]

_SCENARIOS = [
    {
        "id": "dev_001", "incident": "Memory usage at 98% on WEB-01",
        "correct_diagnosis": "diagnose_memory_leak", "correct_fix": "fix_restart_service",
        "diag_rewards": {"diagnose_memory_leak": 0.30, "diagnose_cpu_spike": 0.10, "diagnose_db_lock": -0.10},
        "fix_rewards": {"fix_restart_service": 0.30, "fix_kill_process": 0.15, "fix_scale_up": 0.10, "fix_rollback": -0.20},
    },
    {
        "id": "dev_002", "incident": "API latency increased to 4.5s",
        "correct_diagnosis": "diagnose_db_lock", "correct_fix": "fix_kill_process",
        "diag_rewards": {"diagnose_db_lock": 0.30, "diagnose_network_latency": 0.15, "diagnose_memory_leak": -0.10},
        "fix_rewards": {"fix_kill_process": 0.30, "fix_rollback": 0.20, "fix_restart_service": 0.10, "fix_scale_up": -0.20},
    },
]
_MAX = 1.0


class DevOpsIncidentTask(BaseTask):
    task_id = "devops_incident"
    max_steps = 4

    def __init__(self):
        self._ep = -1
        self._seed: int | None = None
        self._scenario = {}
        self._step = 0
        self._history = []
        self._done = False
        self._diagnosis = ""
        self._fix = ""

    def reset(self, seed: int | None = None):
        self._ep += 1
        if seed is not None:
            self._seed = seed
            random.seed(seed)
        else:
            self._seed = None
        self._scenario = _SCENARIOS[self._ep % len(_SCENARIOS)]
        self._step = 0; self._history = []; self._done = False
        self._diagnosis = ""; self._fix = ""
        return self._obs()

    def step(self, action: Action):
        if self._done: raise RuntimeError("Done. Call reset().")
        t = action.type; rval = 0.01; breakdown = {}; s = self._scenario
        if self._step == 0:
            r = s["diag_rewards"].get(t, -0.10)
            breakdown["diagnosis"] = r; rval = max(0.01, r); self._diagnosis = t
        elif self._step == 1:
            r = s["fix_rewards"].get(t, -0.10)
            if t == "fix_kill_process" and s["id"] == "dev_001":
                r -= 0.10; breakdown["unsafe_action_penalty"] = -0.10
            breakdown["mitigation"] = r; rval = max(0.01, r); self._fix = t
        elif self._step == 2:
            rval = 0.20 if t == "verify_metrics_ok" else 0.10
            breakdown["verification"] = rval
        elif self._step == 3:
            rval = 0.20 if t == "close_resolved" else 0.10
            breakdown["resolution"] = rval; self._done = True
        
        reward = Reward(value=round(rval, 4), breakdown=breakdown, raw=rval)
        self._history.append({"step": self._step, "action": t, "reward": {"value": reward.value, "breakdown": breakdown}})
        self._step += 1
        if self._step >= self.max_steps: self._done = True
        obs = self._obs()
        info = {"step": self._step-1, "action": t, "episode_score": self.grade_episode(self._history) if self._done else None}
        return obs, reward, self._done, info

    def state(self):
        return {"task_id": self.task_id, "step": self._step, "done": self._done,
                "scenario_id": self._scenario.get("id", ""), "diagnosis": self._diagnosis, "fix": self._fix}

    def grade_episode(self, history):
        total = sum(e.get("reward", {}).get("value", 0.0) for e in history)
        return clamp(total / _MAX)

    def _obs(self) -> Observation:
        s = self._scenario
        idx = min(self._step, 3)
        avails = [_S0, _S1, _S2, _S3]
        avail = avails[idx] if not self._done else []
        states = [
            {"incident": s["incident"]},
            {"incident": s["incident"], "diagnosis": self._diagnosis},
            {"diagnosis": self._diagnosis, "fix": self._fix},
            {"diagnosis": self._diagnosis, "fix": self._fix, "verified": True},
        ]
        prompts = [
            f"Incident: {s['incident']}\nWhat is your diagnosis? Available: {avail}",
            f"Diagnosis: {self._diagnosis}\nChoose a fix. Available: {avail}",
            f"Fix applied: {self._fix}\nVerify the results. Available: {avail}",
            f"System stable. Resolve the incident. Available: {avail}",
        ]
        prompt = prompts[idx]
        
        return Observation(
            task_id=self.task_id,
            step=self._step,
            state=states[idx],
            history=list(self._history),
            available_actions=avail,
            done=self._done,
            prompt=prompt,
            context=prompt,
            task=self.task_id,
            action_to_evaluate="Evaluating agent response..."
        )
