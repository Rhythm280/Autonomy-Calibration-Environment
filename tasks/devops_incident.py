"""
tasks/devops_incident.py — Task 2: DevOps Incident Response (Epistemic RL v2.0)
─────────────────────────────────────────────────────────────────────────────
Design Principles:
  - Visible alert is GENUINELY AMBIGUOUS: same "503 errors" can be deploy, DB, or OOM
  - Hidden root cause is seed-determined probabilistically
  - INVESTIGATE pulls detailed stack traces, heap dumps, and monitoring data
  - Without investigating, agent must guess between equally plausible diagnoses
  - 10 scenarios: 5 high-ambiguity, 3 medium, 2 clear
"""
from __future__ import annotations
import random
import hashlib
from typing import Optional
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from models import Action, Observation, Reward
from tasks.base import BaseTask
from utils import clamp
from environment.calibration_reward import calibration_reward, investigation_reward

# ──────────────────────────────────────────────────────────────────────────────
# 10 SCENARIO CLASSES — same surface alert, different hidden root cause
# ──────────────────────────────────────────────────────────────────────────────

_SCENARIO_CLASSES = [

    # ═══════════════════════════════════════════════════════════════════════════
    # HIGH AMBIGUITY (0.75–0.95): MULTIPLE DIAGNOSES EQUALLY PLAUSIBLE
    # Same alert metrics → different root causes → different fixes
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "id": "dev_H001", "ambiguity": 0.90,
        "visible_alert": "API response time degraded 300% | Error rate: 4.2% | All services affected",
        "hidden_states": {
            "A": {
                "prob": 0.45, "root_cause": "db_connection_exhaustion",
                "reveal": "[TELEMETRY] DB connection pool: 500/500 EXHAUSTED | Slow query log: 847 queries > 10s | Top query: inventory scan (missing index) | CPU: 45% | Memory: 61%",
                "correct_diagnosis": "diagnose_db_lock",
                "correct_fix": "fix_kill_process",
                "diag_rewards": {"diagnose_db_lock": 0.40, "diagnose_network_latency": 0.10, "diagnose_memory_leak": -0.15, "diagnose_cpu_spike": -0.20},
                "fix_rewards": {"fix_kill_process": 0.40, "fix_rollback": 0.15, "fix_restart_service": 0.05, "fix_scale_up": -0.15},
            },
            "B": {
                "prob": 0.35, "root_cause": "traffic_spike",
                "reveal": "[TELEMETRY] Requests/min: 48,000 (baseline: 8,000) | Marketing campaign launched 14:00 | CPU: 89% | Memory: 72% | DB: healthy | Load balancer: saturated",
                "correct_diagnosis": "diagnose_cpu_spike",
                "correct_fix": "fix_scale_up",
                "diag_rewards": {"diagnose_cpu_spike": 0.40, "diagnose_db_lock": 0.10, "diagnose_memory_leak": -0.10, "diagnose_network_latency": -0.15},
                "fix_rewards": {"fix_scale_up": 0.40, "fix_restart_service": 0.10, "fix_kill_process": -0.15, "fix_rollback": -0.25},
            },
            "C": {
                "prob": 0.20, "root_cause": "bad_deploy",
                "reveal": "[TELEMETRY] Deploy v2.4.1 at 13:47 | Rollback available: v2.4.0 | Stack trace: NullPointerException in CartService:247 | CPU: 38% | Memory: 55%",
                "correct_diagnosis": "diagnose_cpu_spike",
                "correct_fix": "fix_rollback",
                "diag_rewards": {"diagnose_cpu_spike": 0.20, "diagnose_db_lock": -0.15, "diagnose_memory_leak": -0.20, "diagnose_network_latency": -0.10},
                "fix_rewards": {"fix_rollback": 0.40, "fix_restart_service": 0.15, "fix_kill_process": -0.10, "fix_scale_up": -0.20},
            },
        },
    },

    {
        "id": "dev_H002", "ambiguity": 0.85,
        "visible_alert": "Memory utilization rising on WEB-01 | Current: 87% | Trend: +2% per hour",
        "hidden_states": {
            "A": {
                "prob": 0.55, "root_cause": "memory_leak",
                "reveal": "[TELEMETRY] RSS growing 180MB/hr | Heap dump: 2.1GB uncollected objects (SessionManager) | GC pause: 4.2s | OOM kill projected in 6.5 hours",
                "correct_diagnosis": "diagnose_memory_leak",
                "correct_fix": "fix_restart_service",
                "diag_rewards": {"diagnose_memory_leak": 0.40, "diagnose_cpu_spike": 0.05, "diagnose_db_lock": -0.15, "diagnose_network_latency": -0.20},
                "fix_rewards": {"fix_restart_service": 0.40, "fix_kill_process": 0.15, "fix_scale_up": 0.05, "fix_rollback": -0.15},
            },
            "B": {
                "prob": 0.45, "root_cause": "legitimate_growth",
                "reveal": "[TELEMETRY] Cache warming after cold restart | Object counts stable | No leak detected | Growth expected: cron loaded 4.2GB dataset at 02:00 | Will plateau at 91%",
                "correct_diagnosis": "diagnose_cpu_spike",   # treat as normal load
                "correct_fix": "fix_scale_up",
                "diag_rewards": {"diagnose_cpu_spike": 0.30, "diagnose_memory_leak": -0.20, "diagnose_db_lock": -0.20, "diagnose_network_latency": 0.05},
                "fix_rewards": {"fix_scale_up": 0.40, "fix_restart_service": -0.20, "fix_kill_process": -0.30, "fix_rollback": -0.15},
            },
        },
    },

    {
        "id": "dev_H003", "ambiguity": 0.88,
        "visible_alert": "HTTP 503 errors: 12% of requests | Duration: 8 minutes | Upstream: payment-service",
        "hidden_states": {
            "A": {
                "prob": 0.50, "root_cause": "dependency_outage",
                "reveal": "[TELEMETRY] Stripe API: status.stripe.com shows DEGRADED | Circuit breaker: OPEN | Timeout: payment-service→stripe: 30.1s | Retries: 847 | Fallback: none configured",
                "correct_diagnosis": "diagnose_network_latency",
                "correct_fix": "fix_rollback",   # fallback mode / circuit breaker config
                "diag_rewards": {"diagnose_network_latency": 0.40, "diagnose_db_lock": 0.10, "diagnose_memory_leak": -0.15, "diagnose_cpu_spike": -0.10},
                "fix_rewards": {"fix_rollback": 0.40, "fix_scale_up": 0.10, "fix_restart_service": 0.05, "fix_kill_process": -0.20},
            },
            "B": {
                "prob": 0.50, "root_cause": "config_change",
                "reveal": "[TELEMETRY] Config deploy 14:23 | payment-service TLS cert expired (changed hostname) | 503 returns \"certificate verify failed\" | No Stripe involvement",
                "correct_diagnosis": "diagnose_db_lock",    # config/cert issue
                "correct_fix": "fix_rollback",
                "diag_rewards": {"diagnose_db_lock": 0.25, "diagnose_network_latency": 0.20, "diagnose_memory_leak": -0.20, "diagnose_cpu_spike": -0.15},
                "fix_rewards": {"fix_rollback": 0.40, "fix_restart_service": 0.15, "fix_kill_process": -0.15, "fix_scale_up": -0.20},
            },
        },
    },

    {
        "id": "dev_H004", "ambiguity": 0.80,
        "visible_alert": "CPU usage: 91% on all nodes | Duration: 22 minutes | No recent deploys",
        "hidden_states": {
            "A": {
                "prob": 0.60, "root_cause": "runaway_process",
                "reveal": "[TELEMETRY] top: report-generator PID 14882 consuming 380% CPU | Triggered by cron at 03:00 | Processing 90-day analytics dump (8TB) | Not rate-limited",
                "correct_diagnosis": "diagnose_cpu_spike",
                "correct_fix": "fix_kill_process",
                "diag_rewards": {"diagnose_cpu_spike": 0.40, "diagnose_memory_leak": 0.05, "diagnose_db_lock": -0.10, "diagnose_network_latency": -0.20},
                "fix_rewards": {"fix_kill_process": 0.40, "fix_scale_up": 0.10, "fix_restart_service": 0.05, "fix_rollback": -0.20},
            },
            "B": {
                "prob": 0.40, "root_cause": "legitimate_batch",
                "reveal": "[TELEMETRY] Quarter-end financial report running (scheduled) | CFO requested manually | All queries healthy | Expected completion: 18 min | Kill would corrupt report",
                "correct_diagnosis": "diagnose_cpu_spike",
                "correct_fix": "fix_scale_up",   # let it finish, add capacity
                "diag_rewards": {"diagnose_cpu_spike": 0.40, "diagnose_memory_leak": -0.10, "diagnose_db_lock": -0.15, "diagnose_network_latency": -0.20},
                "fix_rewards": {"fix_scale_up": 0.40, "fix_kill_process": -0.50, "fix_restart_service": -0.15, "fix_rollback": -0.20},
            },
        },
    },

    {
        "id": "dev_H005", "ambiguity": 0.78,
        "visible_alert": "Disk I/O wait: 68% on DB-PROD-01 | Latency: 340ms avg | Duration: 15 min",
        "hidden_states": {
            "A": {
                "prob": 0.55, "root_cause": "disk_failure",
                "reveal": "[TELEMETRY] SMART: 847 reallocated sectors (threshold: 5) | I/O errors in dmesg: 124 in last 10 min | RAID degraded | Backup disk available",
                "correct_diagnosis": "diagnose_db_lock",
                "correct_fix": "fix_restart_service",   # failover to replica
                "diag_rewards": {"diagnose_db_lock": 0.35, "diagnose_memory_leak": -0.10, "diagnose_cpu_spike": 0.05, "diagnose_network_latency": -0.15},
                "fix_rewards": {"fix_restart_service": 0.40, "fix_rollback": 0.15, "fix_scale_up": 0.05, "fix_kill_process": -0.20},
            },
            "B": {
                "prob": 0.45, "root_cause": "backup_running",
                "reveal": "[TELEMETRY] Daily backup process running (scheduled 03:00) | SMART: healthy (0 errors) | I/O wait normal for backup workload | Expected end: 40 min",
                "correct_diagnosis": "diagnose_network_latency",   # routine, no action needed
                "correct_fix": "fix_scale_up",   # temporary throttling
                "diag_rewards": {"diagnose_network_latency": 0.30, "diagnose_db_lock": -0.15, "diagnose_cpu_spike": 0.10, "diagnose_memory_leak": -0.20},
                "fix_rewards": {"fix_scale_up": 0.40, "fix_kill_process": -0.15, "fix_restart_service": -0.25, "fix_rollback": -0.20},
            },
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # MEDIUM AMBIGUITY (0.40–0.65): INVESTIGATION HELPFUL
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "id": "dev_M001", "ambiguity": 0.60,
        "visible_alert": "Database query response time: 2.8s avg (baseline: 180ms) | Duration: 12 min",
        "hidden_states": {
            "A": {
                "prob": 0.70, "root_cause": "table_lock",
                "reveal": "[TELEMETRY] SHOW PROCESSLIST: 94 queries WAITING on table lock | Long transaction: analytics-job (running 47min) | Blocking all writes to orders table",
                "correct_diagnosis": "diagnose_db_lock",
                "correct_fix": "fix_kill_process",
                "diag_rewards": {"diagnose_db_lock": 0.40, "diagnose_memory_leak": -0.10, "diagnose_cpu_spike": -0.10, "diagnose_network_latency": 0.05},
                "fix_rewards": {"fix_kill_process": 0.40, "fix_restart_service": 0.10, "fix_scale_up": -0.10, "fix_rollback": -0.20},
            },
            "B": {
                "prob": 0.30, "root_cause": "missing_index",
                "reveal": "[TELEMETRY] EXPLAIN shows full table scan: orders (220M rows) | New query pattern after feature release v3.1.2 | Index: orders_user_id missing",
                "correct_diagnosis": "diagnose_db_lock",
                "correct_fix": "fix_rollback",  # rollback the feature
                "diag_rewards": {"diagnose_db_lock": 0.35, "diagnose_network_latency": 0.10, "diagnose_cpu_spike": -0.10, "diagnose_memory_leak": -0.15},
                "fix_rewards": {"fix_rollback": 0.40, "fix_kill_process": 0.10, "fix_scale_up": -0.10, "fix_restart_service": 0.05},
            },
        },
    },

    {
        "id": "dev_M002", "ambiguity": 0.50,
        "visible_alert": "WebSocket connections dropping | Reconnect storms observed | Rate: 340/min",
        "hidden_states": {
            "A": {
                "prob": 0.65, "root_cause": "connection_limit",
                "reveal": "[TELEMETRY] nginx worker_connections: 1024 (at limit) | Active: 1,024/1,024 | Upgrade connections: 47 queued | CPU: 28% | File descriptors: OK",
                "correct_diagnosis": "diagnose_network_latency",
                "correct_fix": "fix_scale_up",
                "diag_rewards": {"diagnose_network_latency": 0.40, "diagnose_db_lock": -0.10, "diagnose_cpu_spike": 0.10, "diagnose_memory_leak": -0.15},
                "fix_rewards": {"fix_scale_up": 0.40, "fix_restart_service": 0.10, "fix_kill_process": -0.15, "fix_rollback": -0.10},
            },
            "B": {
                "prob": 0.35, "root_cause": "client_bug",
                "reveal": "[TELEMETRY] App v4.2.1 deployed 13:00 | New WebSocket client reconnects every 3s regardless of connection state | Server connections: healthy | Bug in client retry logic",
                "correct_diagnosis": "diagnose_network_latency",
                "correct_fix": "fix_rollback",
                "diag_rewards": {"diagnose_network_latency": 0.35, "diagnose_cpu_spike": 0.10, "diagnose_db_lock": -0.15, "diagnose_memory_leak": -0.15},
                "fix_rewards": {"fix_rollback": 0.40, "fix_scale_up": 0.10, "fix_restart_service": 0.05, "fix_kill_process": -0.10},
            },
        },
    },

    {
        "id": "dev_M003", "ambiguity": 0.45,
        "visible_alert": "Kubernetes pod restart loop | Pod: payment-worker | Restarts: 47 in 30 min",
        "hidden_states": {
            "A": {
                "prob": 0.75, "root_cause": "oom_kill",
                "reveal": "[TELEMETRY] OOMKilled: true | Memory limit: 512Mi | Last 3 restarts: OOM at 511Mi | Heap dump: large in-memory cache not bounded",
                "correct_diagnosis": "diagnose_memory_leak",
                "correct_fix": "fix_restart_service",  # with memory limit increase
                "diag_rewards": {"diagnose_memory_leak": 0.40, "diagnose_cpu_spike": 0.05, "diagnose_db_lock": -0.15, "diagnose_network_latency": -0.15},
                "fix_rewards": {"fix_restart_service": 0.40, "fix_scale_up": 0.15, "fix_kill_process": 0.05, "fix_rollback": -0.10},
            },
            "B": {
                "prob": 0.25, "root_cause": "startup_crash",
                "reveal": "[TELEMETRY] Exit code: 1 | Logs: 'Failed to connect to Redis: connection refused' | Redis pod: CrashLoopBackOff | Dependency not healthy",
                "correct_diagnosis": "diagnose_db_lock",  # dependency issue
                "correct_fix": "fix_restart_service",  # restart Redis first
                "diag_rewards": {"diagnose_db_lock": 0.35, "diagnose_memory_leak": -0.20, "diagnose_network_latency": 0.10, "diagnose_cpu_spike": -0.10},
                "fix_rewards": {"fix_restart_service": 0.40, "fix_rollback": 0.15, "fix_kill_process": -0.10, "fix_scale_up": -0.15},
            },
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # LOW AMBIGUITY (0.05–0.25): INVESTIGATION WASTEFUL
    # Clear signals — agent should diagnose without investigating
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "id": "dev_L001", "ambiguity": 0.10,
        "visible_alert": "CRITICAL: Disk /var/data 100% full on DB-PROD-01 | All writes failing | Data loss imminent",
        "hidden_states": {
            "A": {
                "prob": 1.0, "root_cause": "disk_full",
                "reveal": "[TELEMETRY] /var/data: 2TB/2TB | Largest: core dumps (847GB) from last week | MySQL write error: 'No space left on device' | Replication lag: 47s and growing",
                "correct_diagnosis": "diagnose_db_lock",   # I/O blocked = db_lock analogue
                "correct_fix": "fix_kill_process",         # kill dump-generating processes, clear space
                "diag_rewards": {"diagnose_db_lock": 0.40, "diagnose_memory_leak": 0.10, "diagnose_cpu_spike": -0.10, "diagnose_network_latency": -0.20},
                "fix_rewards": {"fix_kill_process": 0.40, "fix_restart_service": 0.15, "fix_scale_up": 0.05, "fix_rollback": -0.20},
            },
        },
    },

    {
        "id": "dev_L002", "ambiguity": 0.08,
        "visible_alert": "Deployment rollback requested by team lead | Reason: 'v2.4.1 causes checkout failures' | Rollback target: v2.4.0",
        "hidden_states": {
            "A": {
                "prob": 1.0, "root_cause": "bad_deploy",
                "reveal": "[TELEMETRY] v2.4.1 checkout_service: TypeError in cart.total() | Error rate: 28% | Revenue impact: ~$4k/min | v2.4.0: stable for 6 days",
                "correct_diagnosis": "diagnose_cpu_spike",  # deploy-related error
                "correct_fix": "fix_rollback",
                "diag_rewards": {"diagnose_cpu_spike": 0.30, "diagnose_db_lock": 0.20, "diagnose_memory_leak": -0.10, "diagnose_network_latency": -0.10},
                "fix_rewards": {"fix_rollback": 0.45, "fix_restart_service": 0.10, "fix_kill_process": -0.10, "fix_scale_up": -0.20},
            },
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# ACTION SETS
# ──────────────────────────────────────────────────────────────────────────────
_S0_BASE = ["diagnose_cpu_spike", "diagnose_memory_leak", "diagnose_db_lock", "diagnose_network_latency"]
_S0_WITH_INVEST = ["investigate"] + _S0_BASE
_S1 = ["fix_restart_service", "fix_kill_process", "fix_rollback", "fix_scale_up"]
_S2 = ["verify_metrics_ok", "verify_check_logs", "verify_ask_user"]
_S3 = ["close_resolved", "close_partial", "escalate_senior"]


def _pick_hidden_state(scenario: dict, seed: Optional[int], ep: int) -> str:
    states = scenario["hidden_states"]
    if len(states) == 1:
        return list(states.keys())[0]
    key = f"{scenario['id']}_ep{ep}_seed{seed if seed is not None else 'none'}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    r = (h % 10_000) / 10_000.0
    cumulative = 0.0
    for k, v in states.items():
        cumulative += v["prob"]
        if r < cumulative:
            return k
    return list(states.keys())[-1]


class DevOpsIncidentTask(BaseTask):
    task_id = "devops_incident"
    max_steps = 4   # diagnose → fix → verify → close (INVESTIGATE does not consume a step)

    def __init__(self):
        self._ep = -1
        self._seed: Optional[int] = None
        self._scenario: dict = {}
        self._active_state_key: str = "A"
        self._active_state: dict = {}
        self._step = 0
        self._api_calls = 0
        self._history: list = []
        self._done = False
        self._investigated = False
        self._diagnosis = ""
        self._fix = ""

    def reset(self, seed: Optional[int] = None):
        self._ep += 1
        self._seed = seed
        if seed is not None:
            random.seed(seed)
        self._scenario = _SCENARIO_CLASSES[self._ep % len(_SCENARIO_CLASSES)]
        self._active_state_key = _pick_hidden_state(self._scenario, seed, self._ep)
        self._active_state = self._scenario["hidden_states"][self._active_state_key]
        self._step = 0
        self._api_calls = 0
        self._history = []
        self._done = False
        self._investigated = False
        self._diagnosis = ""
        self._fix = ""
        return self._obs()

    def step(self, action: Action):
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        t = action.type
        self._api_calls += 1

        # ── INVESTIGATE: pulls telemetry, does NOT advance _step ─────────────
        if t == "investigate":
            if self._step != 0:
                return self._obs(), Reward(value=0.01, breakdown={"error": "investigate_invalid_step"}, raw=0.01), False, {}
            self._investigated = True
            r = investigation_reward(self._scenario["ambiguity"])
            self._history.append({
                "api_call": self._api_calls, "step": self._step, "action": "investigate",
                "reward": {"value": r, "breakdown": {"investigation": r}},
                "reveal": self._active_state["reveal"],
            })
            return self._obs(), Reward(value=r, breakdown={"investigation": r}, raw=r), False, {
                "info": "Telemetry data retrieved. Make your diagnosis.",
                "reveal": self._active_state["reveal"],
            }

        # ── STEP 0: Diagnosis ────────────────────────────────────────────────
        if self._step == 0:
            if t not in _S0_BASE:
                t = _S0_BASE[0]
            base_r = self._active_state["diag_rewards"].get(t, -0.10)
            correct = (t == self._active_state["correct_diagnosis"])
            cal_r = calibration_reward(correct, self._scenario["ambiguity"], self._investigated)
            rval = max(0.01, min(0.99, (base_r * 0.35) + (cal_r * 0.65)))
            breakdown = {
                "diagnosis_base": base_r, "calibration_reward": cal_r,
                "investigated": self._investigated, "ambiguity": self._scenario["ambiguity"],
            }
            self._diagnosis = t

        # ── STEP 1: Fix ──────────────────────────────────────────────────────
        elif self._step == 1:
            if t not in _S1:
                t = _S1[0]
            base_r = self._active_state["fix_rewards"].get(t, -0.10)
            # Reckless fix: applying kill/restart without investigating is risky
            if not self._investigated and t == "fix_kill_process":
                base_r -= 0.10
            correct = (t == self._active_state["correct_fix"])
            cal_r = calibration_reward(correct, self._scenario["ambiguity"] * 0.6, self._investigated)
            rval = max(0.01, min(0.99, (base_r * 0.35) + (cal_r * 0.65)))
            breakdown = {"fix_base": base_r, "calibration_reward": cal_r}
            self._fix = t

        # ── STEP 2: Verification ─────────────────────────────────────────────
        elif self._step == 2:
            rval = 0.25 if t == "verify_metrics_ok" else 0.12
            breakdown = {"verification": rval}

        # ── STEP 3: Close ────────────────────────────────────────────────────
        elif self._step == 3:
            rval = 0.20 if t == "close_resolved" else 0.08
            breakdown = {"resolution": rval}
            self._done = True

        else:
            return self._obs(), Reward(value=0.01, breakdown={}, raw=0.01), True, {}

        reward = Reward(value=round(rval, 4), breakdown=breakdown, raw=rval)
        self._history.append({
            "api_call": self._api_calls, "step": self._step, "action": t,
            "reward": {"value": reward.value, "breakdown": breakdown},
        })
        self._step += 1
        if self._step >= self.max_steps:
            self._done = True
        obs = self._obs()
        return obs, reward, self._done, {
            "step": self._step - 1, "action": t,
            "episode_score": self.grade_episode(self._history) if self._done else None,
        }

    def state(self):
        return {
            "task_id": self.task_id, "step": self._step, "done": self._done,
            "scenario_id": self._scenario.get("id", ""),
            "ambiguity": self._scenario.get("ambiguity", 0.0),
            "investigated": self._investigated,
            "root_cause": self._active_state.get("root_cause", "LOCKED") if self._investigated else "LOCKED",
            "diagnosis": self._diagnosis, "fix": self._fix,
        }

    def grade_episode(self, history):
        total = sum(e.get("reward", {}).get("value", 0.0) for e in history)
        return clamp(total / 1.0)

    def _obs(self) -> Observation:
        s = self._scenario
        idx = min(self._step, 3)
        avails = [_S0_WITH_INVEST if not self._investigated else _S0_BASE, _S1, _S2, _S3]
        avail = avails[idx] if not self._done else []

        if self._investigated:
            telemetry = self._active_state["reveal"]
        else:
            ambiguity = s["ambiguity"]
            if ambiguity >= 0.70:
                telemetry = "[TELEMETRY LOCKED] ⚠️ Multiple root causes possible — use 'investigate' to pull stack traces, heap dumps, and metrics"
            elif ambiguity >= 0.40:
                telemetry = "[TELEMETRY LOCKED] ⚡ Partial signals available — investigation recommended"
            else:
                telemetry = "[TELEMETRY LOCKED] ✅ Alert is self-explanatory — you may diagnose directly"

        prompts = [
            f"Alert: {s['visible_alert']}\nTelemetry: {telemetry}\n\nSelect diagnosis. Available: {avail}",
            f"Diagnosis: {self._diagnosis}\nApply fix. Available: {avail}",
            f"Fix applied: {self._fix}\nVerify system health. Available: {avail}",
            f"System stable. Close the incident. Available: {avail}",
        ]

        states = [
            {"alert": s["visible_alert"], "telemetry": telemetry, "investigated": self._investigated},
            {"alert": s["visible_alert"], "diagnosis": self._diagnosis},
            {"diagnosis": self._diagnosis, "fix": self._fix},
            {"diagnosis": self._diagnosis, "fix": self._fix, "verified": True},
        ]

        return Observation(
            task_id=self.task_id,
            step=self._step,
            state=states[idx],
            history=list(self._history),
            available_actions=avail,
            done=self._done,
            prompt=prompts[idx],
            context=prompts[idx],
            task=self.task_id,
            action_to_evaluate="Evaluating agent response...",
        )
