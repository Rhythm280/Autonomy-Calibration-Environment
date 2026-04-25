"""
tasks/email_triage.py — Task 1: Email Forensic Triage (Epistemic RL v2.0)
─────────────────────────────────────────────────────────────────────────────
Design Principles:
  - Visible observation is DELIBERATELY AMBIGUOUS (sender domain masked, metadata locked)
  - Hidden truth is selected probabilistically per (seed, episode_id)
  - INVESTIGATE reveals forensic metadata that disambiguates the hidden truth
  - calibration_reward() ensures blind lucky guesses ≠ informed correct decisions
  - 10 scenarios: 5 high-ambiguity, 3 medium, 2 clear (low ambiguity)

The agent MUST learn: investigate when signal is weak, act directly when signal is strong.
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
# 10 SCENARIO CLASSES
# Each "class" defines a VISIBLE surface and multiple hidden realities.
# The hidden truth is chosen by seed at reset() time.
# ──────────────────────────────────────────────────────────────────────────────

_SCENARIO_CLASSES = [

    # ═══════════════════════════════════════════════════════════════════════════
    # HIGH AMBIGUITY (0.75–0.95): INVESTIGATION STRONGLY REQUIRED
    # Visible surface is genuinely ambiguous — could be legit or attack
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "id": "em_H001", "ambiguity": 0.90,
        "visible": {
            "sender": "security@...",   # domain deliberately masked
            "subject": "Unusual sign-in detected on your account",
            "body": "A new device signed in from an unfamiliar location. Click here if this wasn't you.",
        },
        "hidden_states": {
            "A": {
                "prob": 0.55, "truth": "phishing",
                "reveal": "[FORENSICS] Sender: security@accounts-googIe-verify.net | Domain: 3 days old | IP: 185.220.101.5 (known Tor exit node) | DKIM: FAIL | SPF: FAIL",
                "correct_classification": "classify_phishing",
                "correct_response": "reply_block",
                "clf_rewards": {"classify_phishing": 0.40, "classify_urgent": -0.20, "classify_spam": 0.05, "classify_normal": -0.35},
                "rsp_rewards": {"reply_block": 0.45, "escalate_security": 0.30, "archive": -0.10, "reply_auto": -0.45, "escalate_manager": 0.05},
            },
            "B": {
                "prob": 0.45, "truth": "legitimate",
                "reveal": "[FORENSICS] Sender: security@accounts.google.com | Domain: 9,125 days old | IP: 209.85.220.41 (Google LLC, AS15169) | DKIM: PASS | SPF: PASS",
                "correct_classification": "classify_normal",
                "correct_response": "reply_auto",
                "clf_rewards": {"classify_normal": 0.40, "classify_urgent": 0.15, "classify_spam": -0.20, "classify_phishing": -0.55},
                "rsp_rewards": {"reply_auto": 0.45, "archive": 0.10, "escalate_manager": -0.10, "reply_block": -0.55, "escalate_security": -0.25},
            },
        },
    },

    {
        "id": "em_H002", "ambiguity": 0.85,
        "visible": {
            "sender": "payments@...",
            "subject": "Payment of $1,249.99 processed — Transaction ID: TXN-88823",
            "body": "Your payment was processed. If you did not authorize this transaction, dispute it immediately.",
        },
        "hidden_states": {
            "A": {
                "prob": 0.50, "truth": "phishing",
                "reveal": "[FORENSICS] Sender: payments@paypa1-confirm.ru | Domain: 6 days old | No SPF record | Link: http://paypa1-confirm.ru/dispute (credential harvester) | SSL: self-signed",
                "correct_classification": "classify_phishing",
                "correct_response": "reply_block",
                "clf_rewards": {"classify_phishing": 0.40, "classify_urgent": -0.10, "classify_spam": 0.10, "classify_normal": -0.30},
                "rsp_rewards": {"reply_block": 0.45, "escalate_security": 0.30, "archive": -0.10, "reply_auto": -0.45, "escalate_manager": 0.05},
            },
            "B": {
                "prob": 0.50, "truth": "legitimate",
                "reveal": "[FORENSICS] Sender: payments@paypal.com | Domain: 9,490 days | SPF: PASS | DKIM: PASS | TXN-88823 matches purchase in order history 4 minutes ago",
                "correct_classification": "classify_normal",
                "correct_response": "archive",
                "clf_rewards": {"classify_normal": 0.40, "classify_urgent": 0.05, "classify_spam": 0.10, "classify_phishing": -0.55},
                "rsp_rewards": {"archive": 0.45, "reply_auto": 0.10, "escalate_manager": -0.10, "reply_block": -0.55, "escalate_security": -0.20},
            },
        },
    },

    {
        "id": "em_H003", "ambiguity": 0.88,
        "visible": {
            "sender": "hr@...",
            "subject": "Important: Verify your direct deposit information before Friday",
            "body": "HR is updating payroll records. Please verify your banking details using the link below to avoid payment delays.",
        },
        "hidden_states": {
            "A": {
                "prob": 0.60, "truth": "bec_phishing",
                "reveal": "[FORENSICS] Sender: hr@company-hr-portal.info (NOT company domain) | Not in org directory | IP: 41.78.1.19 (Lagos, Nigeria) | DKIM: FAIL",
                "correct_classification": "classify_phishing",
                "correct_response": "escalate_security",
                "clf_rewards": {"classify_phishing": 0.40, "classify_urgent": -0.10, "classify_spam": 0.05, "classify_normal": -0.40},
                "rsp_rewards": {"escalate_security": 0.45, "reply_block": 0.30, "archive": -0.15, "reply_auto": -0.50, "escalate_manager": 0.10},
            },
            "B": {
                "prob": 0.40, "truth": "legitimate_hr",
                "reveal": "[FORENSICS] Sender: hr@acme-corp.com | DKIM: PASS | SPF: PASS | Matched in directory: HR Manager Jane Doe | Annual payroll update — approved by CFO",
                "correct_classification": "classify_normal",
                "correct_response": "reply_auto",
                "clf_rewards": {"classify_normal": 0.40, "classify_urgent": 0.15, "classify_spam": -0.20, "classify_phishing": -0.55},
                "rsp_rewards": {"reply_auto": 0.45, "escalate_manager": 0.05, "archive": -0.10, "reply_block": -0.55, "escalate_security": -0.30},
            },
        },
    },

    {
        "id": "em_H004", "ambiguity": 0.82,
        "visible": {
            "sender": "it-support@...",
            "subject": "Your corporate password expires in 24 hours",
            "body": "Your password expires tomorrow. Reset now to avoid account lockout. [Reset Password]",
        },
        "hidden_states": {
            "A": {
                "prob": 0.55, "truth": "phishing",
                "reveal": "[FORENSICS] Sender: it-support@corp-password-reset.com | Link: https://corp-password-reset.com (not corporate domain) | DKIM: FAIL | Domain: 12 days old",
                "correct_classification": "classify_phishing",
                "correct_response": "reply_block",
                "clf_rewards": {"classify_phishing": 0.40, "classify_urgent": -0.10, "classify_spam": 0.05, "classify_normal": -0.35},
                "rsp_rewards": {"reply_block": 0.45, "escalate_security": 0.30, "archive": -0.10, "reply_auto": -0.50, "escalate_manager": 0.05},
            },
            "B": {
                "prob": 0.45, "truth": "legitimate_it",
                "reveal": "[FORENSICS] Sender: it-support@acme-corp.com | Link: https://identity.acme-corp.com/reset | DKIM: PASS | Matches AD policy: 90-day password rotation",
                "correct_classification": "classify_urgent",
                "correct_response": "reply_auto",
                "clf_rewards": {"classify_urgent": 0.40, "classify_normal": 0.20, "classify_spam": -0.10, "classify_phishing": -0.55},
                "rsp_rewards": {"reply_auto": 0.45, "archive": 0.10, "escalate_manager": -0.05, "reply_block": -0.55, "escalate_security": -0.20},
            },
        },
    },

    {
        "id": "em_H005", "ambiguity": 0.78,
        "visible": {
            "sender": "no-reply@...",
            "subject": "Invoice #INV-2024-0931 — $3,400.00 — Payment Due in 7 Days",
            "body": "Dear customer, your invoice for $3,400.00 is attached. Please process payment within 7 days.",
        },
        "hidden_states": {
            "A": {
                "prob": 0.45, "truth": "phishing_invoice",
                "reveal": "[FORENSICS] Sender: no-reply@invoice-docs-secure.net | Domain: 8 days | Attachment: .docm with macro (MALICIOUS) | No prior relationship in ERP system",
                "correct_classification": "classify_phishing",
                "correct_response": "reply_block",
                "clf_rewards": {"classify_phishing": 0.40, "classify_spam": 0.10, "classify_normal": -0.25, "classify_urgent": -0.10},
                "rsp_rewards": {"reply_block": 0.45, "escalate_security": 0.25, "archive": -0.10, "reply_auto": -0.40, "escalate_manager": 0.05},
            },
            "B": {
                "prob": 0.55, "truth": "legitimate_invoice",
                "reveal": "[FORENSICS] Sender: no-reply@quickbooks.intuit.com | Domain: 7,300 days | DKIM: PASS | Matches Q4 SaaS subscription in ERP | Amount matches contract",
                "correct_classification": "classify_normal",
                "correct_response": "archive",
                "clf_rewards": {"classify_normal": 0.40, "classify_spam": 0.15, "classify_urgent": 0.05, "classify_phishing": -0.55},
                "rsp_rewards": {"archive": 0.45, "reply_auto": 0.15, "escalate_manager": -0.10, "reply_block": -0.55, "escalate_security": -0.20},
            },
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # MEDIUM AMBIGUITY (0.40–0.65): INVESTIGATION HELPFUL BUT NOT MANDATORY
    # Visible signal is weak; investigating provides strong confirmation
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "id": "em_M001", "ambiguity": 0.60,
        "visible": {
            "sender": "alerts@datacenter-monitor.io",
            "subject": "ALERT: Disk utilization 89% on PROD-DB-01",
            "body": "Automated alert: /var/log partition at 89% capacity. Immediate review recommended.",
        },
        "hidden_states": {
            "A": {
                "prob": 0.70, "truth": "real_urgent",
                "reveal": "[FORENSICS] Verified internal monitoring (DNS match) | /var/log grew 40GB in 2 hours from runaway audit daemon | DB backup not clearing temp files",
                "correct_classification": "classify_urgent",
                "correct_response": "escalate_manager",
                "clf_rewards": {"classify_urgent": 0.40, "classify_normal": -0.20, "classify_spam": -0.30, "classify_phishing": -0.20},
                "rsp_rewards": {"escalate_manager": 0.45, "escalate_security": 0.15, "reply_auto": -0.10, "reply_block": -0.30, "archive": -0.45},
            },
            "B": {
                "prob": 0.30, "truth": "false_alarm",
                "reveal": "[FORENSICS] Known noisy monitor | Maintenance window ongoing | Actual disk usage stable at 61% | Alert misconfigured after patch Tuesday",
                "correct_classification": "classify_normal",
                "correct_response": "archive",
                "clf_rewards": {"classify_normal": 0.40, "classify_spam": 0.20, "classify_urgent": -0.15, "classify_phishing": -0.25},
                "rsp_rewards": {"archive": 0.45, "reply_auto": 0.15, "escalate_manager": -0.25, "reply_block": -0.25, "escalate_security": -0.10},
            },
        },
    },

    {
        "id": "em_M002", "ambiguity": 0.55,
        "visible": {
            "sender": "ceo@company-external.com",
            "subject": "Confidential — wire transfer needed urgently",
            "body": "I am in a board meeting and cannot take calls. Please wire $25,000 to a new vendor. Very time sensitive.",
        },
        "hidden_states": {
            "A": {
                "prob": 0.75, "truth": "bec_fraud",
                "reveal": "[FORENSICS] Sender domain: company-external.com (not company.com) | CEO's registered email: ceo@company.com | Lookalike domain registered 2 days ago | DKIM: FAIL",
                "correct_classification": "classify_phishing",
                "correct_response": "escalate_security",
                "clf_rewards": {"classify_phishing": 0.40, "classify_urgent": -0.20, "classify_spam": 0.05, "classify_normal": -0.40},
                "rsp_rewards": {"escalate_security": 0.45, "reply_block": 0.25, "archive": -0.15, "reply_auto": -0.50, "escalate_manager": 0.15},
            },
            "B": {
                "prob": 0.25, "truth": "legitimate_urgent",
                "reveal": "[FORENSICS] CEO confirmed via phone by CFO | Using personal email due to VPN failure at offsite conference | Wire pre-approved in board meeting minutes",
                "correct_classification": "classify_urgent",
                "correct_response": "escalate_manager",
                "clf_rewards": {"classify_urgent": 0.40, "classify_normal": 0.10, "classify_spam": -0.20, "classify_phishing": -0.50},
                "rsp_rewards": {"escalate_manager": 0.45, "reply_auto": 0.10, "archive": -0.10, "reply_block": -0.50, "escalate_security": -0.25},
            },
        },
    },

    {
        "id": "em_M003", "ambiguity": 0.45,
        "visible": {
            "sender": "info@techdigest-weekly.com",
            "subject": "You've won a $500 Amazon Gift Card — Claim within 24 hours",
            "body": "Congratulations! You were selected from our subscriber list. Click to claim your prize now.",
        },
        "hidden_states": {
            "A": {
                "prob": 0.80, "truth": "spam",
                "reveal": "[FORENSICS] Bulk sender | Mailchimp ESP | Unsubscribe list | No malicious links | Marketing campaign — not phishing",
                "correct_classification": "classify_spam",
                "correct_response": "archive",
                "clf_rewards": {"classify_spam": 0.40, "classify_phishing": 0.20, "classify_normal": -0.15, "classify_urgent": -0.30},
                "rsp_rewards": {"archive": 0.45, "reply_block": 0.20, "reply_auto": -0.10, "escalate_manager": -0.30, "escalate_security": -0.05},
            },
            "B": {
                "prob": 0.20, "truth": "phishing",
                "reveal": "[FORENSICS] Link resolves to credential harvester | SSL: self-signed | Domain: 1 day old | Pixel tracker embedded",
                "correct_classification": "classify_phishing",
                "correct_response": "reply_block",
                "clf_rewards": {"classify_phishing": 0.40, "classify_spam": 0.20, "classify_normal": -0.25, "classify_urgent": -0.20},
                "rsp_rewards": {"reply_block": 0.45, "escalate_security": 0.30, "archive": -0.15, "reply_auto": -0.40, "escalate_manager": 0.05},
            },
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # LOW AMBIGUITY (0.05–0.20): INVESTIGATION IS WASTEFUL
    # Visible signal is unambiguous — agent should act directly
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "id": "em_L001", "ambiguity": 0.08,
        "visible": {
            "sender": "ops@datacenter-monitor.internal",
            "subject": "CRITICAL: Production database DOWN — 47 services failing",
            "body": "CRITICAL ALERT: DB-PROD-01 unreachable. 47 service health checks failing. On-call team paged. Immediate response required.",
        },
        "hidden_states": {
            "A": {
                "prob": 1.0, "truth": "critical_incident",
                "reveal": "[FORENSICS] Internal sender verified | 47/47 health checks failing | Last heartbeat: 8 minutes ago | Runbook: DB-INC-001",
                "correct_classification": "classify_urgent",
                "correct_response": "escalate_manager",
                "clf_rewards": {"classify_urgent": 0.40, "classify_normal": -0.30, "classify_spam": -0.40, "classify_phishing": -0.30},
                "rsp_rewards": {"escalate_manager": 0.45, "escalate_security": 0.10, "reply_auto": -0.20, "reply_block": -0.40, "archive": -0.50},
            },
        },
    },

    {
        "id": "em_L002", "ambiguity": 0.05,
        "visible": {
            "sender": "noreply@medium.com",
            "subject": "Your weekly digest: Top AI stories",
            "body": "Your weekly digest of the best AI stories on Medium. Manage preferences | Unsubscribe at any time.",
        },
        "hidden_states": {
            "A": {
                "prob": 1.0, "truth": "newsletter",
                "reveal": "[FORENSICS] Verified bulk sender: A Medium Corporation | DKIM: PASS | Opted in: 2021-04-15 | SendGrid ESP | No malicious content",
                "correct_classification": "classify_spam",
                "correct_response": "archive",
                "clf_rewards": {"classify_spam": 0.40, "classify_normal": 0.25, "classify_urgent": -0.20, "classify_phishing": -0.35},
                "rsp_rewards": {"archive": 0.45, "reply_auto": 0.10, "reply_block": -0.25, "escalate_security": -0.35, "escalate_manager": -0.25},
            },
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# ACTION SETS
# ──────────────────────────────────────────────────────────────────────────────
_S0_BASE = ["classify_spam", "classify_phishing", "classify_urgent", "classify_normal"]
_S0_WITH_INVEST = ["investigate"] + _S0_BASE
_S1 = ["reply_auto", "reply_block", "escalate_security", "escalate_manager", "archive"]
_S2 = ["confirm", "cancel"]


def _pick_hidden_state(scenario: dict, seed: Optional[int], ep: int) -> str:
    """
    Deterministic probabilistic selection of hidden state per (scenario_id, seed, episode).
    Same seed → same scenario → same hidden truth. Different seeds → distribution sampling.
    """
    states = scenario["hidden_states"]
    if len(states) == 1:
        return list(states.keys())[0]

    # Hash-based deterministic sampling from probability distribution
    key = f"{scenario['id']}_ep{ep}_seed{seed if seed is not None else 'none'}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    r = (h % 10_000) / 10_000.0  # [0.0, 1.0)

    cumulative = 0.0
    for k, v in states.items():
        cumulative += v["prob"]
        if r < cumulative:
            return k
    return list(states.keys())[-1]  # fallback


class EmailTriageTask(BaseTask):
    task_id = "email_triage"
    max_steps = 3   # classify → respond → confirm (INVESTIGATE does not consume a step)

    def __init__(self):
        self._ep = -1
        self._seed: Optional[int] = None
        self._scenario: dict = {}
        self._active_state_key: str = "A"
        self._active_state: dict = {}
        self._step = 0
        self._api_calls = 0       # total API calls including investigate
        self._history: list = []
        self._done = False
        self._investigated = False
        self._classify = ""
        self._response = ""

    # ── OpenEnv API ──────────────────────────────────────────────────────────

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
        self._classify = ""
        self._response = ""
        return self._obs()

    def step(self, action: Action):
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        t = action.type
        self._api_calls += 1

        # ── INVESTIGATE: reveals hidden truth, does NOT advance _step ────────
        if t == "investigate":
            if self._step != 0:
                # Investigate only valid at step 0
                r = 0.01
                return self._obs(), Reward(value=r, breakdown={"error": "investigate_invalid_step"}, raw=r), False, {"info": "investigate only valid at step 0"}
            self._investigated = True
            r = investigation_reward(self._scenario["ambiguity"])
            breakdown = {
                "investigate": True,
                "ambiguity": self._scenario["ambiguity"],
                "cost": -0.05,
                "revealed": self._active_state["reveal"],
            }
            self._history.append({
                "api_call": self._api_calls,
                "step": self._step,
                "action": "investigate",
                "reward": {"value": r, "breakdown": {"investigation": r}},
                "reveal": self._active_state["reveal"],
            })
            # _step NOT incremented — decision still at step 0
            reward = Reward(value=r, breakdown={"investigation": r}, raw=r)
            return self._obs(), reward, False, {
                "info": "Forensic metadata revealed. Make your classification decision.",
                "reveal": self._active_state["reveal"],
            }

        # ── STEP 0: Classification ───────────────────────────────────────────
        if self._step == 0:
            if t not in _S0_BASE:
                t = "classify_normal"
            base_r = self._active_state["clf_rewards"].get(t, -0.20)
            correct = (t == self._active_state["correct_classification"])
            cal_r = calibration_reward(correct, self._scenario["ambiguity"], self._investigated)
            # Weighted blend: calibration drives most of the signal
            rval = max(0.01, min(0.99, (base_r * 0.35) + (cal_r * 0.65)))
            breakdown = {
                "classification_base": base_r,
                "calibration_reward": cal_r,
                "investigated": self._investigated,
                "ambiguity": self._scenario["ambiguity"],
            }
            self._classify = t
            self._history.append({
                "api_call": self._api_calls, "step": self._step, "action": t,
                "correct": correct, "investigated": self._investigated,
                "reward": {"value": round(rval, 4), "breakdown": breakdown},
            })
            self._step += 1
            if self._step >= self.max_steps:
                self._done = True
            reward = Reward(value=round(rval, 4), breakdown=breakdown, raw=rval)
            obs = self._obs()
            return obs, reward, self._done, {
                "step": 0, "action": t, "correct": correct,
                "investigated": self._investigated,
                "hidden_truth": self._active_state["truth"] if self._investigated else "LOCKED",
                "episode_score": self.grade_episode(self._history) if self._done else None,
            }

        # ── STEP 1: Response action ──────────────────────────────────────────
        elif self._step == 1:
            if t not in _S1:
                t = "archive"
            base_r = self._active_state["rsp_rewards"].get(t, -0.20)
            correct = (t == self._active_state["correct_response"])
            cal_r = calibration_reward(correct, self._scenario["ambiguity"] * 0.5, self._investigated)
            rval = max(0.01, min(0.99, (base_r * 0.35) + (cal_r * 0.65)))
            breakdown = {"response_base": base_r, "calibration_reward": cal_r}
            self._response = t
            self._history.append({
                "api_call": self._api_calls, "step": self._step, "action": t,
                "correct": correct,
                "reward": {"value": round(rval, 4), "breakdown": breakdown},
            })
            self._step += 1
            if self._step >= self.max_steps:
                self._done = True
            reward = Reward(value=round(rval, 4), breakdown=breakdown, raw=rval)
            return self._obs(), reward, self._done, {
                "step": 1, "action": t, "correct": correct,
                "episode_score": self.grade_episode(self._history) if self._done else None,
            }

        # ── STEP 2: Confirmation ─────────────────────────────────────────────
        elif self._step == 2:
            rval = 0.20 if t == "confirm" else 0.01
            breakdown = {"confirmation": rval}
            self._done = True
            self._history.append({
                "api_call": self._api_calls, "step": self._step, "action": t,
                "reward": {"value": rval, "breakdown": breakdown},
            })
            self._step += 1
            reward = Reward(value=rval, breakdown=breakdown, raw=rval)
            return self._obs(), reward, True, {
                "step": 2, "action": t,
                "episode_score": self.grade_episode(self._history),
            }

        # Fallback
        return self._obs(), Reward(value=0.01, breakdown={"error": "invalid_step"}, raw=0.01), True, {}

    def state(self):
        return {
            "task_id": self.task_id,
            "step": self._step,
            "done": self._done,
            "scenario_id": self._scenario.get("id", ""),
            "ambiguity": self._scenario.get("ambiguity", 0.0),
            "investigated": self._investigated,
            "hidden_state_key": self._active_state_key if self._investigated else "LOCKED",
            "classification": self._classify,
            "response": self._response,
        }

    def grade_episode(self, history):
        total = sum(e.get("reward", {}).get("value", 0.0) for e in history)
        return clamp(total / 1.0)

    # ── Observation Builder ──────────────────────────────────────────────────

    def _obs(self) -> Observation:
        s = self._scenario
        v = s["visible"]

        if self._step == 0:
            avail = _S0_BASE if self._investigated else _S0_WITH_INVEST

            # Mask domain until investigated
            sender_display = v["sender"] if self._investigated else v["sender"].split('@')[0] + "@[LOCKED]"
            if self._investigated:
                forensics = f"\n[FORENSICS UNLOCKED]: {self._active_state['reveal']}"
                confidence_hint = "You have full forensic data. Decide with certainty."
            else:
                forensics = "\n[METADATA LOCKED]: Use 'investigate' to reveal sender domain, DKIM status, IP reputation, and domain age."
                ambiguity = s["ambiguity"]
                if ambiguity >= 0.70:
                    confidence_hint = "⚠️ Signal confidence: LOW — investigation strongly recommended"
                elif ambiguity >= 0.40:
                    confidence_hint = "⚡ Signal confidence: MEDIUM — investigation may help"
                else:
                    confidence_hint = "✅ Signal confidence: HIGH — you may act without investigating"

            prompt = (
                f"From: {sender_display}\n"
                f"Subject: {v['subject']}\n"
                f"Body: {v['body']}\n"
                f"{forensics}\n"
                f"Confidence: {confidence_hint}\n\n"
                f"Classify this email. Available actions: {avail}"
            )
            state_dict = {
                "email": {
                    "from": sender_display,
                    "subject": v["subject"],
                    "body": v["body"],
                    "forensics": forensics,
                },
                "confidence_hint": confidence_hint,
                "investigated": self._investigated,
            }

        elif self._step == 1:
            avail = _S1
            prompt = f"Classified as '{self._classify}'. Choose response action. Available: {avail}"
            state_dict = {
                "email": {"from": v["sender"], "subject": v["subject"]},
                "classification": self._classify,
            }

        else:
            avail = _S2
            prompt = f"Will classify '{self._classify}' and respond '{self._response}'. Confirm? Available: {avail}"
            state_dict = {"classification": self._classify, "response": self._response}

        return Observation(
            task_id=self.task_id,
            step=self._step,
            state=state_dict,
            history=list(self._history),
            available_actions=avail,
            done=self._done,
            prompt=prompt,
            context=prompt,
            task=self.task_id,
            action_to_evaluate="Evaluating agent response...",
        )
