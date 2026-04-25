"""
tasks/financial_request.py — Task 3: Financial Fraud Detection (Epistemic RL v2.0)
─────────────────────────────────────────────────────────────────────────────
Design Principles:
  - Destination, beneficiary history, and account metadata are HIDDEN until INVESTIGATE
  - Same transfer characteristics → can be legitimate or fraud depending on hidden state
  - A blind "approve" on a fraudulent transfer in high-ambiguity state → 0.01 (reckless)
  - INVESTIGATE reveals: account age, verification status, risk signals
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
# AUTONOMY MAP
# ──────────────────────────────────────────────────────────────────────────────
_AUTONOMY_MAP = {
    "investigate": "ASK",
    "approve_transfer": "ACT", "request_verification": "ASK", "reject_transfer": "STOP",
    "flag_fraud": "STOP", "approve_after_verification": "ACT", "reject_failed_verification": "STOP",
    "escalate_compliance": "ASK", "proceed_low_risk": "ACT", "proceed_high_risk_override": "ACT",
    "hold_for_review": "RECOVER", "log_standard": "ACT", "log_suspicious_activity": "RECOVER",
    "log_fraud_report": "STOP", "close_completed": "ACT", "close_pending_review": "RECOVER",
}

# ──────────────────────────────────────────────────────────────────────────────
# 10 SCENARIO CLASSES — same visible transfer, different hidden truth
# ──────────────────────────────────────────────────────────────────────────────

_SCENARIO_CLASSES = [

    # ═══════════════════════════════════════════════════════════════════════════
    # HIGH AMBIGUITY (0.75–0.95): INVESTIGATION REQUIRED
    # Visible: amount + requestor type only. Destination LOCKED. History LOCKED.
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "id": "fr_H001", "ambiguity": 0.92,
        "visible": {
            "request_type": "wire_transfer",
            "amount_usd": 87500,
            "requestor": "account holder via web portal",
            "destination": "[LOCKED — investigate to reveal]",   # HIDDEN
            "signals": ["Exceeds 30-day average by 340%"],
        },
        "hidden_states": {
            "A": {
                "prob": 0.55, "truth": "fraud",
                "reveal": "[KYC SCAN] Destination: CAYM-9923-0041 (Cayman Islands shell, account age: 3 days) | Device: unrecognized (Ukraine) | 2FA: BYPASSED | Account velocity: 3 large transfers in 48h",
                "step0_rewards": {"flag_fraud": 0.40, "request_verification": 0.25, "reject_transfer": 0.20, "approve_transfer": -0.99},
                "is_fraudulent": True,
            },
            "B": {
                "prob": 0.45, "truth": "legitimate",
                "reveal": "[KYC SCAN] Destination: Chase Business Account (verified 8 years) | Device: recognized home IP | 2FA: PASSED | Purpose: annual property tax payment (matches records)",
                "step0_rewards": {"approve_transfer": 0.40, "request_verification": 0.25, "flag_fraud": -0.40, "reject_transfer": -0.20},
                "is_fraudulent": False,
            },
        },
    },

    {
        "id": "fr_H002", "ambiguity": 0.88,
        "visible": {
            "request_type": "international_wire",
            "amount_usd": 45000,
            "requestor": "operations manager",
            "destination": "[LOCKED — investigate to reveal]",
            "signals": ["New payee not in approved vendor list"],
        },
        "hidden_states": {
            "A": {
                "prob": 0.50, "truth": "fraud",
                "reveal": "[KYC SCAN] Beneficiary: AlphaCorp Ltd (registered 6 days ago, Malta) | Bank: Satabank (high-risk jurisdiction) | Operations manager email compromised 3 days ago per IT alert",
                "step0_rewards": {"flag_fraud": 0.40, "request_verification": 0.30, "reject_transfer": 0.20, "approve_transfer": -0.99},
                "is_fraudulent": True,
            },
            "B": {
                "prob": 0.50, "truth": "legitimate",
                "reveal": "[KYC SCAN] Beneficiary: AlphaCorp Ltd (established 2019, Malta) | Bank: BOV (Bank of Valletta, tier-1) | New vendor onboarded last week — PO#44821 issued by CFO",
                "step0_rewards": {"request_verification": 0.35, "approve_transfer": 0.30, "flag_fraud": -0.35, "reject_transfer": -0.15},
                "is_fraudulent": False,
            },
        },
    },

    {
        "id": "fr_H003", "ambiguity": 0.90,
        "visible": {
            "request_type": "urgent_wire",
            "amount_usd": 125000,
            "requestor": "CEO (email request)",
            "destination": "[LOCKED — investigate to reveal]",
            "signals": ["Marked urgent by executive", "Requests bypass of dual-control"],
        },
        "hidden_states": {
            "A": {
                "prob": 0.65, "truth": "bec_fraud",
                "reveal": "[KYC SCAN] Email domain: ceo@company-corp.io (NOT company.com) | Lookalike domain: 2 days old | CEO confirmed via phone: did NOT send this request | Classic BEC attack",
                "step0_rewards": {"flag_fraud": 0.45, "escalate_compliance": 0.30, "reject_transfer": 0.25, "approve_transfer": -0.99},
                "is_fraudulent": True,
            },
            "B": {
                "prob": 0.35, "truth": "legitimate_urgent",
                "reveal": "[KYC SCAN] CFO confirmed via Slack | Wire authorized: board pre-approval #BOA-2024-112 | Beneficiary: existing acquisition target verified by legal | CEO traveling: using assistant's email",
                "step0_rewards": {"escalate_compliance": 0.35, "request_verification": 0.30, "approve_transfer": 0.20, "flag_fraud": -0.35, "reject_transfer": -0.20},
                "is_fraudulent": False,
            },
        },
    },

    {
        "id": "fr_H004", "ambiguity": 0.82,
        "visible": {
            "request_type": "vendor_payment",
            "amount_usd": 32000,
            "requestor": "AP clerk (3 weeks tenure)",
            "destination": "[LOCKED — investigate to reveal]",
            "signals": ["Requestor payment authority limit: $5,000"],
        },
        "hidden_states": {
            "A": {
                "prob": 0.60, "truth": "fraud",
                "reveal": "[KYC SCAN] Vendor bank account IBAN changed 48 hours ago | Acme Supplies Ltd confirmed: did NOT change banking details | Clerk's login IP: proxy (mismatch from usual) | Social engineering suspected",
                "step0_rewards": {"flag_fraud": 0.40, "request_verification": 0.30, "reject_transfer": 0.20, "approve_transfer": -0.99},
                "is_fraudulent": True,
            },
            "B": {
                "prob": 0.40, "truth": "legitimate",
                "reveal": "[KYC SCAN] IBAN change confirmed by Acme CFO via signed letter (on file) | Clerk processed with manager pre-approval email chain | PO#83021 matches ERP | Amount within contract",
                "step0_rewards": {"request_verification": 0.35, "approve_transfer": 0.30, "flag_fraud": -0.30, "reject_transfer": -0.10},
                "is_fraudulent": False,
            },
        },
    },

    {
        "id": "fr_H005", "ambiguity": 0.78,
        "visible": {
            "request_type": "expense_reimbursement",
            "amount_usd": 12400,
            "requestor": "senior sales director",
            "destination": "[LOCKED — investigate to reveal]",
            "signals": ["3.1x above department average reimbursement"],
        },
        "hidden_states": {
            "A": {
                "prob": 0.45, "truth": "inflated_fraud",
                "reveal": "[KYC SCAN] Receipts: 47% missing or duplicated | $4,200 'client dinner' receipt: restaurant closed that day | Director on PIP for expense policy violations | HR flagged 2 prior incidents",
                "step0_rewards": {"flag_fraud": 0.40, "hold_for_review": 0.30, "escalate_compliance": 0.25, "approve_transfer": -0.70},
                "is_fraudulent": True,
            },
            "B": {
                "prob": 0.55, "truth": "legitimate",
                "reveal": "[KYC SCAN] Receipts: all verified | Quarter-end client entertainment | Director closed $2.1M deal this quarter | VP pre-approved via email | Amount matches policy for deal size",
                "step0_rewards": {"approve_transfer": 0.40, "request_verification": 0.20, "hold_for_review": -0.10, "flag_fraud": -0.40},
                "is_fraudulent": False,
            },
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # MEDIUM AMBIGUITY (0.40–0.65): INVESTIGATION HELPFUL
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "id": "fr_M001", "ambiguity": 0.60,
        "visible": {
            "request_type": "payroll_disbursement",
            "amount_usd": 258000,
            "requestor": "CFO via enterprise portal",
            "destination": "[LOCKED — investigate to reveal]",
            "signals": ["6.2% higher than previous cycle"],
        },
        "hidden_states": {
            "A": {
                "prob": 0.75, "truth": "legitimate",
                "reveal": "[KYC SCAN] CFO 2FA: PASSED (biometric) | Amount variance: 2 new hires + annual raise cycle | Destination: verified payroll account (8 years) | Matches HR headcount report",
                "step0_rewards": {"approve_transfer": 0.40, "request_verification": 0.25, "flag_fraud": -0.35, "reject_transfer": -0.20},
                "is_fraudulent": False,
            },
            "B": {
                "prob": 0.25, "truth": "manipulation",
                "reveal": "[KYC SCAN] CFO account: 2FA device replaced 2 days ago | Destination account: 18% of funds routing to NEW split account added last week | HR headcount: unchanged",
                "step0_rewards": {"flag_fraud": 0.40, "request_verification": 0.30, "hold_for_review": 0.20, "approve_transfer": -0.80},
                "is_fraudulent": True,
            },
        },
    },

    {
        "id": "fr_M002", "ambiguity": 0.55,
        "visible": {
            "request_type": "supplier_payment",
            "amount_usd": 18750,
            "requestor": "procurement department",
            "destination": "[LOCKED — investigate to reveal]",
            "signals": ["Existing supplier relationship"],
        },
        "hidden_states": {
            "A": {
                "prob": 0.70, "truth": "legitimate",
                "reveal": "[KYC SCAN] Supplier: Bostwick Logistics (5-year relationship) | Bank details: unchanged from last 23 payments | PO#91204 matches | CFO counter-signed",
                "step0_rewards": {"approve_transfer": 0.40, "request_verification": 0.15, "flag_fraud": -0.25, "reject_transfer": -0.15},
                "is_fraudulent": False,
            },
            "B": {
                "prob": 0.30, "truth": "fraud",
                "reveal": "[KYC SCAN] Supplier bank account: IBAN changed 5 days ago | Supplier contacted: denies changing account | Change request came via spoofed email from supplier domain | Classic supplier fraud",
                "step0_rewards": {"flag_fraud": 0.40, "request_verification": 0.30, "hold_for_review": 0.20, "approve_transfer": -0.90},
                "is_fraudulent": True,
            },
        },
    },

    {
        "id": "fr_M003", "ambiguity": 0.48,
        "visible": {
            "request_type": "consulting_invoice",
            "amount_usd": 9500,
            "requestor": "marketing department",
            "destination": "[LOCKED — investigate to reveal]",
            "signals": ["First payment to this consultant"],
        },
        "hidden_states": {
            "A": {
                "prob": 0.65, "truth": "legitimate",
                "reveal": "[KYC SCAN] Freelancer: Sarah Chen (LinkedIn verified, 847 reviews) | Contract signed via DocuSign | SOW matches deliverables | Tax form: W-9 on file",
                "step0_rewards": {"approve_transfer": 0.40, "request_verification": 0.20, "flag_fraud": -0.20, "reject_transfer": -0.10},
                "is_fraudulent": False,
            },
            "B": {
                "prob": 0.35, "truth": "ghost_vendor",
                "reveal": "[KYC SCAN] Freelancer profile: created 14 days ago | No prior work history | SSN matches deceased individual | Marketing manager is connected on LinkedIn | Ghost vendor fraud",
                "step0_rewards": {"flag_fraud": 0.40, "escalate_compliance": 0.30, "request_verification": 0.20, "approve_transfer": -0.75},
                "is_fraudulent": True,
            },
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # LOW AMBIGUITY (0.05–0.20): SIGNALS ARE CLEAR — ACT WITHOUT INVESTIGATING
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "id": "fr_L001", "ambiguity": 0.10,
        "visible": {
            "request_type": "wire_transfer",
            "amount_usd": 247000,
            "requestor": "CFO via enterprise portal",
            "destination": "US domestic payroll account #XXXX-4821 (verified 6 years)",   # VISIBLE
            "signals": ["Matches bi-weekly payroll exactly", "CFO 2FA: PASSED", "Amount matches HR headcount report"],
        },
        "hidden_states": {
            "A": {
                "prob": 1.0, "truth": "legitimate",
                "reveal": "[KYC SCAN] All signals verified | Payroll account: 6 years stable | 2FA biometric confirmed | Amount matches to the cent | Matches automated payroll run #PR-2024-089",
                "step0_rewards": {"approve_transfer": 0.45, "request_verification": 0.10, "flag_fraud": -0.50, "reject_transfer": -0.40},
                "is_fraudulent": False,
            },
        },
    },

    {
        "id": "fr_L002", "ambiguity": 0.08,
        "visible": {
            "request_type": "wire_transfer",
            "amount_usd": 500000,
            "requestor": "unknown (no authentication)",
            "destination": "Account in sanctioned country (OFAC list)",   # VISIBLE
            "signals": ["No 2FA", "Requestor not in employee directory", "OFAC match: sanctioned entity", "Submitted via public API without auth"],
        },
        "hidden_states": {
            "A": {
                "prob": 1.0, "truth": "fraud",
                "reveal": "[KYC SCAN] OFAC positive match: sanctioned entity | Submission IP: known fraud ring (Interpol DB) | Zero authentication | Regulatory violation if processed — instant license revocation",
                "step0_rewards": {"flag_fraud": 0.50, "reject_transfer": 0.40, "escalate_compliance": 0.35, "approve_transfer": -0.99},
                "is_fraudulent": True,
            },
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# ACTION SETS
# ──────────────────────────────────────────────────────────────────────────────
_S0_BASE = ["request_verification", "flag_fraud", "approve_transfer", "reject_transfer",
            "hold_for_review", "escalate_compliance"]
_S0_WITH_INVEST = ["investigate"] + _S0_BASE
_S1 = ["approve_after_verification", "reject_failed_verification", "escalate_compliance"]
_S2 = ["proceed_low_risk", "proceed_high_risk_override", "hold_for_review"]
_S3 = ["log_standard", "log_suspicious_activity", "log_fraud_report"]
_S4 = ["close_completed", "close_pending_review"]


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


class FinancialRequestTask(BaseTask):
    task_id = "financial_request"
    max_steps = 5   # assess → secondary → classify → log → close (INVESTIGATE does not consume a step)

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
        self._decisions: list = []

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
        self._decisions = []
        return self._obs()

    def step(self, action: Action):
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        t = action.type
        self._api_calls += 1

        # ── INVESTIGATE: reveals KYC/AML data, does NOT advance _step ────────
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
                "info": "KYC/AML scan complete. Make your initial assessment.",
                "reveal": self._active_state["reveal"],
            }

        # ── STEP 0: Initial Assessment ───────────────────────────────────────
        if self._step == 0:
            if t not in _S0_BASE:
                t = "request_verification"
            base_r = self._active_state["step0_rewards"].get(t, -0.10)

            correct = (
                (t in ["flag_fraud", "reject_transfer", "escalate_compliance", "hold_for_review"]
                 and self._active_state["is_fraudulent"])
                or
                (t in ["approve_transfer", "request_verification"]
                 and not self._active_state["is_fraudulent"])
            )

            # RECKLESS PENALTY: blind approve on fraudulent high-ambiguity scenario
            if (t == "approve_transfer"
                    and self._active_state["is_fraudulent"]
                    and not self._investigated
                    and self._scenario["ambiguity"] >= 0.60):
                base_r = min(base_r, -0.80)

            cal_r = calibration_reward(correct, self._scenario["ambiguity"], self._investigated)
            rval = max(0.01, min(0.99, (base_r * 0.30) + (cal_r * 0.70)))
            breakdown = {
                "decision_base": base_r, "calibration_reward": cal_r,
                "investigated": self._investigated, "ambiguity": self._scenario["ambiguity"],
                "is_fraudulent": self._active_state["is_fraudulent"],
            }
            self._decisions.append(t)
            self._history.append({
                "api_call": self._api_calls, "step": self._step, "action": t,
                "correct": correct, "investigated": self._investigated,
                "reward": {"value": round(rval, 4), "breakdown": breakdown},
                "autonomy": _AUTONOMY_MAP.get(t),
            })
            self._step += 1
            if self._step >= self.max_steps:
                self._done = True
            reward = Reward(value=round(rval, 4), breakdown=breakdown, raw=rval)
            return self._obs(), reward, self._done, {
                "step": 0, "action": t, "correct": correct,
                "investigated": self._investigated,
                "hidden_truth": self._active_state["truth"] if self._investigated else "LOCKED",
                "autonomy": _AUTONOMY_MAP.get(t),
                "episode_score": self.grade_episode(self._history) if self._done else None,
            }

        # ── STEPS 1–4: Secondary decisions (simpler, ambiguity fades) ────────
        step_configs = {
            1: (_S1, [("approve_after_verification", 0.25), ("reject_failed_verification", 0.20), ("escalate_compliance", 0.15)], 0.40),
            2: (_S2, [("hold_for_review", 0.20), ("proceed_low_risk", 0.15), ("proceed_high_risk_override", 0.10)], 0.30),
            3: (_S3, [("log_fraud_report", 0.15), ("log_suspicious_activity", 0.12), ("log_standard", 0.10)], 0.35),
            4: (_S4, [("close_pending_review", 0.15), ("close_completed", 0.12)], 0.25),
        }

        if self._step in step_configs:
            valid_actions, reward_map, default_r = step_configs[self._step]
            if t not in valid_actions:
                t = valid_actions[0]
            rval = next((r for a, r in reward_map if a == t), default_r * 0.5)
            breakdown = {f"step{self._step}": rval}

            if self._step == 4:
                self._done = True

            self._decisions.append(t)
            self._history.append({
                "api_call": self._api_calls, "step": self._step, "action": t,
                "reward": {"value": rval, "breakdown": breakdown},
                "autonomy": _AUTONOMY_MAP.get(t),
            })
            self._step += 1
            if self._step >= self.max_steps:
                self._done = True
            reward = Reward(value=round(rval, 4), breakdown=breakdown, raw=rval)
            return self._obs(), reward, self._done, {
                "step": self._step - 1, "action": t,
                "episode_score": self.grade_episode(self._history) if self._done else None,
            }

        return self._obs(), Reward(value=0.01, breakdown={}, raw=0.01), True, {}

    def state(self):
        return {
            "task_id": self.task_id, "step": self._step, "done": self._done,
            "scenario_id": self._scenario.get("id", ""),
            "ambiguity": self._scenario.get("ambiguity", 0.0),
            "investigated": self._investigated,
            "hidden_truth": self._active_state.get("truth", "LOCKED") if self._investigated else "LOCKED",
            "decisions": list(self._decisions),
        }

    def grade_episode(self, history):
        total = sum(e.get("reward", {}).get("value", 0.0) for e in history)
        return clamp(total / 1.0)

    def _obs(self) -> Observation:
        s = self._scenario
        v = s["visible"]

        step_actions = {
            0: _S0_WITH_INVEST if not self._investigated else _S0_BASE,
            1: _S1, 2: _S2, 3: _S3, 4: _S4,
        }
        avail = step_actions.get(self._step, []) if not self._done else []

        if self._step == 0:
            # Show destination only if investigated OR naturally visible (low ambiguity)
            if self._investigated:
                dest_display = self._active_state["reveal"]
                kyc_info = f"\n[KYC UNLOCKED]: {self._active_state['reveal']}"
            elif s["ambiguity"] < 0.30:
                # Destination always visible on obvious cases
                dest_display = v["destination"]
                kyc_info = "\n[SIGNALS CLEAR] — You may act directly based on visible signals"
            else:
                dest_display = v["destination"]  # shows "[LOCKED]"
                kyc_info = "\n[KYC LOCKED] — Use 'investigate' to run KYC/AML scan and reveal beneficiary details"

            ambiguity = s["ambiguity"]
            if ambiguity >= 0.70:
                confidence = "⚠️ LOW — Beneficiary not verified. Investigation strongly recommended."
            elif ambiguity >= 0.40:
                confidence = "⚡ MEDIUM — Partial signals present. Investigation recommended."
            else:
                confidence = "✅ HIGH — Signals are clear. You may act directly."

            prompt = (
                f"Request Type: {v['request_type']} | Amount: ${v['amount_usd']:,}\n"
                f"Requestor: {v['requestor']}\n"
                f"Destination: {dest_display}\n"
                f"Risk Signals: {v['signals']}\n"
                f"Signal Confidence: {confidence}"
                f"{kyc_info}\n\n"
                f"Initial assessment actions: {avail}"
            )
            state_dict = {
                "request": {
                    "type": v["request_type"],
                    "amount_usd": v["amount_usd"],
                    "requestor": v["requestor"],
                    "destination": dest_display,
                    "signals": v["signals"],
                },
                "kyc_status": "UNLOCKED" if self._investigated else "LOCKED",
                "ambiguity": s["ambiguity"],
                "confidence_hint": confidence,
                "decisions_so_far": list(self._decisions),
                "investigated": self._investigated,
            }
        else:
            ctx = (
                f"Request: {v['request_type']} ${v['amount_usd']:,} | "
                f"Decisions so far: {self._decisions}"
            )
            prompts = {
                1: f"{ctx}\n\nStep 1 — Secondary review. Available: {avail}",
                2: f"{ctx}\n\nStep 2 — Risk classification. Available: {avail}",
                3: f"{ctx}\n\nStep 3 — Activity logging. Available: {avail}",
                4: f"{ctx}\n\nStep 4 — Case closure. Available: {avail}",
            }
            prompt = prompts.get(self._step, f"{ctx}\n\nAvailable: {avail}")
            state_dict = {
                "request": {"type": v["request_type"], "amount_usd": v["amount_usd"]},
                "decisions_so_far": list(self._decisions),
                "step": self._step,
            }

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
