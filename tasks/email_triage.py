"""
tasks/email_triage.py — Task 1 (Easy): Email Triage
Steps: 3 | Max Score: 1.0
"""
from __future__ import annotations
import random
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from models import Action, Observation, Reward
from tasks.base import BaseTask
from utils import clamp

_SCENARIOS = [
    {
        "id": "em_001", "sender": "no-reply@update-paypa1-now.net",
        "subject": "Your PayPal account suspended — verify immediately",
        "body": "Click here: http://paypa1-verify.ru/login",
        "metadata": {"sender_domain_age_days": 2, "link_domain_mismatch": True},
        "correct_classification": "classify_phishing", "correct_response": "reply_block",
        "classification_rewards": {"classify_phishing": 0.40, "classify_spam": 0.15, "classify_urgent": -0.10, "classify_normal": -0.20},
        "response_rewards": {"reply_block": 0.40, "escalate_security": 0.35, "archive": 0.05, "reply_auto": -0.20, "escalate_manager": 0.10},
    },
    {
        "id": "em_002", "sender": "ops-alert@datacenter-monitor.internal",
        "subject": "CRITICAL: Production database unreachable",
        "body": "Alert at 14:32:01 UTC. DB-PROD-01 not responding. Connection timeout 30s.",
        "metadata": {"internal_sender": True, "alert_level": "critical"},
        "correct_classification": "classify_urgent", "correct_response": "escalate_manager",
        "classification_rewards": {"classify_urgent": 0.40, "classify_normal": -0.10, "classify_spam": -0.20, "classify_phishing": -0.20},
        "response_rewards": {"escalate_manager": 0.40, "escalate_security": 0.20, "reply_auto": -0.10, "reply_block": -0.30, "archive": -0.40},
    },
    {
        "id": "em_003", "sender": "newsletter@techdigest.io",
        "subject": "This week in AI — 10 papers you should read",
        "body": "Hello subscriber, here is your weekly digest. Unsubscribe at the bottom.",
        "metadata": {"bulk_sender": True, "unsubscribe_link": True, "sender_domain_age_days": 1820},
        "correct_classification": "classify_spam", "correct_response": "archive",
        "classification_rewards": {"classify_spam": 0.40, "classify_normal": 0.20, "classify_urgent": -0.10, "classify_phishing": -0.30},
        "response_rewards": {"archive": 0.40, "reply_auto": 0.05, "reply_block": -0.20, "escalate_security": -0.30, "escalate_manager": -0.20},
    },
    {
        "id": "em_004", "sender": "hr@company.com",
        "subject": "Action required: Update benefits enrollment by Friday",
        "body": "Open enrollment closes Friday 5 PM. Log in to employee portal to confirm.",
        "metadata": {"internal_sender": True, "deadline": "Friday 17:00", "sender_domain_age_days": 3650},
        "correct_classification": "classify_normal", "correct_response": "reply_auto",
        "classification_rewards": {"classify_normal": 0.40, "classify_urgent": 0.15, "classify_spam": -0.10, "classify_phishing": -0.20},
        "response_rewards": {"reply_auto": 0.40, "archive": 0.15, "escalate_manager": -0.10, "reply_block": -0.30, "escalate_security": -0.20},
    },
]

_S0 = ["classify_spam", "classify_phishing", "classify_urgent", "classify_normal"]
_S1 = ["reply_auto", "reply_block", "escalate_security", "escalate_manager", "archive"]
_S2 = ["confirm", "cancel"]
_MAX = 1.0


class EmailTriageTask(BaseTask):
    task_id = "email_triage"
    max_steps = 3

    def __init__(self):
        self._ep = -1
        self._seed: int | None = None
        self._scenario = {}
        self._step = 0
        self._history = []
        self._done = False
        self._classify = ""
        self._response = ""

    def reset(self, seed: int | None = None):
        self._ep += 1
        if seed is not None:
            self._seed = seed
            random.seed(seed)
        else:
            self._seed = None
        self._scenario = _SCENARIOS[self._ep % len(_SCENARIOS)]
        self._step = 0; self._history = []; self._done = False
        self._classify = ""; self._response = ""
        return self._obs()

    def step(self, action: Action):
        if self._done:
            raise RuntimeError("Done. Call reset().")
        t = action.type; rval = 0.0; breakdown = {}; info = {"step": self._step, "action": t}
        if self._step == 0:
            if t not in _S0: t = "classify_normal"
            r = self._scenario["classification_rewards"].get(t, -0.20)
            breakdown["classification"] = r; rval = max(0.01, r); self._classify = t
            info["correct"] = (t == self._scenario["correct_classification"])
        elif self._step == 1:
            if t not in _S1: t = "archive"
            r = self._scenario["response_rewards"].get(t, -0.20)
            breakdown["response_action"] = r; rval = max(0.01, r); self._response = t
            info["correct"] = (t == self._scenario["correct_response"])
        elif self._step == 2:
            if t == "confirm":
                breakdown["confirmation"] = 0.20; rval = 0.20
            else:
                breakdown["confirmation"] = 0.0; rval = 0.01
            self._done = True
        reward = Reward(value=round(rval, 4), breakdown=breakdown, raw=rval)
        self._history.append({"step": self._step, "action": t, "reward": {"value": reward.value, "breakdown": breakdown}})
        self._step += 1
        if self._step >= self.max_steps: self._done = True
        obs = self._obs()
        info["episode_score"] = self.grade_episode(self._history) if self._done else None
        return obs, reward, self._done, info

    def state(self):
        return {"task_id": self.task_id, "step": self._step, "done": self._done,
                "scenario_id": self._scenario.get("id", ""), "classification": self._classify, "response": self._response}

    def grade_episode(self, history):
        total = sum(e.get("reward", {}).get("value", 0.0) for e in history)
        return clamp(total / _MAX)

    def _obs(self) -> Observation:
        s = self._scenario
        if self._step == 0:
            avail = _S0
            state = {"email": {"from": s["sender"], "subject": s["subject"], "body": s["body"], "metadata": s["metadata"]}}
            prompt = f"From: {s['sender']}\nSubject: {s['subject']}\nBody: {s['body']}\nMetadata: {s['metadata']}\n\nClassify this email. Available: {avail}"
        elif self._step == 1:
            avail = _S1
            state = {"email": {"from": s["sender"], "subject": s["subject"]}, "classification": self._classify}
            prompt = f"Classified as: {self._classify}\nChoose response. Available: {avail}"
        else:
            avail = _S2
            state = {"classification": self._classify, "response": self._response}
            prompt = f"Will classify as '{self._classify}' and respond '{self._response}'. Confirm? Available: {avail}"
        
        return Observation(
            task_id=self.task_id,
            step=self._step,
            state=state,
            history=list(self._history),
            available_actions=avail,
            done=self._done,
            prompt=prompt,
            context=prompt,
            task=self.task_id,
            action_to_evaluate="Evaluating agent response..."
        )
