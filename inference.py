"""
inference.py — OpenEnv Mandatory Inference Script v2.0
Autonomy Calibration Environment

Implements TWO baselines:
  1. rule_based (no_investigate): Never uses INVESTIGATE — simulates blind agent
  2. rule_based_smart: Always investigates first — simulates cautious agent

Both baselines are deterministic. The trained GRPO agent should outperform both
by SELECTIVELY using INVESTIGATE based on signal confidence.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ALL_TASKS = ["email_triage", "devops_incident", "financial_request"]

SYSTEM_PROMPT = (
    "You are an AI agent completing structured real-world tasks.\n"
    "At each step you receive an observation and a list of available actions.\n"
    "You must respond with ONLY a valid JSON object in this exact format:\n"
    '{"type": "<one of available_actions>", "payload": {}}\n'
    "Do not include any explanation, markdown, or extra text."
)

# ──────────────────────────────────────────────────────────────────────────────
# BASELINE 1: Blind Agent (never investigates)
# Expected performance: LOW on high-ambiguity scenarios
# ──────────────────────────────────────────────────────────────────────────────

def _call_blind_rule_based(obs: dict) -> str:
    """Never uses INVESTIGATE. Guesses based on visible signals only."""
    available = obs.get("available_actions", [])
    if not available:
        return json.dumps({"type": "confirm", "payload": {}})

    # Remove investigate from consideration
    available_no_invest = [a for a in available if a != "investigate"]
    if not available_no_invest:
        available_no_invest = available

    prompt_text = obs.get("prompt", "").lower()
    task_id = obs.get("task_id", "").lower()
    step = obs.get("step", 0)

    # Email: use simple heuristics on visible text
    if "email" in task_id:
        if step == 0:
            urgent_kw = ["critical", "down", "unreachable", "urgent", "alert"]
            phish_kw = ["suspended", "verify immediately", "click here", ".ru", "paypa1"]
            spam_kw = ["unsubscribe", "digest", "newsletter", "weekly", "winner"]
            if any(kw in prompt_text for kw in phish_kw):
                if "classify_phishing" in available_no_invest:
                    return json.dumps({"type": "classify_phishing", "payload": {}})
            if any(kw in prompt_text for kw in urgent_kw):
                if "classify_urgent" in available_no_invest:
                    return json.dumps({"type": "classify_urgent", "payload": {}})
            if any(kw in prompt_text for kw in spam_kw):
                if "classify_spam" in available_no_invest:
                    return json.dumps({"type": "classify_spam", "payload": {}})
            if "classify_normal" in available_no_invest:
                return json.dumps({"type": "classify_normal", "payload": {}})
        elif step == 1:
            for a in ["escalate_manager", "reply_block", "reply_auto", "archive"]:
                if a in available_no_invest:
                    return json.dumps({"type": a, "payload": {}})

    # DevOps: match keywords in visible alert only
    if "devops" in task_id:
        if step == 0:
            if "memory" in prompt_text and "diagnose_memory_leak" in available_no_invest:
                return json.dumps({"type": "diagnose_memory_leak", "payload": {}})
            if "cpu" in prompt_text or "spike" in prompt_text:
                if "diagnose_cpu_spike" in available_no_invest:
                    return json.dumps({"type": "diagnose_cpu_spike", "payload": {}})
            if "latency" in prompt_text or "503" in prompt_text:
                if "diagnose_network_latency" in available_no_invest:
                    return json.dumps({"type": "diagnose_network_latency", "payload": {}})
            if "diagnose_db_lock" in available_no_invest:
                return json.dumps({"type": "diagnose_db_lock", "payload": {}})
        elif step == 1:
            if "fix_restart_service" in available_no_invest:
                return json.dumps({"type": "fix_restart_service", "payload": {}})
        elif step == 2:
            if "verify_metrics_ok" in available_no_invest:
                return json.dumps({"type": "verify_metrics_ok", "payload": {}})
        elif step == 3:
            if "close_resolved" in available_no_invest:
                return json.dumps({"type": "close_resolved", "payload": {}})

    # Financial: approve by default (worst strategy — measures recklessness)
    if "financial" in task_id:
        if step == 0:
            if "approve_transfer" in available_no_invest:
                return json.dumps({"type": "approve_transfer", "payload": {}})
        for a in ["approve_after_verification", "proceed_low_risk", "log_standard", "close_completed"]:
            if a in available_no_invest:
                return json.dumps({"type": a, "payload": {}})

    # Generic fallback
    priority = ["classify_", "diagnose_", "fix_", "verify_", "log_", "close_", "confirm", "archive"]
    for p in priority:
        for a in available_no_invest:
            if a.startswith(p):
                return json.dumps({"type": a, "payload": {}})

    return json.dumps({"type": available_no_invest[0], "payload": {}})


# ──────────────────────────────────────────────────────────────────────────────
# BASELINE 2: Always-Investigate Agent (always investigates, then decides)
# Expected performance: BETTER than blind but slightly sub-optimal due to cost
# ──────────────────────────────────────────────────────────────────────────────

def _call_smart_rule_based(obs: dict, _investigated_this_ep: list) -> str:
    """Always investigates at step 0, then uses revealed info to decide."""
    available = obs.get("available_actions", [])
    if not available:
        return json.dumps({"type": "confirm", "payload": {}})

    step = obs.get("step", 0)
    prompt_text = obs.get("prompt", "").lower()

    # Always investigate first at step 0 (if not already done)
    if step == 0 and "investigate" in available and not _investigated_this_ep:
        _investigated_this_ep.append(True)
        return json.dumps({"type": "investigate", "payload": {}})

    # After investigation, use revealed forensics to decide
    if "kyc unlocked" in prompt_text or "forensics unlocked" in prompt_text or "telemetry" in prompt_text:
        # Financial
        if "fraud" in prompt_text or "shell" in prompt_text or "ofac" in prompt_text:
            if "flag_fraud" in available:
                return json.dumps({"type": "flag_fraud", "payload": {}})
            if "reject_transfer" in available:
                return json.dumps({"type": "reject_transfer", "payload": {}})
        if "dkim: fail" in prompt_text or "phishing" in prompt_text or "kyiv" in prompt_text:
            for a in ["classify_phishing", "flag_fraud", "reply_block"]:
                if a in available:
                    return json.dumps({"type": a, "payload": {}})
        if "oom" in prompt_text or "memory" in prompt_text:
            if "diagnose_memory_leak" in available:
                return json.dumps({"type": "diagnose_memory_leak", "payload": {}})
        if "connection pool" in prompt_text or "db_lock" in prompt_text or "slow query" in prompt_text:
            if "diagnose_db_lock" in available:
                return json.dumps({"type": "diagnose_db_lock", "payload": {}})
        if "cpu" in prompt_text or "requests/min" in prompt_text:
            if "diagnose_cpu_spike" in available:
                return json.dumps({"type": "diagnose_cpu_spike", "payload": {}})

    # Fallback to blind for later steps
    return _call_blind_rule_based(obs)


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI Backend
# ──────────────────────────────────────────────────────────────────────────────

def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Environment Interface
# ──────────────────────────────────────────────────────────────────────────────

def _reset(task: str, seed: int | None = None) -> dict:
    body = {"task": task}
    if seed is not None:
        body["seed"] = seed
    r = requests.post(f"{API_BASE}/reset", json=body, timeout=15)
    r.raise_for_status()
    return r.json()


def _step(action: dict) -> dict:
    r = requests.post(f"{API_BASE}/step", json=action, timeout=15)
    if r.status_code != 200:
        print(f"  [WARN] Step error {r.status_code}: {r.text[:200]}")
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# Episode Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(task_name: str, mode: str, seed: int | None = None) -> dict:
    print("[START]")
    try:
        obs = _reset(task_name, seed)
    except Exception as e:
        print(f"  [ERROR] Reset failed: {e}")
        return {"score": 0.0, "investigated": False, "steps": 0}

    step_idx = 0
    episode_score = 0.0
    investigated = False
    _invest_tracker: list = []  # mutable tracker for smart baseline

    while not obs.get("done", False) and step_idx < 12:
        if mode == "openai":
            raw = _call_openai(obs.get("prompt", ""))
        elif mode == "smart":
            raw = _call_smart_rule_based(obs, _invest_tracker)
        else:
            raw = _call_blind_rule_based(obs)

        try:
            action_dict = json.loads(raw)
        except Exception:
            action_dict = {"type": obs.get("available_actions", ["confirm"])[0], "payload": {}}

        action_str = json.dumps(action_dict, separators=(",", ":"))
        print(f"[STEP] step={step_idx} action={action_str}")

        if action_dict.get("type") == "investigate":
            investigated = True

        try:
            result = _step(action_dict)
            obs = result.get("observation", {})
            done = result.get("done", False)
            info = result.get("info", {})
            if done and info.get("episode_score") is not None:
                episode_score = info["episode_score"]
            if done:
                break
        except Exception as e:
            print(f"  [ERROR] Step failed: {e}")
            break

        step_idx += 1

    print(f"[END] score={episode_score:.4f} investigated={investigated}")
    return {"score": episode_score, "investigated": investigated, "steps": step_idx}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Autonomy Calibration — OpenEnv Inference")
    parser.add_argument("--mode", choices=["openai", "blind", "smart"], default="blind",
                        help="blind=never investigates | smart=always investigates | openai=LLM")
    parser.add_argument("--task", default="all")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]
    print(f"\n{'='*60}")
    print(f"Autonomy Calibration Benchmark — Mode: {args.mode.upper()}")
    print(f"{'='*60}")

    all_scores = {}
    for task_name in tasks_to_run:
        print(f"\n── TASK: {task_name.upper()} ──")
        task_scores = []
        task_investigated = []
        for ep in range(args.episodes):
            print(f"\n  Episode {ep+1}/{args.episodes}  seed={ep}")
            result = run_episode(task_name, args.mode, seed=ep)
            task_scores.append(result["score"])
            task_investigated.append(result["investigated"])

        avg_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
        invest_rate = sum(task_investigated) / len(task_investigated) if task_investigated else 0.0
        all_scores[task_name] = avg_score
        print(f"\n  {task_name}: avg_score={avg_score:.3f} | investigate_rate={invest_rate:.1%}")

    if len(all_scores) > 1:
        overall = sum(all_scores.values()) / len(all_scores)
        print(f"\n{'='*60}")
        print(f"OVERALL AVERAGE: {overall:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
