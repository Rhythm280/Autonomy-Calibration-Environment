"""
inference.py — OpenEnv Mandatory Inference Script
Autonomy Calibration Environment v1.0
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import requests

# ─── Config (Local Dev Port: 8000) ───────────────────────────────────────────
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

# ─── LLM Backends ────────────────────────────────────────────────────────────

def _call_openai(prompt: str) -> str:
    """Call OpenAI API with temperature=0 for deterministic output."""
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


def _call_rule_based(obs: dict) -> str:
    """Deterministic keyword heuristic fallback — no API key required."""
    available = obs.get("available_actions", [])
    if not available:
        return json.dumps({"type": "confirm", "payload": {}})

    prompt_text = obs.get("prompt", "").lower()
    task_id = obs.get("task_id", "").lower()
    step = obs.get("step", 0)

    # 1. High-priority Fraud/Safety overrides
    fraud_keywords = ["fraud", "phishing", "offshore", "unrecognized", "suspicious", "typo"]
    if any(kw in prompt_text for kw in fraud_keywords):
        for a in ["flag_fraud", "classify_phishing", "reject_transfer", "reply_block", "log_fraud_report"]:
            if a in available:
                return json.dumps({"type": a, "payload": {}})

    # 2. Context-aware task mapping
    if "email" in task_id:
        if step == 0:
            for a in ["classify_normal", "classify_urgent", "classify_spam", "classify_phishing"]:
                if a in available: return json.dumps({"type": a, "payload": {}})
        if step == 1:
            for a in ["reply_auto", "archive", "escalate_manager", "reply_block"]:
                if a in available: return json.dumps({"type": a, "payload": {}})

    if "incident" in prompt_text or "devops" in task_id:
        if "spike" in prompt_text or "cpu" in prompt_text:
            if "diagnose_cpu_spike" in available: return json.dumps({"type": "diagnose_cpu_spike", "payload": {}})
        if "memory" in prompt_text:
            if "diagnose_memory_leak" in available: return json.dumps({"type": "diagnose_memory_leak", "payload": {}})
        if "fix_restart_service" in available: return json.dumps({"type": "fix_restart_service", "payload": {}})
        if "close_resolved" in available: return json.dumps({"type": "close_resolved", "payload": {}})

    # 3. Safe fallback logic based on common prefixes
    priority = ["classify_", "diagnose_", "fix_", "verify_", "log_", "close_", "confirm", "archive"]
    for p in priority:
        for a in available:
            if a.startswith(p):
                return json.dumps({"type": a, "payload": {}})

    # Ultimate fallback: pick first available
    if available:
        return json.dumps({"type": available[0], "payload": {}})
    return json.dumps({"type": "confirm", "payload": {}})


# ─── Environment Interface ────────────────────────────────────────────────────

def _reset(task: str) -> dict:
    r = requests.post(f"{API_BASE}/reset", json={"task": task}, timeout=15)
    r.raise_for_status()
    return r.json()


def _step(action: dict) -> dict:
    r = requests.post(f"{API_BASE}/step", json=action, timeout=15)
    if r.status_code != 200:
        print(f"DEBUG Error: {r.status_code} - {r.text}")
    r.raise_for_status()
    return r.json()


# ─── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(task_name: str, mode: str) -> float:
    print("[START]")
    try:
        obs = _reset(task_name)
    except Exception as e:
        print(f"Failed to reset: {e}")
        return 0.0

    step_idx = 0
    episode_score = 0.0

    while not obs.get("done", False):
        # Generate action
        if mode == "openai":
            prompt = obs.get("prompt", "")
            raw_response = _call_openai(prompt)
        else:
            raw_response = _call_rule_based(obs)

        # Parse to dict
        try:
            action_dict = json.loads(raw_response)
        except:
            action_dict = {"type": obs.get("available_actions", ["confirm"])[0], "payload": {}}

        action_str = json.dumps(action_dict, separators=(",", ":"))
        print(f"[STEP] step={step_idx} action={action_str}")

        # Submit to environment
        try:
            result = _step(action_dict)
            obs = result.get("observation", {})
            done = result.get("done", False)
            info = result.get("info", {})
            
            if done and info.get("episode_score") is not None:
                episode_score = info["episode_score"]
        except Exception as e:
            print(f"Decision rejected: {e}")
            break

        step_idx += 1
        if step_idx > 10: break

    print(f"[END] score={episode_score:.4f}")
    return episode_score


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenEnv Inference Script")
    parser.add_argument("--mode", choices=["openai", "rule_based"], default="rule_based")
    parser.add_argument("--task", default="all")
    args = parser.parse_args()

    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]
    scores = {}

    for task_name in tasks_to_run:
        print(f"\nTASK: {task_name.upper()} | MODE: {args.mode}")
        score = run_episode(task_name, args.mode)
        scores[task_name] = score

    if len(scores) > 1:
        avg = sum(scores.values()) / len(scores)
        print(f"\nAVERAGE SCORE: {avg:.4f}")

if __name__ == "__main__":
    main()
