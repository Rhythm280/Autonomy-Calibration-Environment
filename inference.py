"""
inference.py — OpenEnv Mandatory Inference Script
Autonomy Calibration Environment v1.0

Reads from environment:
  OPENAI_API_KEY  — required for openai mode
  API_BASE_URL    — environment API base (default: http://localhost:7860/api)
  MODEL_NAME      — model to use (default: gpt-4o-mini)

Runs all 3 tasks sequentially with temperature=0 for determinism.

Output format (strictly enforced):
  [START]
  [STEP] step=0 action={"type": "...", "payload": {}}
  [STEP] step=1 action={"type": "...", "payload": {}}
  [END] score=0.85

Usage:
  python inference.py                          # uses OpenAI
  python inference.py --mode rule_based        # no API key needed (baseline)
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import requests

# ─── Config ──────────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:7860/api").rstrip("/")
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

    state = obs.get("state", {})
    prompt_text = obs.get("prompt", "").lower()

    # Safety-first rules
    fraud_keywords = ["fraud", "phishing", "offshore", "unrecognized", "suspicious", "typo"]
    stop_keywords = ["block", "reject", "flag", "stop", "banned"]
    ask_keywords = ["verify", "verification", "escalate", "compliance", "confirm", "unclear"]
    recover_keywords = ["hold", "review", "pending", "alternative", "suspicious_activity"]

    if any(kw in prompt_text for kw in fraud_keywords):
        for a in ["flag_fraud", "classify_phishing", "reject_transfer", "reply_block"]:
            if a in available:
                return json.dumps({"type": a, "payload": {}})

    if any(kw in prompt_text for kw in recover_keywords):
        for a in ["hold_for_review", "log_suspicious_activity", "close_pending_review"]:
            if a in available:
                return json.dumps({"type": a, "payload": {}})

    if any(kw in prompt_text for kw in ask_keywords):
        for a in ["request_verification", "escalate_compliance", "verify_metrics_ok", "verify_logs_clean"]:
            if a in available:
                return json.dumps({"type": a, "payload": {}})

    if any(kw in prompt_text for kw in stop_keywords):
        for a in ["reject_transfer", "flag_fraud", "reply_block"]:
            if a in available:
                return json.dumps({"type": a, "payload": {}})

    # Safe defaults by action_type patterns
    safe_prefixes = ["classify_", "diagnose_", "verify_", "log_", "close_", "fix_", "confirm", "archive", "reply_auto"]
    for prefix in safe_prefixes:
        for a in available:
            if a.startswith(prefix):
                return json.dumps({"type": a, "payload": {}})

    return json.dumps({"type": available[0], "payload": {}})


# ─── Environment Interface ────────────────────────────────────────────────────

def _reset(task: str) -> dict:
    r = requests.post(f"{API_BASE}/reset", json={"task": task}, timeout=15)
    r.raise_for_status()
    return r.json()


def _step(action: dict) -> dict:
    r = requests.post(f"{API_BASE}/step", json=action, timeout=15)
    r.raise_for_status()
    return r.json()


# ─── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(task_name: str, mode: str) -> float:
    """
    Run one full episode for a task.
    Prints exact [START] / [STEP] / [END] format.
    Returns final episode score.
    """
    print("[START]")

    obs = _reset(task_name)
    step_idx = 0
    episode_score = 0.0

    while not obs.get("done", False):
        prompt = obs.get("prompt", json.dumps(obs.get("state", {})))

        # Generate action
        if mode == "openai":
            raw_response = _call_openai(prompt)
        else:
            raw_response = _call_rule_based(obs)

        # Parse to dict
        try:
            action_dict = json.loads(raw_response)
            if "type" not in action_dict:
                action_dict = {"type": obs.get("available_actions", ["confirm"])[0], "payload": {}}
        except (json.JSONDecodeError, ValueError):
            # Fallback: pick first available action
            action_dict = {"type": obs.get("available_actions", ["confirm"])[0], "payload": {}}

        action_str = json.dumps(action_dict, separators=(",", ":"))
        print(f"[STEP] step={step_idx} action={action_str}")

        # Submit to environment
        result = _step(action_dict)
        obs = result.get("observation", {})
        reward = result.get("reward", {})
        done = result.get("done", False)
        info = result.get("info", {})

        if done and info.get("episode_score") is not None:
            episode_score = info["episode_score"]

        step_idx += 1

        # Safety: break on max steps
        if step_idx > 10:
            break

    print(f"[END] score={episode_score:.4f}")
    return episode_score


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenEnv Inference Script — Autonomy Calibration Environment"
    )
    parser.add_argument(
        "--mode",
        choices=["openai", "rule_based"],
        default="openai" if OPENAI_API_KEY else "rule_based",
        help="Inference backend (default: openai if OPENAI_API_KEY is set, else rule_based)",
    )
    parser.add_argument(
        "--task",
        choices=ALL_TASKS + ["all"],
        default="all",
        help="Which task to run (default: all)",
    )
    args = parser.parse_args()

    if args.mode == "openai" and not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set. Use --mode rule_based or export OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]
    scores = {}

    for task_name in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"TASK: {task_name.upper()} | MODE: {args.mode} | MODEL: {MODEL_NAME if args.mode == 'openai' else 'rule_based'}")
        print(f"{'='*60}")
        score = run_episode(task_name, args.mode)
        scores[task_name] = score

    # Summary
    if len(scores) > 1:
        avg = sum(scores.values()) / len(scores)
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        for t, s in scores.items():
            print(f"  {t:30s}: {s:.4f}")
        print(f"  {'AVERAGE':30s}: {avg:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
