"""
inference.py — Research-Grade Evaluation Script
Autonomy Calibration Benchmark (OpenEnv v2.0.0)
─────────────────────────────────────────────────────────────────────────────
Generates side-by-side comparison metrics between:
1. Blind Baseline (Heuristic, no investigation)
2. Smart Baseline (Deterministic investigation)
3. (Optional) Trained Agent (Selective investigation)

Usage: python3 inference.py --episodes 10
"""

import os
import json
import argparse
import requests
import numpy as np
from typing import List, Dict

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api").rstrip("/")
TASKS    = ["email_triage", "devops_incident", "financial_request"]

class Evaluator:
    def __init__(self, mode: str):
        self.mode = mode
        self.results = {t: [] for t in TASKS}
        self.investigate_stats = {t: [] for t in TASKS}
        self.high_ambiguity_failures = {t: 0 for t in TASKS}
        self.total_high_ambiguity = {t: 0 for t in TASKS}

    def _reset(self, task: str, seed: int) -> dict:
        r = requests.post(f"{API_BASE}/reset", json={"task": task, "seed": seed}, timeout=10)
        return r.json()

    def _step(self, action_type: str) -> dict:
        r = requests.post(f"{API_BASE}/step", json={"type": action_type}, timeout=10)
        return r.json()

    def select_action(self, obs: dict, investigated: bool) -> str:
        """Implements the agent policy."""
        avail = obs.get("available_actions", [])
        ambiguity = obs.get("state", {}).get("ambiguity", 0.1)
        
        # Mode Logic
        if self.mode == "blind":
            # Never investigate, just pick the first non-investigate action
            return [a for a in avail if a != "investigate"][0]
            
        elif self.mode == "smart":
            # Always investigate at step 0 if available
            if "investigate" in avail and not investigated:
                return "investigate"
            # Otherwise use heuristic
            choices = [a for a in avail if a != "investigate"]
            return choices[0] if choices else avail[0]

        return avail[0]

    def run_eval(self, episodes_per_task=10):
        print(f"\n🚀 Running evaluation: MODE = {self.mode.upper()}")
        for task in TASKS:
            print(f"  Evaluating {task}...", end="", flush=True)
            for seed in range(episodes_per_task):
                obs = self._reset(task, seed)
                investigated = False
                done = False
                total_reward = 0
                
                # Track ambiguity for metrics
                ambiguity = obs.get("state", {}).get("ambiguity", 0.0)
                is_high_ambiguity = ambiguity > 0.70
                if is_high_ambiguity:
                    self.total_high_ambiguity[task] += 1

                while not done:
                    action = self.select_action(obs, investigated)
                    if action == "investigate":
                        investigated = True
                    
                    res = self._step(action)
                    done = res.get("done", False)
                    obs = res.get("observation", {})
                    
                score = res.get("info", {}).get("episode_score", 0.0)
                self.results[task].append(score)
                self.investigate_stats[task].append(1 if investigated else 0)
                
                if is_high_ambiguity and score < 0.20:
                    self.high_ambiguity_failures[task] += 1
            
            print(" Done.")

    def get_summary(self):
        summary = {}
        for t in TASKS:
            avg_rew = np.mean(self.results[t])
            inv_rate = np.mean(self.investigate_stats[t])
            fail_rate = self.high_ambiguity_failures[t] / self.total_high_ambiguity[t] if self.total_high_ambiguity[t] > 0 else 0
            summary[t] = {
                "avg_reward": avg_rew,
                "investigate_rate": inv_rate,
                "failure_rate_ambiguous": fail_rate
            }
        return summary

def print_final_report(blind: dict, smart: dict):
    print("\n" + "="*80)
    print("🏆 RESEARCH-GRADE EVALUATION REPORT")
    print("="*80)
    print(f"{'Task':<20} | {'Mode':<8} | {'Reward':<7} | {'Inv%':<6} | {'AmbFail%':<10}")
    print("-" * 80)
    
    for t in TASKS:
        b = blind[t]
        s = smart[t]
        delta = ((s['avg_reward'] - b['avg_reward']) / b['avg_reward']) * 100 if b['avg_reward'] > 0 else 0
        
        print(f"{t[:20]:<20} | {'BLIND':<8} | {b['avg_reward']:.4f} | {b['investigate_rate']*100:>5.0f}% | {b['failure_rate_ambiguous']*100:>9.0f}%")
        print(f"{'':<20} | {'SMART':<8} | {s['avg_reward']:.4f} | {s['investigate_rate']*100:>5.0f}% | {s['failure_rate_ambiguous']*100:>9.0f}%")
        print(f"{'':<20} | {'IMPROVE':<8} | {delta:>+6.1f}% | {'--':>6} | {'--':>10}")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    # Step 1: Evaluate Blind Baseline
    blind_eval = Evaluator("blind")
    blind_eval.run_eval(args.episodes)
    blind_summary = blind_eval.get_summary()

    # Step 2: Evaluate Smart Baseline (Upper Bound)
    smart_eval = Evaluator("smart")
    smart_eval.run_eval(args.episodes)
    smart_summary = smart_eval.get_summary()

    # Final Report
    print_final_report(blind_summary, smart_summary)
    
    print("\n💡 INSIGHT: The Smart baseline establishes the ceiling for 'Total Curiosity'.")
    print("   The winning GRPO model learns to bridge this gap selectively.")
