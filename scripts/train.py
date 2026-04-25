"""
train.py — Autonomy Calibration Training & Trajectory Collection
This script runs agents through the environment, collects rewards, 
and prepares datasets for Hugging Face integration.
"""
import os
import time
import json
import random
import requests
from typing import List, Dict, Any
from huggingface_hub import HfApi, login

# --- CONFIGURATION ---
API_BASE = "http://localhost:8000/api"
TASKS = ["email_triage", "devops_incident", "financial_request"]
NUM_EPISODES = 10
HF_REPO_NAME = "autonomy-calibration-results" # Change to your HF username/repo

class TrainingAgent:
    """A simple agent that can be extended to use LLMs or Neural Networks."""
    def __init__(self, name: str = "baseline_agent"):
        self.name = name

    def select_action(self, observation: Dict[str, Any]) -> str:
        # Heuristic: If it's a financial request and signals look bad, STOP.
        # Otherwise, pick a random available action for exploration.
        avail = observation.get("available_actions", [])
        if not avail: return ""
        
        prompt = observation.get("prompt", "").lower()
        if "fraud" in prompt or "suspicious" in prompt:
            for a in avail:
                if "reject" in a or "flag" in a: return a
                
        return random.choice(avail)

def run_training_cycle():
    print(f"🚀 Starting Autonomy Training Cycle...")
    agent = TrainingAgent()
    trajectories = []

    for i in range(NUM_EPISODES):
        task_name = random.choice(TASKS)
        print(f"--- Episode {i+1}/{NUM_EPISODES} | Task: {task_name} ---")
        
        # 1. Reset
        res = requests.post(f"{API_BASE}/reset", json={"task": task_name})
        if res.status_code != 200:
            print(f"❌ Reset failed: {res.text}")
            continue
            
        obs = res.json()
        done = False
        episode_reward = 0.0
        steps = []

        # 2. Step Loop
        while not done:
            action_type = agent.select_action(obs)
            step_res = requests.post(f"{API_BASE}/step", json={"type": action_type})
            
            if step_res.status_code != 200:
                print(f"❌ Step failed: {step_res.text}")
                break
                
            data = step_res.json()
            reward = data["reward"]["value"]
            episode_reward += reward
            
            steps.append({
                "observation": obs["prompt"],
                "action": action_type,
                "reward": reward,
                "breakdown": data["reward"]["breakdown"]
            })
            
            obs = data["observation"]
            done = data["done"]

        print(f"🏁 Episode Finished. Total Reward: {episode_reward:.2f}")
        trajectories.append({
            "episode": i,
            "task": task_name,
            "total_reward": episode_reward,
            "steps": steps
        })

    # 3. Save locally
    with open("training_trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"💾 Trajectories saved to training_trajectories.json")

    # 4. Integrate with Hugging Face (Optional)
    if os.getenv("HF_TOKEN"):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj="training_trajectories.json",
                path_in_repo=f"results_{int(time.time())}.json",
                repo_id=HF_REPO_NAME,
                repo_type="dataset",
                token=os.getenv("HF_TOKEN")
            )
            print(f"📤 Results successfully pushed to Hugging Face Dataset: {HF_REPO_NAME}")
        except Exception as e:
            print(f"⚠️ Hugging Face push failed: {e}")

if __name__ == "__main__":
    # Ensure uvicorn is running before starting
    try:
        run_training_cycle()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the environment server.")
        print("💡 Make sure uvicorn is running: uvicorn main:app --port 8000")
