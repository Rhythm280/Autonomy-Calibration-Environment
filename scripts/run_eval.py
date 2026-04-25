import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from tasks.email_triage import EmailTriageTask
from tasks.devops_incident import DevOpsIncidentTask
from tasks.financial_request import FinancialRequestTask
from models import Action

def run_evaluation(num_episodes=100):
    tasks = [EmailTriageTask, DevOpsIncidentTask, FinancialRequestTask]
    results = []
    
    print(f"Running {num_episodes} random agent episodes...")
    for i in range(num_episodes):
        TaskClass = random.choice(tasks)
        task = TaskClass()
        obs = task.reset()
        
        episode_rewards = []
        episode_actions = []
        episode_ambiguity = task._scenario.get("ambiguity", 0.5)
        investigated = False
        
        done = False
        while not done:
            actions = obs.available_actions
            if not actions:
                break
            action_type = random.choice(actions)
            if action_type == "investigate":
                investigated = True
            
            obs, reward, done, info = task.step(Action(type=action_type))
            episode_rewards.append(reward.value)
            episode_actions.append(action_type)
        
        results.append({
            "task": TaskClass.__name__,
            "total_reward": sum(episode_rewards),
            "avg_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "actions": episode_actions,
            "ambiguity": episode_ambiguity,
            "investigated": investigated
        })
        if (i+1) % 20 == 0:
            print(f"Progress: {i+1}/{num_episodes}")
            
    return results

def plot_results(results):
    rewards = [r["avg_reward"] for r in results]
    all_actions = []
    for r in results:
        all_actions.extend(r["actions"])
    
    # 1. Reward Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, color='#3498DB', edgecolor='white')
    plt.title('Random Agent Reward Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Average Step Reward')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.savefig('reward_dist.png', dpi=150)
    plt.close()
    
    # 2. Action Distribution
    from collections import Counter
    counts = Counter(all_actions)
    labels, values = zip(*counts.most_common(10))
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='#2ECC71')
    plt.title('Top 10 Actions Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('action_dist.png', dpi=150)
    plt.close()
    
    # 3. Ambiguity vs Investigation
    ambiguities = [r["ambiguity"] for r in results]
    investigated = [1 if r["investigated"] else 0 for r in results]
    
    # Bin ambiguity and calculate investigation rate
    bins = np.linspace(0, 1, 6)
    bin_indices = np.digitize(ambiguities, bins)
    bin_rates = []
    bin_centers = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.any(mask):
            bin_rates.append(np.mean(np.array(investigated)[mask]))
            bin_centers.append((bins[i-1] + bins[i]) / 2)
            
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, bin_rates, 'o-', color='#E74C3C', linewidth=2, markersize=8)
    plt.title('Ambiguity vs Investigation Rate (Random Policy)', fontsize=14, fontweight='bold')
    plt.xlabel('Scenario Ambiguity')
    plt.ylabel('Investigation Probability')
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.savefig('ambiguity_investigation.png', dpi=150)
    plt.close()
    
    print("✅ Evaluation plots generated: reward_dist.png, action_dist.png, ambiguity_investigation.png")

if __name__ == "__main__":
    results = run_evaluation(100)
    plot_results(results)
