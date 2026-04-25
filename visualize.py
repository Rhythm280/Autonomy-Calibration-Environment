import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_winning_plots(rewards_per_episode, losses, baseline_score, trained_scores):
    """
    Generates the premium, judge-ready plots for the Autonomy Calibration Benchmark.
    """
    
    # 1. Reward Curve (Calibration Accuracy)
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, color='#27AE60', linewidth=2.5, alpha=0.3)
    # Smoothed trend
    smooth_rewards = np.convolve(rewards_per_episode, np.ones(10)/10, mode='valid')
    plt.plot(range(9, len(rewards_per_episode)), smooth_rewards, color='#1A8A4A', linewidth=3, label='Calibrated Policy Reward')
    
    plt.axhline(y=baseline_score, color='#E74C3C', linestyle='--', linewidth=2, label=f'Rule-Based Baseline ({baseline_score})')
    plt.title('🛡️ Autonomy Calibration: Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Training Episode')
    plt.ylabel('Episode Reward (0.01 - 0.99)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('reward_curve.png', dpi=150)
    plt.close()

    # 2. Policy Loss (Divergence)
    plt.figure(figsize=(10, 5))
    plt.plot(losses, color='#2980B9', linewidth=2)
    plt.title('📈 GRPOTrainer Policy Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    plt.close()

    # 3. Final Comparison (The ROI)
    plt.figure(figsize=(8, 6))
    categories = ['Rule-Based Baseline', 'Trained Agent (GRPO)']
    values = [baseline_score, np.mean(trained_scores)]
    
    bars = plt.bar(categories, values, color=['#BDC3C7', '#2ECC71'], width=0.6)
    plt.title('🏆 Performance Uplift: Accuracy + Calibration', fontsize=14, fontweight='bold')
    plt.ylabel('Average Episode Reward')
    plt.ylim(0, 1.0)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('baseline_vs_trained.png', dpi=150)
    plt.close()
    print("✅ Winning visuals generated: reward_curve.png, loss_curve.png, baseline_vs_trained.png")
