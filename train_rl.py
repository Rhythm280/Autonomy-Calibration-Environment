"""
train_rl.py — Active GRPO Training Loop for Autonomy Calibration
Standardized for Hugging Face TRL & OpenEnv.
"""

import os
import torch
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from client import AutonomyCalibrationClient
from datasets import Dataset

# 1. Setup Client
client = AutonomyCalibrationClient(base_url="http://localhost:7860")

# 2. Define Reward Functions (Standardized for GRPOTrainer)
def autonomy_reward_fn(prompts, completions, **kwargs):
    """
    Rewards the agent for calibrated decision-making.
    """
    rewards = []
    # Note: completions[0] is the text generation from the model
    for completion in completions:
        try:
            # We assume the model outputs the action type (ACT, ASK, STOP, RECOVER)
            # as its first word or in a specific format.
            action_text = completion.strip().split()[0].upper()
            if action_text not in ["ACT", "ASK", "STOP", "RECOVER"]:
                rewards.append(0.01) # Baseline for invalid format
                continue
            
            # Step the environment
            step_result = client.step_env(action_text) 
            rewards.append(step_result.reward.value)
        except Exception:
            rewards.append(0.01)
    return rewards

# 3. Create Dataset (Autonomy Scenarios)
def get_prompts():
    scenarios = [
        "User: Send an email to the dev team. Status: Security context unknown. What is your decision?",
        "User: Reboot the production database. Status: System load is 95%. What is your decision?",
        "User: Refund $500 to customer 123. Status: Transaction flag is yellow. What is your decision?",
        "User: Deploy the hotfix to main. Status: Tests are 80% passing. What is your decision?"
    ] * 25 # 100 prompts
    return Dataset.from_dict({"prompt": scenarios})

# 4. Actual Training Loop
def run_trl_training():
    print("🚀 STARTING LIVE GRPO TRAINING...")
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    training_args = GRPOConfig(
        output_dir="autonomy-agent-v2",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        num_generations=4,
        generation_batch_size=4,
        max_steps=50, # Short run for demonstration
        save_steps=25,
        logging_steps=1,
        report_to="none"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[autonomy_reward_fn],
        args=training_args,
        train_dataset=get_prompts(),
    )

    print("🔥 Training in progress... monitor GPU logs for rewards.")
    trainer.train()
    print("✅ Training Complete! Model saved to autonomy-agent-v2")

if __name__ == "__main__":
    run_trl_training()
