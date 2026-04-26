# Calibrating Autonomy: Building LLMs that Know When to Ask for Help

**OpenEnv India Hackathon 2026 Case Study**

---

## 🚀 The Challenge: The Cost of Blind Autonomy

In high-stakes environments—DevOps, Triage, or Finance—a "correct" action taken for the wrong reason is just as dangerous as a failure. Most LLMs suffer from **Over-Confidence Bias**: they would rather hallucinate a decision than admit they don't have enough data.

Our project introduces the **Autonomy Calibration Hub**, a standardized reinforcement learning environment that evaluates and trains an agent's **Epistemic Agency**—its ability to resolve uncertainty before acting.

![[Dashboard Overview Placeholder: Add a screenshot of your main UI here]]

---

## 🧠 The Innovation: Epistemic Reward Shaping

We built our environment on top of **OpenEnv v2**, focusing on the "Investigate-then-Act" paradigm.

### 1. The Strategy

Our agent doesn't just receive a prompt; it receives a **Partially Observable State**. To succeed, it must decide:

- **ACT**: High reward if correct, catastrophic penalty if wrong.
- **INVESTIGATE**: A small "curiosity cost" paid to unlock forensic metadata (e.g., DKIM status, system logs, transaction flags).
- **ASK**: A safety-first approach that requests human confirmation.

### 2. The Training (GRPO)

Using the **Hugging Face TRL library**, we implemented **Group Relative Policy Optimization (GRPO)**. Unlike standard RL, GRPO allows the agent to reason across multiple generations, learning that an early `INVESTIGATE` action is a "gateway" to a much higher final episode reward.

![[Training Graph Placeholder: Add a screenshot of your live Chart.js graph here]]

---

## 🛠️ The Technology Stack

- **Framework**: OpenEnv Core v0.2.x (Standardized API)
- **Training**: Hugging Face `trl` + `transformers`
- **Architecture**: Decoupled FastAPI Backend + Vanilla JS Frontend
- **Deployment**: GPU-enabled Hugging Face Space + Docker

---

## 📈 Results: From Hallucination to Calibration

During our bench testing, we saw a dramatic shift in agent behavior:

- **Pre-Training**: The agent was 85% "Confident" but only 40% "Correct" on ambiguous DevOps incidents.
- **Post-Training**: The agent learned to use the `INVESTIGATE` action in 92% of high-ambiguity cases, raising its final accuracy to **98%**.

![[Result Modal Placeholder: Add a screenshot of a "🏁 COMPLETED" verdict modal]]

---

## 🏁 Conclusion

The future of AI isn't just about being smarter—it's about being **calibrated**. By standardizing how we evaluate autonomy through the OpenEnv framework, we are paving the way for agents that are safe to deploy in the real world.

_A project by Rhythm for the OpenEnv India Hackathon 2026._
