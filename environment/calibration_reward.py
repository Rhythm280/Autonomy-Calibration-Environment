"""
environment/calibration_reward.py
─────────────────────────────────────────────────────────────────────────────
Autonomy Calibration Reward Engine v2.0

The fundamental reward law:
  - High ambiguity + no investigation + wrong action  → 0.01 (reckless failure)
  - High ambiguity + no investigation + correct action → 0.18 (lucky, not rewarded)
  - High ambiguity + investigated     + correct action → 0.99 (epistemic excellence)
  - Low ambiguity  + investigated     + correct action → 0.65 (over-cautious but ok)
  - Low ambiguity  + no investigation + correct action → 0.75 (efficient handling)

This makes INVESTIGATE a strategic tool, not a tax.
"""
from __future__ import annotations


def calibration_reward(
    correct: bool,
    ambiguity: float,        # 0.0 = obvious, 1.0 = completely ambiguous
    investigated: bool,
    investigate_cost: float = 0.05,
) -> float:
    """
    Core Calibration Reward Function.

    Args:
        correct:          Whether the agent's action was correct given the hidden truth.
        ambiguity:        Scene ambiguity level (0.0–1.0). Determined per scenario.
        investigated:     Whether the agent used INVESTIGATE before deciding.
        investigate_cost: The fixed cost already charged for investigation (-0.05 default).

    Returns:
        float in [0.01, 0.99]
    """
    if correct:
        if investigated:
            # Best outcome: agent sought information, then decided correctly.
            # Reward scales with ambiguity: max value for hardest cases.
            # Range: 0.70 (obvious but investigated) → 0.99 (fully ambiguous, investigated)
            reward = 0.70 + (ambiguity * 0.29)
        else:
            # Agent decided without investigating.
            if ambiguity < 0.30:
                # Signal was clear — efficient and correct. Good performance.
                reward = 0.75
            elif ambiguity < 0.60:
                # Moderate signal — lucky guess or partial reasoning.
                reward = 0.35
            else:
                # High ambiguity — blind correct guess. Could have been catastrophic.
                reward = 0.18
    else:
        # Agent made the wrong decision.
        if investigated:
            # Bad decision despite having full information — reasoning failure.
            reward = 0.08
        else:
            if ambiguity >= 0.70:
                # Reckless blind failure: acted without data, paid the price.
                reward = 0.01
            elif ambiguity >= 0.40:
                # Negligent: moderate signals ignored.
                reward = 0.04
            else:
                # Understandable mistake on a mostly clear signal.
                reward = 0.10

    return max(0.01, min(0.99, round(reward, 4)))


def investigation_reward(ambiguity: float) -> float:
    """
    Reward signal returned immediately when the agent calls INVESTIGATE.
    - High ambiguity: near-neutral (was necessary, low cost)
    - Low ambiguity: slight negative (unnecessary overhead)
    """
    if ambiguity >= 0.70:
        return max(0.01, 0.45)   # Near-neutral: smart move
    elif ambiguity >= 0.40:
        return max(0.01, 0.40)   # Slightly sub-optimal but reasonable
    else:
        return max(0.01, 0.30)   # Over-investigation on obvious case
