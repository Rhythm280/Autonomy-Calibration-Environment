"""
utils.py — Shared utilities for the Autonomy Calibration Environment.
"""
from __future__ import annotations

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def clamp(score: float) -> float:
    """
    Clamp a score into the open interval (0.01, 0.99).

    OpenEnv validator requirement: scores must never be exactly 0.0 or 1.0.
    Handles NaN, inf, None, and all float edge cases safely.
    """
    try:
        s = float(score)
    except (TypeError, ValueError):
        return _SCORE_MIN

    # Guard NaN (only NaN != NaN)
    if s != s:
        return _SCORE_MIN

    if s <= 0.0:
        return _SCORE_MIN
    if s >= 1.0:
        return _SCORE_MAX

    s = round(s, 4)

    # Final hard clamp after rounding
    if s <= 0.0:
        return _SCORE_MIN
    if s >= 1.0:
        return _SCORE_MAX

    return s
