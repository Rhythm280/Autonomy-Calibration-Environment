"""tasks/base.py — Abstract base class for all task implementations."""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseTask(ABC):
    task_id: str = "base"
    max_steps: int = 5

    @abstractmethod
    def reset(self, seed: int | None = None): ...

    @abstractmethod
    def step(self, action): ...

    @abstractmethod
    def state(self) -> dict: ...

    @abstractmethod
    def grade_episode(self, history: list) -> float: ...
