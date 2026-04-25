"""
client.py — OpenEnv-Compliant Client for Autonomy Calibration
This client provides a strictly separated interface to the environment,
satisfying the hackathon's architectural requirements.
"""

from openenv.core.env_client import EnvClient
from models import Action, Observation, StepResult

class AutonomyCalibrationClient(EnvClient):
    """
    Client for interacting with the Autonomy Calibration Space.
    Inheriting from OpenEnv's EnvClient ensures standard communication protocols.
    """
    def __init__(self, base_url: str = "http://localhost:7860"):
        super().__init__(base_url=base_url)

    def reset_env(self, task: str = "email_triage", seed: int = None) -> Observation:
        """Resets the remote environment and returns a Pydantic Observation."""
        response = self._post("/api/reset", json={"task": task, "seed": seed})
        return Observation(**response.json())

    def step_env(self, action_type: str) -> StepResult:
        """Takes a step in the remote environment and returns a Pydantic StepResult."""
        action = {"type": action_type, "payload": {}}
        response = self._post("/api/step", json=action)
        return StepResult(**response.json())
