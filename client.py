"""
client.py — OpenEnv-Compliant Client for Autonomy Calibration
Standardized for OpenEnv Core v0.2.x.
"""

from openenv.core.env_client import EnvClient
from models import Action, Observation, StepResult, Reward, TaskState
import requests

class AutonomyCalibrationClient(EnvClient):
    """
    Standardized client implementation for the Autonomy Calibration Hub.
    Implements all abstract methods required by OpenEnv v0.2.x.
    """
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        super().__init__(base_url=base_url)

    def _parse_state(self, response_data: dict) -> Observation:
        """Required by OpenEnv Core: converts dict to Observation model."""
        return Observation(**response_data)

    def _parse_result(self, response_data: dict) -> StepResult:
        """Required by OpenEnv Core: Converts dict to StepResult model."""
        # Use our Pydantic model for validation
        return StepResult(**response_data)

    def _step_payload(self, action: str) -> dict:
        """Required by OpenEnv Core: formats action for the POST request."""
        return {"type": action, "payload": {}}

    def reset_env(self, task: str = "email_triage", seed: int = None) -> Observation:
        """Compliant reset method."""
        resp = requests.post(f"{self.base_url}/api/reset", json={"task": task, "seed": seed})
        return self._parse_state(resp.json())

    def step_env(self, action: str) -> StepResult:
        """Compliant step method."""
        payload = self._step_payload(action)
        resp = requests.post(f"{self.base_url}/api/step", json=payload)
        return self._parse_result(resp.json())
