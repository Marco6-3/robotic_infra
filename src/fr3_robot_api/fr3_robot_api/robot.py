"""High-level Robot protocol.

No ROS message, topic, controller, or simulator type appears in this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import Action, Observation


class Robot(ABC):
    """Interface implemented by the MuJoCo and future real-robot adapters."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Return the latest synchronized policy observation."""

    @abstractmethod
    def send_action(self, action: Action) -> None:
        """Submit a 30 Hz policy action to the robot."""

    def close(self) -> None:
        """Release resources; adapters may override this."""
