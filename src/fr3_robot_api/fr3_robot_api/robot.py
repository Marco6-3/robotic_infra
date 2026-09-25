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
    def send_action(self, action: Action) -> Action:
        """Validate and submit a policy target, returning its canonical form.

        The returned value is the accepted high-level target for recording,
        not a measurement of the robot or the intermediate interpolated target.
        """

    def close(self) -> None:
        """Release resources; adapters may override this."""
