"""LeRobot-compatible recording primitives."""

from .schema import build_lerobot_features
from .synchronizer import nearest_by_timestamp

__all__ = ["build_lerobot_features", "nearest_by_timestamp"]
