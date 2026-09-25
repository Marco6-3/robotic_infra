"""Franka Hand continuous-width mapping.

The limits are supplied by the pinned description instead of being embedded in
the policy interface. For the official description each prismatic finger is
limited independently; the physical opening is the sum of the two displacements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GripperLimits:
    finger1_lower_m: float
    finger1_upper_m: float
    finger2_lower_m: float
    finger2_upper_m: float

    @property
    def width_lower_m(self) -> float:
        return self.finger1_lower_m + self.finger2_lower_m

    @property
    def width_upper_m(self) -> float:
        return self.finger1_upper_m + self.finger2_upper_m

    def __post_init__(self) -> None:
        if self.finger1_upper_m < self.finger1_lower_m or self.finger2_upper_m < self.finger2_lower_m:
            raise ValueError("finger upper limits must be >= lower limits")
        if self.width_upper_m <= self.width_lower_m:
            raise ValueError("gripper width range must be non-empty")


@dataclass(frozen=True)
class GripperMapper:
    limits: GripperLimits

    def width_from_joints(self, joints_m: np.ndarray) -> float:
        joints = np.asarray(joints_m, dtype=np.float64)
        if joints.shape != (2,):
            raise ValueError(f"gripper joints must have shape (2,), got {joints.shape}")
        return float(joints[0] + joints[1])

    def normalized_from_width(self, width_m: float) -> float:
        width = float(np.clip(width_m, self.limits.width_lower_m, self.limits.width_upper_m))
        return (width - self.limits.width_lower_m) / (
            self.limits.width_upper_m - self.limits.width_lower_m
        )

    def width_from_normalized(self, normalized: float) -> float:
        value = float(np.clip(normalized, 0.0, 1.0))
        return self.limits.width_lower_m + value * (
            self.limits.width_upper_m - self.limits.width_lower_m
        )

    def joints_from_width(self, width_m: float) -> np.ndarray:
        width = float(np.clip(width_m, self.limits.width_lower_m, self.limits.width_upper_m))
        # Franka Hand is symmetric; each finger travels half of the physical opening.
        return np.array([width / 2.0, width / 2.0], dtype=np.float64)

    def joints_from_normalized(self, normalized: float) -> np.ndarray:
        return self.joints_from_width(self.width_from_normalized(normalized))
