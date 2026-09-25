"""Small, ROS-independent data contracts shared by simulation and hardware."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array.copy()


@dataclass(frozen=True)
class Pose:
    """Rigid pose represented as xyz + xyzw quaternion."""

    position_m: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vector(self.position_m, 3, "position_m"))
        object.__setattr__(
            self,
            "quaternion_xyzw",
            _vector(self.quaternion_xyzw, 4, "quaternion_xyzw"),
        )


@dataclass(frozen=True)
class CameraFrame:
    """An RGB frame and the simulator/source timestamp that produced it."""

    timestamp_ns: int
    rgb: Any
    camera_name: str

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not self.camera_name:
            raise ValueError("camera_name must not be empty")


@dataclass(frozen=True)
class Action:
    """Policy action at the 30 Hz boundary."""

    timestamp_ns: int
    arm_joint_position_target_rad: np.ndarray
    gripper_width_m: Optional[float] = None
    gripper_width_normalized: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        object.__setattr__(
            self,
            "arm_joint_position_target_rad",
            _vector(self.arm_joint_position_target_rad, 7, "arm_joint_position_target_rad"),
        )
        if self.gripper_width_m is None and self.gripper_width_normalized is None:
            raise ValueError("one gripper representation is required")


@dataclass(frozen=True)
class Observation:
    """Synchronized state exposed to a policy or recorder."""

    timestamp_ns: int
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    ee_pose: Pose
    gripper_width_m: float
    gripper_width_normalized: float
    external_rgb: CameraFrame
    wrist_rgb: CameraFrame
    state_timestamp_ns: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        object.__setattr__(self, "joint_position_rad", _vector(self.joint_position_rad, 7, "joint_position_rad"))
        object.__setattr__(self, "joint_velocity_rad_s", _vector(self.joint_velocity_rad_s, 7, "joint_velocity_rad_s"))
        if not 0.0 <= self.gripper_width_normalized <= 1.0:
            raise ValueError("gripper_width_normalized must be in [0, 1]")
        if self.gripper_width_m < 0.0:
            raise ValueError("gripper_width_m must be non-negative")
