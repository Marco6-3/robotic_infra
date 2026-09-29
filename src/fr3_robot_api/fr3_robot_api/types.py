"""Small, ROS-independent data contracts shared by simulation and hardware."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
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
class TactileFrame:
    """Timestamped low-dimensional tactile/contact channels or privileged labels."""

    timestamp_ns: int
    values: np.ndarray
    channel_names: tuple[str, ...]
    source_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("tactile values must be a non-empty vector")
        if not np.all(np.isfinite(values)):
            raise ValueError("tactile values must contain only finite values")
        if len(self.channel_names) != values.size:
            raise ValueError("channel_names length must match tactile values")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("tactile channel names must be unique")
        if not self.source_name:
            raise ValueError("source_name must not be empty")
        object.__setattr__(self, "values", values.copy())


@dataclass(frozen=True)
class VisionTactileFrame:
    """Raw RGB frame from one camera-based tactile sensor."""

    timestamp_ns: int
    rgb: Any
    sensor_name: str
    sequence: int
    arrival_timestamp_ns: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0 or self.sequence < 0:
            raise ValueError("timestamp_ns and sequence must be non-negative")
        if self.arrival_timestamp_ns is not None and self.arrival_timestamp_ns < self.timestamp_ns:
            raise ValueError("arrival_timestamp_ns cannot precede the source timestamp")
        if not self.sensor_name:
            raise ValueError("sensor_name must not be empty")
        image = np.asarray(self.rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("vision tactile RGB must be uint8 HxWx3")
        object.__setattr__(self, "rgb", image.copy())


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
        if self.gripper_width_m is not None and (
            not np.isfinite(self.gripper_width_m) or self.gripper_width_m < 0
        ):
            raise ValueError("gripper_width_m must be finite and non-negative")
        if self.gripper_width_normalized is not None and (
            not np.isfinite(self.gripper_width_normalized)
            or not 0 <= self.gripper_width_normalized <= 1
        ):
            raise ValueError("gripper_width_normalized must be finite and in [0, 1]")


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
    tactile: Optional[TactileFrame] = None
    vision_tactile: tuple[VisionTactileFrame, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        object.__setattr__(self, "joint_position_rad", _vector(self.joint_position_rad, 7, "joint_position_rad"))
        object.__setattr__(self, "joint_velocity_rad_s", _vector(self.joint_velocity_rad_s, 7, "joint_velocity_rad_s"))
        if not 0.0 <= self.gripper_width_normalized <= 1.0:
            raise ValueError("gripper_width_normalized must be in [0, 1]")
        if not np.isfinite(self.gripper_width_m) or self.gripper_width_m < 0.0:
            raise ValueError("gripper_width_m must be finite and non-negative")
        vision_tactile = tuple(self.vision_tactile)
        sensor_names = tuple(frame.sensor_name for frame in vision_tactile)
        if len(sensor_names) != len(set(sensor_names)):
            raise ValueError("vision tactile sensor names must be unique")
        if any(frame.timestamp_ns > self.timestamp_ns for frame in vision_tactile):
            raise ValueError("vision tactile frames cannot come from the future")
        object.__setattr__(self, "vision_tactile", vision_tactile)
