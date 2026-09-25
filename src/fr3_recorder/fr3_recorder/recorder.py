"""Build synchronized LeRobot frames from camera/state/action streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from fr3_robot_api.types import Action, CameraFrame, Observation
from fr3_robot_api.gripper import GripperLimits, GripperMapper
from fr3_robot_api.validation import canonical_action

from .synchronizer import nearest_by_timestamp
from .writer import LeRobotDatasetWriter


@dataclass(frozen=True)
class RecorderConfig:
    fps: int = 30
    state_match_tolerance_ns: int = 10_000_000
    wrist_match_tolerance_ns: int = 40_000_000
    action_match_tolerance_ns: int = 40_000_000
    decision_tolerance_ns: int = 40_000_000
    causal: bool = True

    def __post_init__(self):
        if self.fps != 30:
            raise ValueError("v1 dataset timeline is fixed at 30 Hz")
        for name in ("state_match_tolerance_ns", "wrist_match_tolerance_ns",
                     "action_match_tolerance_ns", "decision_tolerance_ns"):
            value = getattr(self, name)
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative tolerance")


def _state_vector(observation: Observation) -> np.ndarray:
    return np.concatenate(
        [
            observation.joint_position_rad,
            observation.joint_velocity_rad_s,
            observation.ee_pose.position_m,
            observation.ee_pose.quaternion_xyzw,
            np.array([observation.gripper_width_m, observation.gripper_width_normalized]),
        ]
    ).astype(np.float32)


class SynchronizedRecorder:
    """Use external-camera time as the 30 Hz frame clock."""

    def __init__(self, writer: LeRobotDatasetWriter, config: RecorderConfig = RecorderConfig(),
                 gripper_mapper: Optional[GripperMapper] = None) -> None:
        self.writer = writer
        self.config = config
        self._mapper = gripper_mapper or GripperMapper(GripperLimits(0, 0.04, 0, 0.04))
        self._last_recorded_ns = None

    def reset(self) -> None:
        """Call after saving an episode, before resetting the simulation clock."""
        self._last_recorded_ns = None

    def make_frame(
        self,
        external_frame: CameraFrame,
        wrist_frames: Iterable[CameraFrame],
        states: Iterable[Observation],
        action: Action,
        *,
        decision_timestamp_ns: Optional[int] = None,
    ) -> Optional[Mapping[str, Any]]:
        action = canonical_action(action, self._mapper)
        decision_ns = external_frame.timestamp_ns if decision_timestamp_ns is None else decision_timestamp_ns
        if not 0 <= decision_ns - external_frame.timestamp_ns <= self.config.decision_tolerance_ns:
            return None
        if not 0 <= decision_ns - action.timestamp_ns <= self.config.action_match_tolerance_ns:
            return None
        wrist = nearest_by_timestamp(
            wrist_frames,
            external_frame.timestamp_ns,
            max_delta_ns=self.config.wrist_match_tolerance_ns,
            causal=self.config.causal,
        )
        state = nearest_by_timestamp(
            states,
            external_frame.timestamp_ns,
            timestamp=lambda item: item.state_timestamp_ns
            if item.state_timestamp_ns is not None
            else item.timestamp_ns,
            max_delta_ns=self.config.state_match_tolerance_ns,
            causal=self.config.causal,
        )
        if wrist is None or state is None:
            return None
        if action.gripper_width_m is None:
            raise ValueError("recorder expects physical gripper_width_m on each action")
        normalized = action.gripper_width_normalized
        if normalized is None:
            raise ValueError("recorder expects normalized gripper_width_normalized on each action")
        return {
            "observation.state": _state_vector(state),
            "action": np.concatenate(
                [
                    action.arm_joint_position_target_rad,
                    np.array([action.gripper_width_m, normalized]),
                ]
            ).astype(np.float32),
            "observation.images.external": external_frame.rgb,
            "observation.images.wrist": wrist.rgb,
            "timestamp_ns": np.array([external_frame.timestamp_ns], dtype=np.int64),
            "action_timestamp_ns": np.array([action.timestamp_ns], dtype=np.int64),
            "decision_timestamp_ns": np.array([decision_ns], dtype=np.int64),
            "state_timestamp_ns": np.array(
                [
                    state.state_timestamp_ns
                    if state.state_timestamp_ns is not None
                    else state.timestamp_ns
                ],
                dtype=np.int64,
            ),
            "external_timestamp_ns": np.array([external_frame.timestamp_ns], dtype=np.int64),
            "wrist_timestamp_ns": np.array([wrist.timestamp_ns], dtype=np.int64),
        }

    def record_frame(
        self,
        external_frame: CameraFrame,
        wrist_frames: Iterable[CameraFrame],
        states: Iterable[Observation],
        action: Action,
        *,
        decision_timestamp_ns: Optional[int] = None,
    ) -> bool:
        if self._last_recorded_ns is not None and external_frame.timestamp_ns <= self._last_recorded_ns:
            raise ValueError("frame timestamps must increase; reset recorder between episodes")
        frame = self.make_frame(external_frame, wrist_frames, states, action,
                                decision_timestamp_ns=decision_timestamp_ns)
        if frame is None:
            return False
        self.writer.add_frame(frame)
        self._last_recorded_ns = external_frame.timestamp_ns
        return True
