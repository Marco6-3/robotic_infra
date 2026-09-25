"""Build synchronized LeRobot frames from camera/state/action streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from fr3_robot_api.types import Action, CameraFrame, Observation

from .synchronizer import nearest_by_timestamp
from .writer import LeRobotDatasetWriter


@dataclass(frozen=True)
class RecorderConfig:
    fps: int = 30
    state_match_tolerance_ns: Optional[int] = None
    wrist_match_tolerance_ns: Optional[int] = None


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

    def __init__(self, writer: LeRobotDatasetWriter, config: RecorderConfig = RecorderConfig()) -> None:
        self.writer = writer
        self.config = config

    def make_frame(
        self,
        external_frame: CameraFrame,
        wrist_frames: Iterable[CameraFrame],
        states: Iterable[Observation],
        action: Action,
    ) -> Optional[Mapping[str, Any]]:
        wrist = nearest_by_timestamp(
            wrist_frames,
            external_frame.timestamp_ns,
            max_delta_ns=self.config.wrist_match_tolerance_ns,
        )
        state = nearest_by_timestamp(
            states,
            external_frame.timestamp_ns,
            timestamp=lambda item: item.state_timestamp_ns
            if item.state_timestamp_ns is not None
            else item.timestamp_ns,
            max_delta_ns=self.config.state_match_tolerance_ns,
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
    ) -> bool:
        frame = self.make_frame(external_frame, wrist_frames, states, action)
        if frame is None:
            return False
        self.writer.add_frame(frame)
        return True
