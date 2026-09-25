"""Policy-rate to low-level-rate action bridge.

The class is deliberately ROS-independent. A ROS node can call ``submit`` at
30 Hz and ``step`` from the ros2_control/MuJoCo update callback at 1 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from fr3_robot_api.gripper import GripperMapper
from fr3_robot_api.timing import TargetInterpolator
from fr3_robot_api.types import Action
from fr3_robot_api.validation import canonical_action


@dataclass(frozen=True)
class LowLevelCommand:
    timestamp_ns: int
    arm_joint_position_target_rad: np.ndarray
    gripper_actuated_joint_position_target_m: float


class PolicyTargetBridge:
    """Hold/interpolate 30 Hz policy targets at a 1 kHz update boundary."""

    def __init__(
        self,
        gripper_mapper: GripperMapper,
        interpolation: str = "linear",
        publish: Optional[Callable[[LowLevelCommand], None]] = None,
        arm_limits=None,
    ) -> None:
        self._arm = TargetInterpolator(7, interpolation)
        self._gripper = TargetInterpolator(1, interpolation)
        self._mapper = gripper_mapper
        self._publish = publish
        self._last_command: Optional[LowLevelCommand] = None
        self._interpolation = interpolation
        self._arm_limits = arm_limits
        self._last_action: Optional[Action] = None

    def submit(self, action: Action) -> Action:
        action = canonical_action(action, self._mapper, self._arm_limits)
        if self._last_action is not None and action.timestamp_ns <= self._last_action.timestamp_ns:
            raise ValueError("policy timestamps must be strictly increasing")
        self._arm.set_target(action.timestamp_ns, action.arm_joint_position_target_rad)
        if action.gripper_width_m is not None:
            width_m = action.gripper_width_m
        else:
            assert action.gripper_width_normalized is not None
            width_m = self._mapper.width_from_normalized(action.gripper_width_normalized)
        self._gripper.set_target(action.timestamp_ns, np.array([width_m], dtype=np.float64))
        self._last_action = action
        return action

    @property
    def has_target(self) -> bool:
        return self._last_action is not None

    def reset(self) -> None:
        """Discard commands from the previous simulation epoch."""
        self._arm = TargetInterpolator(7, self._interpolation)
        self._gripper = TargetInterpolator(1, self._interpolation)
        self._last_action = None
        self._last_command = None

    def step(self, timestamp_ns: int) -> LowLevelCommand:
        if self._last_action is None:
            raise RuntimeError("no policy target has been set")
        if timestamp_ns < self._last_action.timestamp_ns:
            raise ValueError("cannot apply a future target; check the simulation clock")
        arm = self._arm.evaluate(timestamp_ns)
        width = float(self._gripper.evaluate(timestamp_ns)[0])
        command = LowLevelCommand(
            timestamp_ns=timestamp_ns,
            arm_joint_position_target_rad=arm,
            gripper_actuated_joint_position_target_m=float(self._mapper.joints_from_width(width)[0]),
        )
        self._last_command = command
        if self._publish is not None:
            self._publish(command)
        return command

    @property
    def last_command(self) -> Optional[LowLevelCommand]:
        return self._last_command
