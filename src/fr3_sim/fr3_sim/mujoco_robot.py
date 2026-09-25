"""MuJoCo-backed Robot adapter for model and control-loop smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from fr3_control.bridge import PolicyTargetBridge
from fr3_robot_api.gripper import GripperMapper
from fr3_robot_api.robot import Robot
from fr3_robot_api.types import Action, CameraFrame, Observation, Pose


class MujocoUnavailable(RuntimeError):
    pass


class MujocoRobot(Robot):
    """Step a generated FR3+Hand MJCF at exactly one physics tick per call."""

    def __init__(
        self,
        scene_path: Path,
        gripper_mapper: GripperMapper,
        *,
        interpolation: str = "linear",
        camera_width: int = 640,
        camera_height: int = 480,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise MujocoUnavailable("install the pinned MuJoCo Python package") from exc
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        if not np.isclose(self.model.opt.timestep, 0.001):
            raise ValueError(f"expected MuJoCo timestep 0.001 s, got {self.model.opt.timestep}")
        self._arm_actuators = [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"fr3_joint{i}") for i in range(1, 8)]
        self._finger_actuator = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "fr3_finger_joint1")
        self._arm_joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, f"fr3_joint{i}") for i in range(1, 8)]
        self._finger_joints = [
            self._id(mujoco.mjtObj.mjOBJ_JOINT, f"fr3_finger_joint{i}") for i in (1, 2)
        ]
        self._tcp_site = self._id(mujoco.mjtObj.mjOBJ_SITE, "fr3_hand_tcp")
        self._external_camera = self._id(mujoco.mjtObj.mjOBJ_CAMERA, "external")
        self._wrist_camera = self._id(mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        self._renderer = None
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._mapper = gripper_mapper
        self._bridge = PolicyTargetBridge(gripper_mapper, interpolation=interpolation)

    def _id(self, object_type, name: str) -> int:
        value = self._mujoco.mj_name2id(self.model, object_type, name)
        if value < 0:
            raise ValueError(f"MuJoCo model is missing {name!r}")
        return value

    def send_action(self, action: Action) -> None:
        self._bridge.submit(action)

    def step(self) -> None:
        """Advance one 1 ms tick and apply the interpolated low-level target."""
        timestamp_ns = int(round(float(self.data.time) * 1_000_000_000))
        command = self._bridge.step(timestamp_ns)
        for actuator, target in zip(self._arm_actuators, command.arm_joint_position_target_rad):
            self.data.ctrl[actuator] = target
        self.data.ctrl[self._finger_actuator] = command.gripper_actuated_joint_position_target_m
        self._mujoco.mj_step(self.model, self.data)

    def _render(self, camera_id: int) -> np.ndarray:
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(self.model, self._camera_height, self._camera_width)
        self._renderer.update_scene(self.data, camera=camera_id)
        return np.asarray(self._renderer.render()).copy()

    def _pose(self) -> Pose:
        position = self.data.site_xpos[self._tcp_site].copy()
        matrix = self.data.site_xmat[self._tcp_site].reshape(3, 3)
        quaternion_wxyz = np.zeros(4)
        self._mujoco.mju_mat2Quat(quaternion_wxyz, matrix.reshape(-1))
        return Pose(position, [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])

    def get_observation(self) -> Observation:
        timestamp_ns = int(round(float(self.data.time) * 1_000_000_000))
        q = np.array([self.data.qpos[self.model.jnt_qposadr[joint]] for joint in self._arm_joints])
        dq = np.array([self.data.qvel[self.model.jnt_dofadr[joint]] for joint in self._arm_joints])
        fingers = np.array([self.data.qpos[self.model.jnt_qposadr[joint]] for joint in self._finger_joints])
        width = self._mapper.width_from_joints(fingers)
        external = CameraFrame(timestamp_ns, self._render(self._external_camera), "external")
        wrist = CameraFrame(timestamp_ns, self._render(self._wrist_camera), "wrist")
        return Observation(
            timestamp_ns=timestamp_ns,
            state_timestamp_ns=timestamp_ns,
            joint_position_rad=q,
            joint_velocity_rad_s=dq,
            ee_pose=self._pose(),
            gripper_width_m=width,
            gripper_width_normalized=self._mapper.normalized_from_width(width),
            external_rgb=external,
            wrist_rgb=wrist,
        )
