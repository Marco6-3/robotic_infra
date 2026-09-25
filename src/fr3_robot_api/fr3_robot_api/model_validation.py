"""Description-level validation helpers for the pinned FR3 models."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class JointLimit:
    name: str
    lower: float
    upper: float


def parse_mjcf_joint_limits(path: Path) -> dict[str, JointLimit]:
    root = ET.parse(path).getroot()
    result: dict[str, JointLimit] = {}
    for joint in root.findall(".//joint"):
        name, value = joint.get("name"), joint.get("range")
        if not name or not value:
            continue
        lower, upper = (float(part) for part in value.split())
        result[name] = JointLimit(name, lower, upper)
    return result


def parse_urdf_joint_limits(path: Path) -> dict[str, JointLimit]:
    root = ET.parse(path).getroot()
    result: dict[str, JointLimit] = {}
    for joint in root.findall("joint"):
        name, limit = joint.get("name"), joint.find("limit")
        if not name or limit is None or limit.get("lower") is None or limit.get("upper") is None:
            continue
        result[name] = JointLimit(name, float(limit.get("lower")), float(limit.get("upper")))
    return result


def compare_joint_limits(
    mujoco: Mapping[str, JointLimit],
    ros: Mapping[str, JointLimit],
    mapping: Mapping[str, str],
    *,
    atol: float = 1e-5,
) -> list[str]:
    errors: list[str] = []
    for mj_name, ros_name in mapping.items():
        if mj_name not in mujoco:
            errors.append(f"missing MuJoCo joint: {mj_name}")
            continue
        if ros_name not in ros:
            errors.append(f"missing ROS joint: {ros_name}")
            continue
        a, b = mujoco[mj_name], ros[ros_name]
        if not math.isclose(a.lower, b.lower, abs_tol=atol) or not math.isclose(a.upper, b.upper, abs_tol=atol):
            errors.append(f"limit mismatch {mj_name} <-> {ros_name}: {a} vs {b}")
    return errors


def quaternion_angle_error(q1: Sequence[float], q2: Sequence[float]) -> float:
    a = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    b = np.asarray(q2, dtype=float) / np.linalg.norm(q2)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return 2.0 * math.acos(dot)


def compare_tcp_poses(
    mujoco_position: Sequence[float],
    mujoco_quaternion_xyzw: Sequence[float],
    ros_position: Sequence[float],
    ros_quaternion_xyzw: Sequence[float],
    *,
    position_atol_m: float = 1e-4,
    orientation_atol_rad: float = 1e-3,
) -> list[str]:
    errors: list[str] = []
    position_error = float(np.linalg.norm(np.asarray(mujoco_position) - np.asarray(ros_position)))
    orientation_error = quaternion_angle_error(mujoco_quaternion_xyzw, ros_quaternion_xyzw)
    if position_error > position_atol_m:
        errors.append(f"TCP position error {position_error:.6g} m")
    if orientation_error > orientation_atol_rad:
        errors.append(f"TCP orientation error {orientation_error:.6g} rad")
    return errors


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    x, y, z = vector
    c, s, one_minus_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def urdf_fk_pose(
    path: Path,
    joint_positions: Mapping[str, float],
    *,
    tip_link: str = "fr3_hand_tcp",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a dependency-free URDF FK pose for the requested tip link.

    This intentionally covers the standard fixed/revolute/prismatic joints
    emitted by the pinned ``franka_description`` xacro.  It keeps the FK
    check runnable on a host that has MuJoCo but not a full ROS KDL stack.
    """
    root = ET.parse(path).getroot()
    joints = list(root.findall("joint"))
    links = {link.get("name") for link in root.findall("link") if link.get("name")}
    children = {joint.find("child").get("link") for joint in joints if joint.find("child") is not None}
    roots = links - children
    if len(roots) != 1:
        raise ValueError(f"URDF must have one root link, got {sorted(roots)}")
    poses: dict[str, np.ndarray] = {next(iter(roots)): np.eye(4)}
    pending = joints[:]
    while pending:
        progressed = False
        remaining = []
        for joint in pending:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in poses:
                remaining.append(joint)
                continue
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                if origin.get("xyz"):
                    transform[:3, 3] = np.asarray(origin.get("xyz").split(), dtype=float)
                if origin.get("rpy"):
                    transform[:3, :3] = _rpy_matrix(origin.get("rpy").split())
            joint_type = joint.get("type")
            if joint_type in {"revolute", "continuous"}:
                axis = joint.find("axis")
                if axis is None:
                    raise ValueError(f"missing axis for URDF joint {joint.get('name')}")
                transform[:3, :3] = transform[:3, :3] @ _axis_rotation(
                    axis.get("xyz", "0 0 1").split(), joint_positions.get(joint.get("name"), 0.0)
                )
            elif joint_type == "prismatic":
                axis = joint.find("axis")
                if axis is None:
                    raise ValueError(f"missing axis for URDF joint {joint.get('name')}")
                transform[:3, 3] += transform[:3, :3] @ (
                    np.asarray(axis.get("xyz", "0 0 1").split(), dtype=float)
                    * joint_positions.get(joint.get("name"), 0.0)
                )
            poses[child] = poses[parent] @ transform
            progressed = True
        if not progressed:
            unresolved = [joint.get("name") for joint in remaining]
            raise ValueError(f"URDF contains an unresolved joint chain: {unresolved}")
        pending = remaining
    if tip_link not in poses:
        raise ValueError(f"URDF is missing tip link {tip_link!r}")
    transform = poses[tip_link]
    # Convert the rotation matrix to the xyzw convention used by Robot API.
    rotation = transform[:3, :3]
    quaternion_wxyz = np.empty(4, dtype=float)
    try:
        import mujoco
    except ImportError:
        # A small local matrix-to-quaternion conversion keeps URDF FK useful
        # without making MuJoCo a hard dependency of the API package.
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = 2.0 * math.sqrt(trace + 1.0)
            quaternion_wxyz[:] = [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        else:
            diagonal = np.diag(rotation)
            index = int(np.argmax(diagonal))
            if index == 0:
                scale = 2.0 * math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-15))
                quaternion_wxyz[:] = [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            elif index == 1:
                scale = 2.0 * math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-15))
                quaternion_wxyz[:] = [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            else:
                scale = 2.0 * math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-15))
                quaternion_wxyz[:] = [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
    else:
        mujoco.mju_mat2Quat(quaternion_wxyz, rotation.reshape(-1))
    return transform[:3, 3].copy(), np.array(
        [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]],
        dtype=float,
    )


def mujoco_tcp_pose(
    path: Path,
    joint_positions: Mapping[str, float],
    *,
    site_name: str = "fr3_hand_tcp",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a generated MJCF TCP site at the supplied named joint values."""
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - runtime-only check
        raise ImportError("MuJoCo is required for the MJCF FK check") from exc
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for name, value in joint_positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo is missing joint {name!r}")
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo is missing TCP site {site_name!r}")
    quaternion_wxyz = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion_wxyz, data.site_xmat[site_id].reshape(-1))
    return data.site_xpos[site_id].copy(), np.array(
        [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]],
        dtype=float,
    )


def validate_fk_tcp_poses(mjcf: Path, urdf: Path) -> list[str]:
    """Compare MuJoCo and URDF TCP FK at zero and the pinned home pose."""
    home = {
        "fr3_joint1": 0.0,
        "fr3_joint2": 0.0,
        "fr3_joint3": 0.0,
        "fr3_joint4": -1.57079,
        "fr3_joint5": 0.0,
        "fr3_joint6": 1.57079,
        "fr3_joint7": -0.7853,
        "fr3_finger_joint1": 0.04,
        "fr3_finger_joint2": 0.04,
    }
    checks = [("zero", {name: 0.0 for name in home}), ("home", home)]
    errors: list[str] = []
    for label, positions in checks:
        mujoco_pose = mujoco_tcp_pose(mjcf, positions)
        urdf_pose = urdf_fk_pose(urdf, positions)
        errors.extend(f"{label}: {error}" for error in compare_tcp_poses(*mujoco_pose, *urdf_pose))
    return errors


def validate_gripper_width(
    finger_limits: Iterable[JointLimit],
    width_from_joints,
    *,
    atol_m: float = 1e-6,
) -> list[str]:
    limits = list(finger_limits)
    if len(limits) != 2:
        return ["expected exactly two continuous finger joints"]
    expected_min = limits[0].lower + limits[1].lower
    expected_max = limits[0].upper + limits[1].upper
    errors = []
    if not math.isclose(width_from_joints([limits[0].lower, limits[1].lower]), expected_min, abs_tol=atol_m):
        errors.append("gripper minimum width mapping mismatch")
    if not math.isclose(width_from_joints([limits[0].upper, limits[1].upper]), expected_max, abs_tol=atol_m):
        errors.append("gripper maximum width mapping mismatch")
    return errors
