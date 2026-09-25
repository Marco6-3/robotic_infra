"""LeRobot feature schema used by the v1 recorder."""

from __future__ import annotations

from typing import Any, Dict


def build_lerobot_features() -> Dict[str, Dict[str, Any]]:
    """Return a stable feature declaration for ``LeRobotDataset.create``.

    ``dtype=video`` asks LeRobot to use its normal encoded-video pipeline. The
    schema intentionally includes source timestamps as numeric features so
    future higher-rate modalities can be added without changing the frame
    identity.
    """

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (7 + 7 + 7 + 2,),
            "names": [
                *[f"joint_position_rad_{i}" for i in range(1, 8)],
                *[f"joint_velocity_rad_s_{i}" for i in range(1, 8)],
                "ee_position_x_m",
                "ee_position_y_m",
                "ee_position_z_m",
                "ee_quaternion_x",
                "ee_quaternion_y",
                "ee_quaternion_z",
                "ee_quaternion_w",
                "gripper_width_m",
                "gripper_width_normalized",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7 + 1 + 1,),
            "names": [
                *[f"arm_joint_position_target_rad_{i}" for i in range(1, 8)],
                "gripper_width_m",
                "gripper_width_normalized",
            ],
        },
        "observation.images.external": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "timestamp_ns": {"dtype": "int64", "shape": (1,), "names": ["timestamp_ns"]},
        "action_timestamp_ns": {"dtype": "int64", "shape": (1,), "names": ["action_timestamp_ns"]},
        "decision_timestamp_ns": {"dtype": "int64", "shape": (1,), "names": ["decision_timestamp_ns"]},
        "state_timestamp_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["state_timestamp_ns"],
        },
        "external_timestamp_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["external_timestamp_ns"],
        },
        "wrist_timestamp_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["wrist_timestamp_ns"],
        },
    }
