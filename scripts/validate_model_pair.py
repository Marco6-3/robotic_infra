#!/usr/bin/env python3
"""Run description-level validation after generating and expanding models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/fr3_robot_api"))

from fr3_robot_api.model_validation import (  # noqa: E402
    compare_joint_limits,
    parse_mjcf_joint_limits,
    parse_urdf_joint_limits,
    validate_fk_tcp_poses,
    validate_gripper_width,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, default=Path("src/fr3_description/models/fr3_hand.xml"))
    parser.add_argument("--urdf", type=Path, required=True, help="expanded fr3 + franka_hand URDF")
    parser.add_argument(
        "--skip-fk",
        action="store_true",
        help="skip runtime MuJoCo-vs-URDF TCP FK checks (useful without MuJoCo installed)",
    )
    args = parser.parse_args()
    mj = parse_mjcf_joint_limits(args.mjcf)
    ros = parse_urdf_joint_limits(args.urdf)
    mapping = {f"fr3_joint{i}": f"fr3_joint{i}" for i in range(1, 8)}
    mapping.update({f"fr3_finger_joint{i}": f"fr3_finger_joint{i}" for i in (1, 2)})
    errors = compare_joint_limits(mj, ros, mapping)
    finger_limits = [mj[name] for name in ("fr3_finger_joint1", "fr3_finger_joint2") if name in mj]
    errors.extend(validate_gripper_width(finger_limits, lambda joints: sum(joints)))
    fk_skipped = args.skip_fk
    if not args.skip_fk:
        try:
            errors.extend(validate_fk_tcp_poses(args.mjcf, args.urdf))
        except ImportError:
            fk_skipped = True
    if errors:
        print("MODEL VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("MODEL VALIDATION PASSED: joint names/limits and gripper width")
    if fk_skipped:
        print("FK/TCP pose checks SKIPPED: install the pinned MuJoCo runtime to enable them")
    else:
        print("FK/TCP pose checks PASSED: zero and home TCP poses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
