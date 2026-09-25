#!/usr/bin/env python3
"""Run a short scripted FR3 control-loop smoke test without ROS.

This exercises the generated MJCF and the same 30 Hz action / 1 kHz stepping
semantics used by the ROS bridge. Camera rendering is intentionally not called
so the check also works on a headless machine; camera validation is performed
by the renderer-backed adapter when an OpenGL context is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/fr3_robot_api"))
sys.path.insert(0, str(ROOT / "src/fr3_control"))
sys.path.insert(0, str(ROOT / "src/fr3_sim"))

from fr3_robot_api.gripper import GripperLimits, GripperMapper  # noqa: E402
from fr3_robot_api.types import Action  # noqa: E402
from fr3_sim import MujocoRobot, MujocoUnavailable  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=ROOT / "src/fr3_description/models/scene.xml")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    mapper = GripperMapper(GripperLimits(0.0, 0.04, 0.0, 0.04))
    try:
        robot = MujocoRobot(args.scene, mapper, interpolation="hold")
    except MujocoUnavailable as exc:
        print(exc, file=sys.stderr)
        return 2
    home = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.7])
    ticks = int(round(args.seconds * 1000))
    for tick in range(ticks):
        if tick % 33 == 0:
            timestamp_ns = int(round(tick * 1_000_000))
            phase = 0.15 * np.sin(2.0 * np.pi * (tick / 1000.0))
            robot.send_action(Action(timestamp_ns, home + phase, gripper_width_m=0.04, gripper_width_normalized=0.5))
        robot.step()
    print(f"smoke test complete: sim_time={robot.data.time:.6f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
