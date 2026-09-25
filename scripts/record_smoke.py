#!/usr/bin/env python3
"""Record two short simulated episodes, then read frames and decode both videos.

This is pipeline validation, not task demonstrations or a learned policy.
No dataset is uploaded. Existing output directories are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for package in ("fr3_robot_api", "fr3_control", "fr3_sim", "fr3_recorder"):
    sys.path.insert(0, str(ROOT / "src" / package))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    if args.episodes < 1 or args.frames < 2:
        parser.error("need at least one episode and two frames")
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from fr3_robot_api.gripper import GripperLimits, GripperMapper
    from fr3_robot_api.types import Action
    from fr3_robot_api.timing import physics_tick_for_frame
    from fr3_sim import MujocoRobot
    from fr3_recorder.recorder import SynchronizedRecorder
    from fr3_recorder.writer import LeRobotDatasetWriter
    from verify_runtime import verify

    verify(require_lerobot=True)
    output = args.output or ROOT / "datasets" / datetime.now(timezone.utc).strftime("smoke-%Y%m%d-%H%M%S-%f")
    if output.exists():
        parser.error(f"output already exists: {output}")
    repo_id = "local/fr3_pipeline_smoke"
    mapper = GripperMapper(GripperLimits(0, .04, 0, .04))
    robot = MujocoRobot(ROOT / "src/fr3_description/models/scene.xml", mapper, interpolation="hold")
    writer = None
    try:
        writer = LeRobotDatasetWriter(repo_id, output, task="pipeline smoke: scripted joint motion")
        recorder = SynchronizedRecorder(writer, gripper_mapper=mapper)
        for episode in range(args.episodes):
            robot.reset()
            recorder.reset()
            home = robot.data.qpos[:7].copy()
            initial = home.copy()
            for frame in range(args.frames):
                observation = robot.get_observation()
                target = home.copy()
                target[0] += .10 * np.sin(2 * np.pi * frame / args.frames + episode * .3)
                action = robot.send_action(Action(observation.timestamp_ns, target, gripper_width_m=.04))
                if not recorder.record_frame(observation.external_rgb, [observation.wrist_rgb], [observation], action):
                    raise RuntimeError("unexpected frame rejection in synchronized simulation")
                current_tick = physics_tick_for_frame(frame)
                next_tick = physics_tick_for_frame(frame + 1)
                for _ in range(next_tick - current_tick):
                    robot.step()
                if not np.isfinite(robot.data.qpos).all():
                    raise RuntimeError("non-finite robot state")
            if np.allclose(robot.data.qpos[:7], initial):
                raise RuntimeError("robot did not move")
            writer.save_episode()
            print(f"Recorded episode {episode + 1}/{args.episodes}", flush=True)
        writer.finalize()
        writer = None
    finally:
        robot.close()
        if writer is not None:
            writer.finalize()

    # PyAV is an explicit decoding choice so this check does not depend on
    # the host's torch/torchcodec/FFmpeg ABI combination.
    dataset = LeRobotDataset(repo_id, root=output, video_backend="pyav")
    if len(dataset) != args.episodes * args.frames:
        raise RuntimeError("wrong number of frames after readback")
    for episode in range(args.episodes):
        previous_ns = -1
        for frame in range(args.frames):
            item = dataset[episode * args.frames + frame]
            timestamp_ns = int(item["timestamp_ns"].item())
            if timestamp_ns <= previous_ns:
                raise RuntimeError("source timestamps are not increasing within an episode")
            previous_ns = timestamp_ns
            if timestamp_ns != physics_tick_for_frame(frame) * 1_000_000:
                raise RuntimeError("source timeline differs from the 30 Hz scheduler")
            if int(item["action_timestamp_ns"].item()) != timestamp_ns:
                raise RuntimeError("action/observation alignment lost")
            for camera in ("external", "wrist"):
                image = item[f"observation.images.{camera}"]
                if tuple(image.shape) != (3, 480, 640) or not image.isfinite().all():
                    raise RuntimeError("invalid decoded image")
            if not item["observation.state"].isfinite().all():
                raise RuntimeError("invalid recorded state")
            if not np.allclose(item["action"][-2:].numpy(), [.04, .5]):
                raise RuntimeError("canonical gripper action changed in storage")
    report = {"episodes": args.episodes, "frames": len(dataset), "fps": 30,
              "source_physics_hz": 1000, "decoded_cameras": ["external", "wrist"],
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
              "status": "passed", "scope": "scripted pipeline validation; no task success claim"}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: recorded and read back {len(dataset)} frames: {output}")


if __name__ == "__main__":
    main()
