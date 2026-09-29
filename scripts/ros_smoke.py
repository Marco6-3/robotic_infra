#!/usr/bin/env python3
"""Validate an already launched FR3 ROS simulation on an isolated ROS_DOMAIN_ID."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from rosgraph_msgs.msg import Clock
from controller_manager_msgs.srv import ListControllers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch", action="store_true", help="launch and clean up a private headless simulation")
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output = args.output or root / "runs" / datetime.now(timezone.utc).strftime("ros-%Y%m%d-%H%M%S-%f") / "validation.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = None
    log = None
    if args.launch:
        os.environ.setdefault("ROS_DOMAIN_ID", "173")
        os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        log = args.output.with_suffix(".launch.log").open("w")
        process = subprocess.Popen(["ros2", "launch", "fr3_description", "sim.launch.py", "headless:=true"],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rclpy.init()
    node = rclpy.create_node("fr3_runtime_validation")
    states = []
    stamps = {"external": [], "wrist": []}
    shapes = {}
    images = {}
    clock = []
    subscriptions = []

    def image(name, msg):
        stamps[name].append(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)
        shapes[name] = [msg.height, msg.width, msg.encoding, len(msg.data)]
        images[name] = np.frombuffer(msg.data, dtype=np.uint8).copy()

    for name in stamps:
        subscriptions.append(node.create_subscription(
            Image, f"/fr3/cameras/{name}/color/image_raw",
            lambda msg, n=name: image(n, msg), qos_profile_sensor_data))
    subscriptions.append(node.create_subscription(JointState, "/joint_states", states.append, qos_profile_sensor_data))
    subscriptions.append(node.create_subscription(Clock, "/clock", lambda msg: clock.append(msg.clock.sec + msg.clock.nanosec / 1e9), qos_profile_sensor_data))
    publisher = node.create_publisher(Float64MultiArray, "/fr3/policy_action", 10)
    client = node.create_client(ListControllers, "/controller_manager/list_controllers")
    start = time.monotonic()
    future = None
    baseline = None
    target = None
    last_publish = 0
    sent = 0
    try:
        while time.monotonic() - start < args.timeout:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"ROS launch exited early: {process.returncode}")
            rclpy.spin_once(node, timeout_sec=.005)
            if future is None and client.service_is_ready():
                future = client.call_async(ListControllers.Request())
            active = set()
            if future is not None and future.done():
                active = {c.name for c in future.result().controller if c.state == "active"}
                if len(active) < 3:
                    future = None
            ready = {"joint_state_broadcaster", "arm_position_controller", "gripper_position_controller"} <= active
            if ready and states and baseline is None:
                by_name = dict(zip(states[-1].name, states[-1].position))
                baseline = np.array([by_name[f"fr3_joint{i}"] for i in range(1, 8)])
                target = baseline.copy()
                target[0] += .12
            if baseline is not None and clock and clock[-1] - last_publish >= 1 / 30:
                message = Float64MultiArray()
                message.data = [*target.tolist(), .04, .5]
                publisher.publish(message)
                sent += 1
                last_publish = clock[-1]
            if baseline is not None and sent >= 60 and all(len(s) >= 10 for s in stamps.values()):
                current = dict(zip(states[-1].name, states[-1].position))
                q = np.array([current[f"fr3_joint{i}"] for i in range(1, 8)])
                if abs(q[0] - target[0]) < .04:
                    break
        else:
            raise RuntimeError(f"ROS smoke timed out: states={len(states)}, images={dict((k, len(v)) for k,v in stamps.items())}, actions={sent}")
        if not np.isfinite(q).all() or abs(q[0] - baseline[0]) < .05:
            raise RuntimeError("robot did not respond to policy actions")
        for name, shape in shapes.items():
            if shape != [480, 640, "rgb8", 480 * 640 * 3]:
                raise RuntimeError(f"invalid {name} image: {shape}")
            if float(images[name].std()) < 1:
                raise RuntimeError(f"empty or constant {name} image")
            np.save(args.output.parent / f"{name}.npy", images[name].reshape(480, 640, 3))
        report = {
            "status": "passed", "active_controllers": sorted(active),
            "policy_actions_sent": sent, "joint_state_messages": len(states),
            "joint1_displacement_rad": float(q[0] - baseline[0]),
            "joint1_target_error_rad": float(abs(q[0] - target[0])),
            "camera_shapes": shapes, "camera_frames": {k: len(v) for k, v in stamps.items()},
            "camera_pixel_std": {k: float(v.std()) for k, v in images.items()},
            "camera_sim_fps": {k: (len(v) - 1) / (v[-1] - v[0]) for k, v in stamps.items()},
            "sim_time_s": clock[-1], "wall_time_s": time.monotonic() - start,
            "scope": "ROS transport, camera and scripted command validation; no hard real-time guarantee",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
            log.close()


if __name__ == "__main__":
    main()
