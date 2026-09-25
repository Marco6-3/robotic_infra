#!/usr/bin/env python3
"""Publish a minimal 30 Hz keyboard policy action for ROS validation.

The transport intentionally matches ``fr3_control/ros_policy_bridge.py``:
``[q1..q7, gripper_width_m, gripper_width_normalized]`` on
``/fr3/policy_action``. Keys are incremental and safe for validating the
control chain, not a teleoperation implementation.

Controls: ``1..7`` select a joint, ``a/d`` move it, ``o/c`` open/close the
gripper, ``space`` returns to the pinned home target, and ``q`` exits.
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


HOME = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]


class KeyboardPolicy(Node):
    def __init__(self) -> None:
        super().__init__("fr3_keyboard_policy")
        self.publisher = self.create_publisher(Float64MultiArray, "/fr3/policy_action", 10)
        self.timer = self.create_timer(1.0 / 30.0, self.tick)
        self.target = list(HOME)
        self.selected_joint = 0
        self.width_m = 0.04
        self._stdin_fd = sys.stdin.fileno()
        self._old_terminal = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        self.get_logger().info("Keyboard FR3 policy: 1-7 select, a/d move, o/c gripper, space home, q quit")

    def tick(self) -> None:
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            if key == "q":
                rclpy.shutdown()
                return
            if key in "1234567":
                self.selected_joint = int(key) - 1
            elif key == "a":
                self.target[self.selected_joint] += 0.02
            elif key == "d":
                self.target[self.selected_joint] -= 0.02
            elif key == "o":
                self.width_m = min(0.08, self.width_m + 0.005)
            elif key == "c":
                self.width_m = max(0.0, self.width_m - 0.005)
            elif key == " ":
                self.target = list(HOME)
                self.width_m = 0.04
        message = Float64MultiArray()
        message.data = self.target + [self.width_m, self.width_m / 0.08]
        self.publisher.publish(message)

    def close_terminal(self) -> None:
        termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_terminal)


def main() -> None:
    rclpy.init()
    node = KeyboardPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_terminal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
