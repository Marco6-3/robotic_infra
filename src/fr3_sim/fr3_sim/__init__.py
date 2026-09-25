"""Optional direct MuJoCo validation adapter.

The production ROS path uses ``mujoco_ros2_control``. This adapter is useful
for smoke-testing the generated model and the framework-neutral Robot contract
on hosts where ROS is not yet sourced.
"""

from .mujoco_robot import MujocoUnavailable, MujocoRobot

__all__ = ["MujocoRobot", "MujocoUnavailable"]
