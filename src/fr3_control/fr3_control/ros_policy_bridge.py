"""ROS 2 adapter around :class:`PolicyTargetBridge`.

The v1 validation command uses ``Float64MultiArray`` to keep the scaffold free
from a custom message package. The array layout is
``[q1..q7, gripper_width_m, gripper_width_normalized]``. A future policy node
can replace this transport without changing the Robot API or recorder.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from fr3_robot_api.gripper import GripperLimits, GripperMapper
from fr3_robot_api.types import Action

from .bridge import PolicyTargetBridge


class PolicyBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("fr3_policy_bridge")
        self.declare_parameter("interpolation", "linear")
        self.declare_parameter("policy_action_topic", "/fr3/policy_action")
        self.declare_parameter("arm_command_topic", "/arm_position_controller/commands")
        self.declare_parameter("gripper_command_topic", "/gripper_position_controller/commands")
        mapper = GripperMapper(GripperLimits(0.0, 0.04, 0.0, 0.04))
        self._arm_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("arm_command_topic").value, 10
        )
        self._gripper_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("gripper_command_topic").value, 10
        )
        self._bridge = PolicyTargetBridge(
            mapper,
            interpolation=str(self.get_parameter("interpolation").value),
            publish=self._publish_command,
        )
        self._subscription = self.create_subscription(
            Float64MultiArray,
            self.get_parameter("policy_action_topic").value,
            self._on_policy_action,
            10,
        )
        self._timer = self.create_timer(0.001, self._on_low_level_tick)

    def _on_policy_action(self, message: Float64MultiArray) -> None:
        if len(message.data) != 9:
            self.get_logger().error("policy action must contain 9 values: q7, width_m, width_normalized")
            return
        now = self.get_clock().now().nanoseconds
        self._bridge.submit(Action(now, message.data[:7], message.data[7], message.data[8]))

    def _on_low_level_tick(self) -> None:
        if self._bridge.last_command is None:
            return
        self._bridge.step(self.get_clock().now().nanoseconds)

    def _publish_command(self, command) -> None:
        arm = Float64MultiArray()
        arm.data = command.arm_joint_position_target_rad.tolist()
        self._arm_pub.publish(arm)
        gripper = Float64MultiArray()
        gripper.data = [command.gripper_actuated_joint_position_target_m]
        self._gripper_pub.publish(gripper)


def main() -> None:
    rclpy.init()
    node = PolicyBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
