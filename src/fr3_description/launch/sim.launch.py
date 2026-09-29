"""ROS 2 Jazzy FR3 MuJoCo bringup using the pinned control plugin."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mujoco_model",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fr3_description"), "models", "scene.xml"]
                ),
            ),
            DeclareLaunchArgument("target_interpolation", default_value="linear"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("start_policy_bridge", default_value="true"),
            LogInfo(msg=["Target interpolation: ", LaunchConfiguration("target_interpolation")]),
            Node(
                package="fr3_control",
                executable="fr3_policy_bridge",
                condition=IfCondition(LaunchConfiguration("start_policy_bridge")),
                parameters=[
                    {"use_sim_time": True, "interpolation": LaunchConfiguration("target_interpolation")},
                    ParameterFile(PathJoinSubstitution([
                        FindPackageShare("fr3_description"), "models", "control_limits.yaml"
                    ])),
                ],
                output="screen",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[
                    {"use_sim_time": True},
                    {
                        "robot_description": ParameterValue(Command(
                            [
                                FindExecutable(name="xacro"),
                                " ",
                                PathJoinSubstitution(
                                    [FindPackageShare("fr3_description"), "urdf", "fr3_mujoco.urdf.xacro"]
                                ),
                                " mujoco_model:=",
                                LaunchConfiguration("mujoco_model"),
                                " headless:=", LaunchConfiguration("headless"),
                            ]
                        ), value_type=str),
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="mujoco_ros2_control",
                executable="ros2_control_node",
                parameters=[
                    {"use_sim_time": True},
                    {
                        "robot_description": ParameterValue(Command(
                            [
                                FindExecutable(name="xacro"),
                                " ",
                                PathJoinSubstitution(
                                    [FindPackageShare("fr3_description"), "urdf", "fr3_mujoco.urdf.xacro"]
                                ),
                                " mujoco_model:=",
                                LaunchConfiguration("mujoco_model"),
                                " headless:=", LaunchConfiguration("headless"),
                            ]
                        ), value_type=str)
                    },
                    ParameterFile(
                        PathJoinSubstitution([FindPackageShare("fr3_description"), "config", "controllers.yaml"])
                    ),
                    ParameterFile(
                        PathJoinSubstitution([FindPackageShare("fr3_description"), "config", "mujoco_plugins.yaml"])
                    ),
                ],
                output="both",
                remappings=[("~/robot_description", "/robot_description")],
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["arm_position_controller", "--controller-manager", "/controller_manager"],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["gripper_position_controller", "--controller-manager", "/controller_manager"],
                output="screen",
            ),
        ]
    )
