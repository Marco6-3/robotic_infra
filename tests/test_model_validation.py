from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

from fr3_robot_api.model_validation import (
    JointLimit,
    compare_joint_limits,
    compare_tcp_poses,
    parse_mjcf_joint_limits,
    parse_urdf_joint_limits,
    validate_gripper_width,
)


def test_joint_limit_mapping_and_gripper_geometry():
    mj = {
        "fr3_joint1": JointLimit("fr3_joint1", -2.0, 2.0),
        "fr3_finger_joint1": JointLimit("fr3_finger_joint1", 0.0, 0.04),
        "fr3_finger_joint2": JointLimit("fr3_finger_joint2", 0.0, 0.04),
    }
    ros = {
        "fr3_joint1": JointLimit("fr3_joint1", -2.0, 2.0),
        "fr3_finger_joint1": JointLimit("fr3_finger_joint1", 0.0, 0.04),
        "fr3_finger_joint2": JointLimit("fr3_finger_joint2", 0.0, 0.04),
    }
    assert compare_joint_limits(mj, ros, {"fr3_joint1": "fr3_joint1"}) == []
    assert validate_gripper_width(
        [mj["fr3_finger_joint1"], mj["fr3_finger_joint2"]], sum
    ) == []


def test_tcp_pose_validation_accepts_quaternion_sign_equivalence():
    assert compare_tcp_poses(
        [0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.1], [0.0, 0.0, 0.0, -1.0],
    ) == []


def test_xml_joint_limit_parsers_cover_mujoco_and_ros(tmp_path):
    mjcf = tmp_path / "model.xml"
    mjcf.write_text(
        '<mujoco><worldbody><body><joint name="fr3_joint1" range="-2 2"/>'
        '<joint name="fr3_finger_joint1" range="0 0.04"/>'
        '</body></worldbody></mujoco>'
    )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="fr3"><joint name="fr3_joint1" type="revolute">'
        '<limit lower="-2" upper="2"/></joint>'
        '<joint name="fr3_finger_joint1" type="prismatic">'
        '<limit lower="0" upper="0.04"/></joint></robot>'
    )
    assert parse_mjcf_joint_limits(mjcf)["fr3_joint1"].upper == 2.0
    assert parse_urdf_joint_limits(urdf)["fr3_finger_joint1"].upper == 0.04


@pytest.mark.integration
def test_generated_mjcf_contract_has_one_tendon_gripper_and_two_rgb_cameras():
    model = Path(__file__).parents[1] / "src/fr3_description/models/fr3_hand.xml"
    root = ET.parse(model).getroot()
    assert root.find("option").get("timestep") == "0.001"
    actuators = {a.get("name"): a for a in root.findall(".//actuator/*")}
    assert "fr3_finger_joint1" in actuators
    assert "fr3_finger_joint2" not in actuators
    assert actuators["fr3_finger_joint1"].get("tendon") == "fr3_split"
    cameras = {c.get("name"): c for c in root.findall(".//camera")}
    assert set(cameras) == {"wrist"}
    scene = ET.parse(model.parent / "scene.xml").getroot()
    cameras.update({c.get("name"): c for c in scene.findall(".//camera")})
    assert set(cameras) == {"external", "wrist"}
    assert all(c.get("resolution") == "640 480" for c in cameras.values())
    key = root.find(".//keyframe/key")
    assert len(key.get("qpos").split()) == 9
    assert len(key.get("ctrl").split()) == 8
