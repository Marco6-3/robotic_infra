import numpy as np
import pytest

from fr3_control.bridge import PolicyTargetBridge
from fr3_recorder.schema import build_lerobot_features
from fr3_robot_api.camera import CameraConfig
from fr3_robot_api.gripper import GripperLimits, GripperMapper
from fr3_robot_api.tactile import ArrayVisionTactileSource, VisionTactileConfig
from fr3_robot_api.timing import TargetInterpolator
from fr3_robot_api.types import (
    Action,
    CameraFrame,
    Observation,
    Pose,
    TactileFrame,
    VisionTactileFrame,
)


def mapper():
    return GripperMapper(GripperLimits(0.0, 0.04, 0.0, 0.04))


def test_gripper_physical_and_normalized_round_trip():
    m = mapper()
    assert m.width_from_joints(np.array([0.02, 0.02])) == pytest.approx(0.04)
    assert m.normalized_from_width(0.04) == pytest.approx(0.5)
    np.testing.assert_allclose(m.joints_from_normalized(1.0), [0.04, 0.04])


def test_target_interpolator_handles_1000_to_30_hz_interval():
    i = TargetInterpolator(1, "linear")
    i.set_target(0, np.array([0.0]))
    i.set_target(33_333_333, np.array([1.0]))
    # Causal interpolation starts when the second policy target arrives and
    # reaches it one policy period later.
    assert i.evaluate(33_333_333)[0] == pytest.approx(0.0, abs=1e-6)
    assert i.evaluate(49_999_999)[0] == pytest.approx(0.5, abs=1e-6)
    assert i.evaluate(66_666_666)[0] == pytest.approx(1.0)


def test_bridge_maps_normalized_width_to_two_finger_targets():
    b = PolicyTargetBridge(mapper(), interpolation="hold")
    b.submit(Action(0, np.zeros(7), gripper_width_normalized=0.5))
    command = b.step(0)
    assert command.gripper_actuated_joint_position_target_m == pytest.approx(0.02)


def test_observation_and_action_shapes():
    frame = CameraFrame(10, np.zeros((480, 640, 3), dtype=np.uint8), "external")
    wrist = CameraFrame(10, np.zeros((480, 640, 3), dtype=np.uint8), "wrist")
    obs = Observation(
        timestamp_ns=10,
        joint_position_rad=np.zeros(7),
        joint_velocity_rad_s=np.zeros(7),
        ee_pose=Pose(np.zeros(3), [0, 0, 0, 1]),
        gripper_width_m=0.04,
        gripper_width_normalized=0.5,
        external_rgb=frame,
        wrist_rgb=wrist,
    )
    action = Action(10, np.zeros(7), gripper_width_m=0.04, gripper_width_normalized=0.5)
    assert obs.joint_position_rad.shape == (7,)
    assert action.arm_joint_position_target_rad.shape == (7,)


def test_tactile_frame_requires_finite_named_channels():
    frame = TactileFrame(10, [0.0, 1.5], ("left_n", "right_n"), "test")
    assert frame.values.shape == (2,)
    with pytest.raises(ValueError):
        TactileFrame(10, [0.0, np.nan], ("left_n", "right_n"), "test")
    with pytest.raises(ValueError):
        TactileFrame(10, [0.0], ("left_n", "right_n"), "test")


def test_common_vision_tactile_contract_is_dual_digit_compatible():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    config = VisionTactileConfig("left_digit")
    source = ArrayVisionTactileSource(config, lambda: (10, 2, image))
    frame = source.read()
    assert frame.sensor_name == "left_digit"
    assert frame.rgb.shape == (480, 640, 3)
    assert (config.width, config.height, config.fps) == (640, 480, 60)
    with pytest.raises(ValueError):
        VisionTactileFrame(10, np.zeros((10, 10)), "bad", 0)


def test_observation_carries_two_causal_vision_tactile_frames():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    camera = CameraFrame(20, image, "external")
    wrist = CameraFrame(20, image, "wrist")
    tactile_frames = (
        VisionTactileFrame(18, image, "left_digit", 1),
        VisionTactileFrame(19, image, "right_digit", 1),
    )
    observation = Observation(
        timestamp_ns=20,
        joint_position_rad=np.zeros(7),
        joint_velocity_rad_s=np.zeros(7),
        ee_pose=Pose(np.zeros(3), [0, 0, 0, 1]),
        gripper_width_m=0.04,
        gripper_width_normalized=0.5,
        external_rgb=camera,
        wrist_rgb=wrist,
        vision_tactile=tactile_frames,
    )
    assert tuple(frame.sensor_name for frame in observation.vision_tactile) == (
        "left_digit",
        "right_digit",
    )
    with pytest.raises(ValueError):
        Observation(
            timestamp_ns=20,
            joint_position_rad=np.zeros(7),
            joint_velocity_rad_s=np.zeros(7),
            ee_pose=Pose(np.zeros(3), [0, 0, 0, 1]),
            gripper_width_m=0.04,
            gripper_width_normalized=0.5,
            external_rgb=camera,
            wrist_rgb=wrist,
            vision_tactile=(VisionTactileFrame(21, image, "left_digit", 2),),
        )


def test_lerobot_video_features_are_640x480():
    features = build_lerobot_features()
    assert len(features["observation.state"]["names"]) == features["observation.state"]["shape"][0]
    assert len(features["action"]["names"]) == features["action"]["shape"][0]
    assert features["observation.images.external"]["dtype"] == "video"
    assert features["observation.images.external"]["shape"] == (480, 640, 3)
    assert features["observation.images.wrist"]["dtype"] == "video"


def test_v1_camera_contract_is_rgb_only_and_d435_sized():
    camera = CameraConfig("external")
    assert camera.model == "intel_realsense_d435_rgb"
    assert (camera.width, camera.height, camera.fps) == (640, 480, 30)
    assert camera.fov_y_deg == pytest.approx(42.5)
    with pytest.raises(ValueError):
        CameraConfig("depth", depth=True)
