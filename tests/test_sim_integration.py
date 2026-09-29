from pathlib import Path

import numpy as np
import pytest

from fr3_robot_api.gripper import GripperLimits, GripperMapper
from fr3_robot_api.types import Action
from fr3_sim import MujocoRobot


@pytest.mark.integration
def test_simulator_moves_resets_and_rejects_future_targets():
    scene = Path(__file__).parents[1] / "src/fr3_description/models/scene.xml"
    robot = MujocoRobot(scene, GripperMapper(GripperLimits(0, .04, 0, .04)))
    try:
        initial = robot.data.qpos.copy()
        assert initial[3] < 0  # actual home keyframe, not an invalid zero pose
        with pytest.raises(ValueError):
            robot.send_action(Action(1, initial[:7], .04))
        target = initial[:7].copy()
        target[0] += .1
        robot.send_action(Action(0, target, .04))
        for _ in range(250):
            robot.step()
        assert robot.data.time == pytest.approx(.25)
        assert robot.data.qpos[0] > initial[0] + .01
        assert np.isfinite(robot.data.qpos).all()
        robot.reset()
        assert robot.data.time == 0
        np.testing.assert_allclose(robot.data.qpos, initial)
        with pytest.raises(RuntimeError):
            robot.step()
    finally:
        robot.close()


@pytest.mark.integration
def test_wrist_camera_includes_tcp_and_sees_past_hand():
    """A correctly sized camera frame can still be blocked by the palm."""
    import mujoco

    scene = Path(__file__).parents[1] / "src/fr3_description/models/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fr3_hand_tcp")
    rotation = data.cam_xmat[camera].reshape(3, 3)
    relative = rotation.T @ (data.site_xpos[tcp] - data.cam_xpos[camera])
    assert relative[2] < 0
    half_height = -relative[2] * np.tan(np.deg2rad(model.cam_fovy[camera] / 2))
    assert abs(relative[1]) < half_height
    assert abs(relative[0]) < half_height * 640 / 480
    # The center ray must reach the workspace rather than hit the camera's own
    # hand housing a few centimetres away. This does not require a GL context.
    geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, data.cam_xpos[camera], -rotation[:, 2],
                             None, 1, -1, geom)
    assert distance > .15


@pytest.mark.integration
def test_scene_has_reachable_collision_workbench():
    import mujoco

    scene = Path(__file__).parents[1] / "src/fr3_description/models/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "workbench_top")
    workspace = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "task_workspace_center"
    )
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fr3_hand_tcp")
    assert top >= 0 and workspace >= 0
    assert model.geom_contype[top] != 0
    assert data.site_xpos[workspace][2] > data.geom_xpos[top][2]
    # The home TCP should begin above and close enough to the task surface for
    # a later contact task; this is a geometry gate, not a reachability proof.
    distance = np.linalg.norm(data.site_xpos[tcp] - data.site_xpos[workspace])
    assert distance < .20


@pytest.mark.integration
def test_fast_tactile_observation_does_not_require_camera_rendering():
    scene = Path(__file__).parents[1] / "src/fr3_description/models/scene.xml"
    robot = MujocoRobot(scene, GripperMapper(GripperLimits(0, .04, 0, .04)))
    try:
        tactile = robot.get_tactile_observation()
        assert tactile.channel_names == ("left_normal_force_n", "right_normal_force_n")
        assert tactile.metadata["privileged"] is True
        assert tactile.metadata["proxy"] is True
        assert tactile.source_name == "mujoco_fingertip_contact_force_proxy"
        assert tactile.timestamp_ns == 0
        assert np.all(tactile.values >= 0)
        assert robot._renderer is None
    finally:
        robot.close()
