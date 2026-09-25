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
