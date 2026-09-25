"""Regressions for failures at the control/recording boundaries."""
import importlib
import sys
from types import ModuleType, SimpleNamespace, MethodType

import numpy as np
import pytest

from fr3_control.bridge import PolicyTargetBridge
from fr3_robot_api.gripper import GripperLimits, GripperMapper
from fr3_robot_api.types import Action, CameraFrame, Observation, Pose
from fr3_robot_api.validation import canonical_action
from fr3_robot_api.timing import physics_tick_for_frame
from fr3_recorder.recorder import SynchronizedRecorder, RecorderConfig


@pytest.fixture
def mapper():
    return GripperMapper(GripperLimits(0, .04, 0, .04))


@pytest.mark.parametrize("bad", [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_commands_rejected(bad):
    with pytest.raises(ValueError):
        Action(0, [bad] * 7, gripper_width_m=.04)
    with pytest.raises(ValueError):
        Action(0, np.zeros(7), gripper_width_m=bad)


def test_canonical_width_and_rejection_are_atomic(mapper):
    bridge = PolicyTargetBridge(mapper, interpolation="hold")
    accepted = bridge.submit(Action(0, np.zeros(7), gripper_width_normalized=.5))
    assert accepted.gripper_width_m == .04
    for bad in [Action(1, np.ones(7), .10, .1), Action(1, np.ones(7), .04, .1)]:
        with pytest.raises(ValueError):
            bridge.submit(bad)
        np.testing.assert_array_equal(bridge.step(1).arm_joint_position_target_rad, np.zeros(7))
    assert bridge.step(1).gripper_actuated_joint_position_target_m == .02


def test_joint_bounds_and_duplicate_time_are_rejected(mapper):
    bridge = PolicyTargetBridge(mapper, arm_limits=[[-1, 1]] * 7)
    bridge.submit(Action(10, np.zeros(7), .04))
    with pytest.raises(ValueError):
        bridge.submit(Action(20, np.full(7, 2.), .04))
    with pytest.raises(ValueError):
        bridge.submit(Action(10, np.zeros(7), .04))
    with pytest.raises(ValueError):
        bridge.step(9)
    bridge.reset()
    assert not bridge.has_target
    bridge.submit(Action(0, np.zeros(7), .04))
    bridge.step(0)


def test_ros_callbacks_publish_first_command_and_reset_on_clock_jump(monkeypatch, mapper):
    # Execute the actual ROS callbacks with a fake transport/clock; no ROS install
    # is needed, and these stubs are removed after this test.
    rclpy = ModuleType("rclpy")
    node = ModuleType("rclpy.node")
    node.Node = object
    std_msgs = ModuleType("std_msgs")
    msg = ModuleType("std_msgs.msg")
    msg.Float64MultiArray = object
    for name, module in {"rclpy": rclpy, "rclpy.node": node,
                         "std_msgs": std_msgs, "std_msgs.msg": msg}.items():
        monkeypatch.setitem(sys.modules, name, module)
    name = "fr3_control.ros_policy_bridge"
    sys.modules.pop(name, None)
    try:
        cls = importlib.import_module(name).PolicyBridgeNode
        sent, warnings = [], []
        clock = SimpleNamespace(nanoseconds=100)
        fake = SimpleNamespace(
            _bridge=PolicyTargetBridge(mapper, publish=sent.append),
            _last_clock_ns=None,
            get_clock=lambda: SimpleNamespace(now=lambda: clock),
            get_logger=lambda: SimpleNamespace(warning=warnings.append, error=warnings.append),
        )
        fake._clock_tick = MethodType(cls._clock_tick, fake)
        cls._on_low_level_tick(fake)
        assert sent == []
        cls._on_policy_action(fake, SimpleNamespace(data=[0.] * 7 + [.04, .5]))
        cls._on_low_level_tick(fake)
        assert len(sent) == 1
        clock.nanoseconds = 200
        cls._on_policy_action(fake, SimpleNamespace(data=[float('nan')] * 7 + [.04, .5]))
        assert warnings
        clock.nanoseconds = 0
        cls._on_low_level_tick(fake)
        assert len(sent) == 1 and not fake._bridge.has_target
        cls._on_policy_action(fake, SimpleNamespace(data=[0.] * 7 + [.04, .5]))
        cls._on_low_level_tick(fake)
        assert len(sent) == 2
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def recording():
    frames = []
    rec = SynchronizedRecorder(SimpleNamespace(add_frame=frames.append))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    t = 1_000_000_000
    external = CameraFrame(t, image, "external")
    wrist = CameraFrame(t, image, "wrist")
    state = Observation(t, np.zeros(7), np.zeros(7), Pose(np.zeros(3), [0, 0, 0, 1]),
                        .04, .5, external, wrist)
    return rec, frames, external, wrist, state


@pytest.mark.parametrize("offset", [-100_000_000, 1])
def test_causal_recorder_rejects_stale_or_future_samples(recording, offset):
    rec, _, external, wrist, state = recording
    bad_wrist = CameraFrame(wrist.timestamp_ns + offset, wrist.rgb, "wrist")
    assert rec.make_frame(external, [bad_wrist], [state], Action(external.timestamp_ns, np.zeros(7), .04)) is None


@pytest.mark.parametrize("offset", [-100_000_000, 1])
def test_recorder_rejects_misaligned_actions(recording, offset):
    rec, _, external, wrist, state = recording
    assert rec.make_frame(external, [wrist], [state], Action(external.timestamp_ns + offset, np.zeros(7), .04)) is None


def test_recorded_action_is_canonical_and_keeps_decision_time(recording):
    rec, frames, external, wrist, state = recording
    decision = external.timestamp_ns + 1_000_000
    action = Action(decision, np.zeros(7), gripper_width_normalized=.5)
    assert rec.record_frame(external, [wrist], [state], action, decision_timestamp_ns=decision)
    np.testing.assert_allclose(frames[0]["action"][-2:], [.04, .5])
    assert frames[0]["action_timestamp_ns"].item() == decision
    assert frames[0]["decision_timestamp_ns"].item() == decision
    with pytest.raises(ValueError):
        rec.record_frame(external, [wrist], [state], action, decision_timestamp_ns=decision)
    rec.reset()
    assert rec.record_frame(external, [wrist], [state], action, decision_timestamp_ns=decision)


def test_explicit_offline_nearest_mode(recording):
    _, frames, external, wrist, state = recording
    rec = SynchronizedRecorder(SimpleNamespace(add_frame=frames.append), RecorderConfig(causal=False))
    future = CameraFrame(wrist.timestamp_ns + 1, wrist.rgb, "wrist")
    assert rec.record_frame(external, [future], [state], Action(external.timestamp_ns, np.zeros(7), .04))


def test_policy_clock_has_no_cumulative_33_tick_drift():
    ticks = [physics_tick_for_frame(i) for i in range(301)]
    assert ticks[-1] == 10_000
    assert set(np.diff(ticks)) == {33, 34}
    assert sum(t < 1000 for t in ticks) == 30
