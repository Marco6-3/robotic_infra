from dataclasses import dataclass

import numpy as np

from fr3_recorder.recorder import SynchronizedRecorder
from fr3_robot_api.types import Action, CameraFrame, Observation, Pose
from fr3_recorder.synchronizer import nearest_by_timestamp


@dataclass
class S:
    timestamp_ns: int


def test_nearest_sample_and_tolerance():
    samples = [S(0), S(1_000_000), S(2_000_000)]
    assert nearest_by_timestamp(samples, 1_600_000).timestamp_ns == 2_000_000
    assert nearest_by_timestamp(samples, 10_000_000, max_delta_ns=100) is None


class FakeWriter:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)


def test_recorder_uses_external_camera_timeline_and_nearest_state():
    writer = FakeWriter()
    recorder = SynchronizedRecorder(writer)
    external = CameraFrame(1_000_000_000, np.zeros((480, 640, 3), dtype=np.uint8), "external")
    wrist = [CameraFrame(1_000_100_000, np.ones((480, 640, 3), dtype=np.uint8), "wrist")]
    state = Observation(
        timestamp_ns=999_900_000,
        state_timestamp_ns=999_900_000,
        joint_position_rad=np.zeros(7),
        joint_velocity_rad_s=np.zeros(7),
        ee_pose=Pose(np.zeros(3), [0, 0, 0, 1]),
        gripper_width_m=0.04,
        gripper_width_normalized=0.5,
        external_rgb=external,
        wrist_rgb=wrist[0],
    )
    action = Action(1_000_000_000, np.zeros(7), 0.04, 0.5)
    assert recorder.record_frame(external, wrist, [state], action)
    assert writer.frames[0]["timestamp_ns"].item() == 1_000_000_000
    assert writer.frames[0]["wrist_timestamp_ns"].item() == 1_000_100_000
    assert writer.frames[0]["state_timestamp_ns"].item() == 999_900_000
