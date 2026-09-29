from pathlib import Path

import pytest

from experiments.iros2027.stage0 import (
    ExperimentPlan,
    latest_causal_age_ns,
    periodic_timestamps_ns,
)


ROOT = Path(__file__).parents[1]


def test_periodic_schedule_is_drift_bounded_at_30_hz():
    timestamps = periodic_timestamps_ns(30, 1.0)
    assert len(timestamps) == 30
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:])]
    assert set(intervals) == {33_333_333, 33_333_334}
    assert abs(timestamps[-1] - round(29 * 1_000_000_000 / 30)) <= 1


def test_latest_age_never_uses_a_future_sample():
    source = [0, 200_000_000, 400_000_000]
    assert latest_causal_age_ns(350_000_000, source) == 150_000_000
    with pytest.raises(ValueError):
        latest_causal_age_ns(-1, source)


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_name",
    ["i001_temporal_context.yaml", "i002_async_communication.yaml"],
)
def test_stage0_configs_validate_against_generated_scene(config_name):
    plan = ExperimentPlan.load(ROOT / "experiments/iros2027/configs" / config_name)
    report = plan.validate(ROOT / "src/fr3_description/models/scene.xml")
    assert report["status"] == "contract_checks_passed"
    assert report["causality"]["future_samples_used"] == 0
    expected_fast = 120 if config_name.startswith("i002") else 200
    expected_tactile = 120 if config_name.startswith("i002") else 400
    assert report["sample_counts"] == {
        "controller": 2000,
        "tactile_sensor": expected_tactile,
        "fast_policy": expected_fast,
        "semantic_policy": 10,
    }
