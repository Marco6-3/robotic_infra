import json
from pathlib import Path

import pytest
import yaml

from experiments.iros2027.common import RunArtifacts
from experiments.iros2027.i002.channel import ChannelConfig, DelayedLatestChannel
from experiments.iros2027.i002.contracts import SemanticMessage
from experiments.iros2027.i002.runtime import SmokeConfig, run_smoke


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments/iros2027/configs/i002_async_communication.yaml"


def _message(sequence: int, timestamp_ns: int) -> SemanticMessage:
    return SemanticMessage(sequence, timestamp_ns, (0.0,) * 7, (1.0, 2.0))


def test_channel_delays_reuses_and_expires_latest_value():
    channel = DelayedLatestChannel(
        ChannelConfig(20, 0, 0, max_queue=4, seed=1)
    )
    assert channel.send(_message(0, 0), now_ns=0)
    assert channel.read_latest(now_ns=19_000_000, max_age_ns=100_000_000) is None
    received = channel.read_latest(now_ns=20_000_000, max_age_ns=100_000_000)
    assert received is not None and received.age_ns == 20_000_000
    reused = channel.read_latest(now_ns=90_000_000, max_age_ns=100_000_000)
    assert reused is not None and reused.payload.sequence == 0
    assert channel.read_latest(now_ns=101_000_000, max_age_ns=100_000_000) is None
    assert channel.stats.stale_reads == 1


def test_channel_rejects_future_source_timestamp():
    channel = DelayedLatestChannel(ChannelConfig(0, 0, 0, 4, 1))
    with pytest.raises(ValueError):
        channel.send(_message(0, 1), now_ns=0)


def test_i002_smoke_is_deterministic_and_causal():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config = SmokeConfig.from_mapping(raw)
    first, _ = run_smoke(config)
    second, _ = run_smoke(config)
    assert first == second
    assert first["status"] == "communication_smoke_passed"
    assert first["future_context_uses"] == 0
    assert first["semantic_channel"]["sent"] == 10
    assert first["fast_decisions"] == 120
    assert first["feedback_channel"]["sent"] == 40


def test_run_directory_resume_keeps_attempt_history(tmp_path):
    first = RunArtifacts.create(
        root=ROOT,
        output_root=tmp_path,
        experiment_id="test",
        config_path=CONFIG,
    )
    first.finalize({"status": "ok"}, success=True)
    resumed = RunArtifacts.create(
        root=ROOT,
        output_root=tmp_path,
        experiment_id="test",
        config_path=CONFIG,
        resume=first.run_path,
    )
    resumed.finalize({"status": "ok-again"}, success=True)
    manifest = json.loads((first.run_path / "manifest.json").read_text(encoding="utf-8"))
    assert [attempt["status"] for attempt in manifest["attempts"]] == [
        "completed",
        "completed",
    ]
    assert manifest["attempts"][1]["resumed"] is True
