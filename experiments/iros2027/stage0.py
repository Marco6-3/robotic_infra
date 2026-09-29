"""Stage-0 contract checks for the IROS 2027 research ideas.

This module does not claim task performance.  It validates the experiment
configuration, produces causal multi-rate schedules, and records enough
provenance to make later physics and policy runs inspectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = Path(__file__).resolve().parent / "configs"


def _positive_number(mapping: dict[str, Any], name: str) -> float:
    value = float(mapping[name])
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def periodic_timestamps_ns(rate_hz: float, duration_s: float) -> list[int]:
    """Return drift-bounded periodic timestamps in ``[0, duration_s)``."""
    if rate_hz <= 0 or duration_s <= 0:
        raise ValueError("rate_hz and duration_s must be positive")
    count = int(rate_hz * duration_s + 1e-12)
    return [round(index * 1_000_000_000 / rate_hz) for index in range(count)]


def latest_causal_age_ns(query_ns: int, source_timestamps_ns: Iterable[int]) -> int:
    """Age of the latest source value available no later than ``query_ns``."""
    latest = None
    for timestamp_ns in source_timestamps_ns:
        if timestamp_ns > query_ns:
            break
        latest = timestamp_ns
    if latest is None:
        raise ValueError(f"no causal source sample is available at {query_ns} ns")
    return query_ns - latest


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    idea: str
    duration_s: float
    controller_hz: float
    tactile_sensor_hz: float
    fast_policy_hz: float
    semantic_policy_hz: float
    config_path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "ExperimentPlan":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        experiment = raw["experiment"]
        rates = raw["rates"]
        return cls(
            experiment_id=str(experiment["id"]),
            idea=str(experiment["idea"]),
            duration_s=_positive_number(experiment, "duration_s"),
            controller_hz=_positive_number(rates, "controller_hz"),
            tactile_sensor_hz=_positive_number(rates, "tactile_sensor_hz"),
            fast_policy_hz=_positive_number(rates, "fast_policy_hz"),
            semantic_policy_hz=_positive_number(rates, "semantic_policy_hz"),
            config_path=path,
            raw=raw,
        )

    def validate(self, scene_path: Path) -> dict[str, Any]:
        if self.controller_hz < max(
            self.tactile_sensor_hz, self.fast_policy_hz, self.semantic_policy_hz
        ):
            raise ValueError("controller_hz must be the highest configured rate")
        if self.fast_policy_hz > self.tactile_sensor_hz:
            raise ValueError("fast_policy_hz cannot exceed tactile_sensor_hz in Stage 0")

        semantic = periodic_timestamps_ns(self.semantic_policy_hz, self.duration_s)
        fast = periodic_timestamps_ns(self.fast_policy_hz, self.duration_s)
        tactile = periodic_timestamps_ns(self.tactile_sensor_hz, self.duration_s)
        controller = periodic_timestamps_ns(self.controller_hz, self.duration_s)
        semantic_ages = [latest_causal_age_ns(timestamp, semantic) for timestamp in fast]
        tactile_ages = [latest_causal_age_ns(timestamp, tactile) for timestamp in fast]

        root = ET.parse(scene_path).getroot()
        geom_names = {item.get("name") for item in root.findall(".//geom")}
        site_names = {item.get("name") for item in root.findall(".//site")}
        required = self.raw.get("scene", {}).get("required_names", [])
        available = geom_names | site_names
        missing = sorted(name for name in required if name not in available)
        if missing:
            raise ValueError(f"scene is missing required named elements: {missing}")

        config_bytes = self.config_path.read_bytes()
        return {
            "experiment_id": self.experiment_id,
            "idea": self.idea,
            "status": "contract_checks_passed",
            "config": str(self.config_path.relative_to(ROOT)),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "scene": str(scene_path.relative_to(ROOT)),
            "scene_required_names": required,
            "duration_s": self.duration_s,
            "sample_counts": {
                "controller": len(controller),
                "tactile_sensor": len(tactile),
                "fast_policy": len(fast),
                "semantic_policy": len(semantic),
            },
            "causality": {
                "future_samples_used": 0,
                "max_tactile_age_ms": max(tactile_ages, default=0) / 1_000_000,
                "max_semantic_age_ms": max(semantic_ages, default=0) / 1_000_000,
            },
            "claim_boundary": (
                "Only configuration, scene anchors, rates, and causal schedule were checked; "
                "no policy quality or manipulation success was measured."
            ),
        }


def run(config_paths: list[Path], scene_path: Path) -> dict[str, Any]:
    reports = [ExperimentPlan.load(path).validate(scene_path) for path in config_paths]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append")
    parser.add_argument(
        "--scene",
        type=Path,
        default=ROOT / "src/fr3_description/models/scene.xml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configs = args.config or sorted(DEFAULT_CONFIGS.glob("*.yaml"))
    if not configs:
        parser.error("no experiment configs found")
    report = run(configs, args.scene)
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = ROOT / "runs" / "iros-stage0" / stamp / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
