"""Run the deterministic I002 communication smoke and persist its evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from experiments.iros2027.common import RunArtifacts
from experiments.iros2027.i002.runtime import SmokeConfig, run_smoke


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiments/iros2027/configs/i002_async_communication.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/i002")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = SmokeConfig.from_mapping(raw)
    artifacts = RunArtifacts.create(
        root=ROOT,
        output_root=args.output_root,
        experiment_id=str(raw["experiment"]["id"]),
        config_path=args.config,
        resume=args.resume,
    )
    try:
        metrics, events = run_smoke(config)
        for event in events:
            artifacts.append_jsonl("events.jsonl", event)
        artifacts.append_jsonl("metrics.jsonl", metrics)
        artifacts.finalize(metrics, success=True)
    except Exception as exc:
        failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        artifacts.append_jsonl("metrics.jsonl", failure)
        artifacts.finalize(failure, success=False)
        raise
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"run_dir: {artifacts.run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
