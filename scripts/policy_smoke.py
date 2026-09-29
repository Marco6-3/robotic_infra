#!/usr/bin/env python3
"""ACT CUDA optimization, checkpoint reload and bounded MuJoCo rollout.

This is an execution smoke test using a tiny local scripted batch, not a trained
manipulation task or a benchmark score. No pretrained weights are downloaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for package in ("fr3_robot_api", "fr3_control", "fr3_sim"):
    sys.path.insert(0, str(ROOT / "src" / package))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="resume this script's local checkpoint")
    args = parser.parse_args()
    if args.steps < 1 or args.frames < 2:
        parser.error("steps >= 1 and frames >= 2 are required")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from fr3_robot_api.gripper import GripperLimits, GripperMapper
    from fr3_robot_api.types import Action
    from fr3_robot_api.timing import physics_tick_for_frame
    from fr3_sim import MujocoRobot
    from verify_runtime import verify

    verify(require_lerobot=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this test must not silently fall back to CPU")
    torch.manual_seed(7)
    np.random.seed(7)
    torch.set_num_threads(4)
    device = "cuda"
    output = args.output or ROOT / "runs" / datetime.now(timezone.utc).strftime("act-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "scope": "ACT execution smoke test; tiny scripted batch; no task success or generalization claim",
        "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "seed": 7,
        "train_batch_size": 2, "eval_batch_size": 1, "num_workers": 0,
        "training_steps": args.steps, "rollout_frames": args.frames,
        "image_shape": [3, 96, 128], "amp": False,
        "action_contract": "7 joint residuals in units of 0.1 rad from home; normalized gripper width",
        "image_preprocessing": "RGB / 255; bilinear resize to 96x128; no additional normalization",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_checkpoint": str(args.resume) if args.resume else None,
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    config = ACTConfig(
        device=device, chunk_size=4, n_action_steps=1,
        normalization_mapping={name: NormalizationMode.IDENTITY for name in ("VISUAL", "STATE", "ACTION")},
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            **{f"observation.images.{name}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 128))
               for name in ("external", "wrist")},
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
        pretrained_backbone_weights=None, dim_model=128, n_heads=4,
        dim_feedforward=512, n_encoder_layers=2, n_decoder_layers=1,
        n_vae_encoder_layers=2, dropout=0.0,
    )
    policy = ACTPolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    completed = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=True)
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["step"]
        torch.set_rng_state(saved["rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
    metadata["starting_step"] = completed
    mapper = GripperMapper(GripperLimits(0, .04, 0, .04))
    robot = MujocoRobot(ROOT / "src/fr3_description/models/scene.xml", mapper, interpolation="hold")
    home = robot.data.qpos[:7].copy()
    metadata["home_joint_position_rad"] = home.tolist()
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def encode(obs):
        # Joint residuals use a fixed 0.1 rad scale; gripper uses normalized width.
        state = np.r_[(obs.joint_position_rad - home) / .1, obs.gripper_width_normalized]
        batch = {"observation.state": torch.as_tensor(state, dtype=torch.float32, device=device)[None]}
        for name, frame in (("external", obs.external_rgb), ("wrist", obs.wrist_rgb)):
            value = torch.tensor(frame.rgb, device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255
            batch[f"observation.images.{name}"] = F.interpolate(value, size=(96, 128), mode="bilinear", align_corners=False)
        return batch

    losses = []
    latencies = []
    clipped = 0
    try:
        examples = []
        targets = []
        for i in range(2):
            obs = robot.get_observation()
            examples.append(encode(obs))
            residual = np.zeros(8, dtype=np.float32)
            residual[0] = .4 * (i + 1)
            residual[-1] = .5
            targets.append(residual)
            robot.send_action(Action(obs.timestamp_ns, home + .1 * residual[:7], gripper_width_m=.04))
            for _ in range(34):
                robot.step()
        batch = {key: torch.cat([ex[key] for ex in examples]) for key in examples[0]}
        batch["action"] = torch.tensor(np.stack(targets), device=device)[:, None].repeat(1, config.chunk_size, 1)
        batch["action_is_pad"] = torch.zeros((2, config.chunk_size), dtype=torch.bool, device=device)
        checkpoint = output / "checkpoint.pt"

        def save_checkpoint(step):
            torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all()}, checkpoint)

        for step in range(completed, args.steps):
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss, details = policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append({"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(norm), **details})
            if step == completed or (step + 1) % 5 == 0 or step + 1 == args.steps:
                save_checkpoint(step + 1)
                print(json.dumps(losses[-1]), flush=True)
        if not checkpoint.exists():
            save_checkpoint(completed)
        policy.eval()
        before = policy.predict_action_chunk(examples[0])
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        policy.load_state_dict(saved["model"])
        after = policy.predict_action_chunk(examples[0])
        torch.testing.assert_close(before, after)
        policy.save_pretrained(output / "pretrained_model")
        robot.reset()
        policy.reset()
        start_q = robot.data.qpos[:7].copy()
        for frame in range(args.frames):
            obs = robot.get_observation()
            inputs = encode(obs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            predicted = policy.select_action(inputs)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
            raw = predicted[0].cpu().numpy()
            if raw.shape != (8,) or not np.isfinite(raw).all():
                raise RuntimeError("invalid ACT action")
            bounded = np.clip(raw, [-1] * 7 + [0], [1] * 8)
            clipped += int(np.any(raw != bounded))
            target = np.clip(home + .1 * bounded[:7], robot.model.jnt_range[:7, 0], robot.model.jnt_range[:7, 1])
            robot.send_action(Action(obs.timestamp_ns, target, gripper_width_normalized=float(bounded[7])))
            for _ in range(physics_tick_for_frame(frame + 1) - physics_tick_for_frame(frame)):
                robot.step()
            if not np.isfinite(robot.data.qpos).all():
                raise RuntimeError("non-finite simulation state")
        final = robot.get_observation()
        displacement = float(np.linalg.norm(final.joint_position_rad - start_q))
        if displacement <= 1e-4:
            raise RuntimeError("ACT rollout did not move the robot")
        for name, camera in (("external", final.external_rgb), ("wrist", final.wrist_rgb)):
            Image.fromarray(camera.rgb).save(output / f"{name}.png")
        metadata.update({
            "status": "passed", "checkpoint_reload": "passed", "losses": losses,
            "completed_step": saved["step"],
            "sim_time_s": robot.data.time, "joint_displacement_l2_rad": displacement,
            "clipped_frames": clipped,
            "inference_ms_median": float(np.median(latencies)),
            "inference_ms_p95": float(np.percentile(latencies, 95)),
            "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        })
        (output / "validation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"PASS: {output}", flush=True)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
