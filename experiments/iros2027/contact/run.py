"""Run paired real-physics episodes; all scheduling uses simulation time."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
for package in ('fr3_sim', 'fr3_control', 'fr3_robot_api'):
    sys.path.insert(0, str(ROOT / 'src' / package))

import mujoco
import numpy as np
import yaml
from fr3_sim.contact_proxy import ContactProxy
from fr3_control.tactile_residual import GripResidual
from experiments.iros2027.contact.task import DisturbedGrasp
from experiments.iros2027.contact.settings import Settings, load_config
from experiments.iros2027.common.run_artifacts import RunArtifacts
from experiments.iros2027.i002.channel import ChannelConfig, DelayedLatestChannel
from experiments.iros2027.i002.contracts import SemanticMessage


def run_episode(*, seed=0, mode='fast', sensor_delay_ms=0, semantic_delay_ms=0, disturbance_scale=1.0, output=None, settings=None, observer_factory=None):
    if mode not in ('nominal', 'fast'):
        raise ValueError('mode must be nominal or fast')
    if not np.isfinite(disturbance_scale) or disturbance_scale < 0:
        raise ValueError('disturbance_scale must be finite and non-negative')
    settings = settings or Settings()
    env = DisturbedGrasp(ROOT, seed=seed, settings=settings.task)
    env.disturbance_n *= disturbance_scale
    sensor = ContactProxy(env.model, env.object_body, seed=seed, delay_ms=sensor_delay_ms, noise_std=settings.sensor.noise_std_n)
    c = settings.controller
    residual = GripResidual(max_closure_m=c.max_closure_m, step_m=c.step_m,
                            force_limit_n=c.force_limit_n, shear_ratio_threshold=c.shear_ratio_threshold)
    channel = DelayedLatestChannel(ChannelConfig(semantic_delay_ms, 0, 0, 64, seed))
    counts = dict(tactile=0, control=0, decisions=0, semantic=0)
    peak_force = 0.; slip_onset = None; first_correction = None; drop = False
    slip_ticks = 0; max_displacement = 0.; reference = None
    lifted = False; stable_tail = []; ages = []; stale_count = 0
    recovery_ns = None; stable_ticks = 0
    semantic_age = None; nominal_width = .032; tactile_source = None
    start = time.perf_counter()
    with ExitStack() as stack:
        observer = stack.enter_context(observer_factory(env)) if observer_factory else None
        streams = {}
        if output is not None:
            output = Path(output); output.mkdir(parents=True, exist_ok=False)
            (output / 'scene.xml').write_text(env.xml)
            for name in ('tactile', 'labels', 'control', 'decisions', 'semantic'):
                streams[name] = stack.enter_context((output / f'{name}.jsonl').open('w'))

        def write(name, row):
            if name in streams:
                streams[name].write(json.dumps(row, allow_nan=False) + '\n')

        for tick in range(4000):
            if observer is not None:
                observer.update(env, tick, {"mode": mode, "residual_m": residual.offset_m, "dropped": bool(drop)})
            now = tick * 1_000_000
            forces, speed, contacts = sensor.truth(env.data)
            pos = env.object_position()
            tcp = env.data.site_xpos[env.tcp]
            relative = pos - tcp
            if tick == 2400:
                reference = relative.copy()
                lifted = bool(pos[2] > .49 and min(forces[:, 0]) > .15)
            if reference is not None:
                distance = float(np.linalg.norm(relative - reference))
                max_displacement = max(max_displacement, distance)
                drop |= distance > .04 or pos[2] < .46
                slip_ticks = slip_ticks + 1 if speed > .01 and contacts else 0
                if slip_ticks >= 5 and slip_onset is None:
                    slip_onset = now - 4_000_000
            stable = bool(speed < .003 and min(forces[:, 0]) > .15 and pos[2] > .49)
            if tick >= 2800:
                stable_ticks = stable_ticks + 1 if stable and not drop else 0
                if stable_ticks >= 100 and recovery_ns is None:
                    recovery_ns = now
            peak_force = max(peak_force, float(forces[:, 0].max()))
            write('labels', dict(timestamp_ns=now, normal_shear_n=forces.tolist(),
                                 relative_tangent_speed_m_s=speed, object_position_m=pos.tolist(),
                                 object_minus_tcp_m=relative.tolist(), contacts=contacts,
                                 slip_active=slip_ticks >= 5, dropped=bool(drop),
                                 disturbance_force_world_n=[0., 0., -env.disturbance_n * min((tick - 2500) / 100, 1.) if 2500 <= tick < 2800 else 0.]))
            if tick % (1000 // settings.sensor.hz) == 0:
                sample = sensor.sample(now, forces)
                counts['tactile'] += 1
                write('tactile', asdict(sample))
            if tick % (1000 // c.semantic_hz) == 0:
                # Scripted slow intent is deliberately independent of contact truth.
                width = .034 if 2400 <= tick < 3000 else .032
                message = SemanticMessage(tick // (1000 // c.semantic_hz), now, (width,), ())
                channel.send(message, now_ns=now)
                counts['semantic'] += 1
                write('semantic', dict(source_ns=now, nominal_width_m=width,
                                       scheduled_delivery_ns=now + round(semantic_delay_ms * 1e6)))
            if tick % (1000 // c.fast_hz) == 0:
                sample = sensor.read(now, max_age_ns=round(settings.sensor.max_age_ms * 1e6))
                received = channel.read_latest(now_ns=now, max_age_ns=round(c.semantic_max_age_ms * 1e6))
                semantic_age = None if received is None else received.age_ns
                nominal_width = .032 if received is None else received.payload.coarse_action[0]
                tactile_source = None if sample is None else sample.source_ns
                if sample is not None:
                    assert sample.available_ns <= now and sample.source_ns <= now
                    ages.append(now - sample.source_ns)
                stale_count += int(sample is None or received is None)
                before = residual.offset_m
                # Startup/approach uses the nominal scripted controller. The fast
                # loop begins after lift and only sees force proxy + slow intent.
                if mode == 'fast' and tick >= 2100:
                    residual.update(None if sample is None else sample.values)
                if residual.offset_m < before and first_correction is None and tick >= 2400:
                    first_correction = now
                counts['decisions'] += 1
                write('decisions', dict(timestamp_ns=now, tactile_source_ns=tactile_source,
                                       tactile_available_ns=None if sample is None else sample.available_ns,
                                       semantic_source_ns=None if received is None else received.payload.source_timestamp_ns,
                                       semantic_age_ns=semantic_age, residual_m=residual.offset_m,
                                       nominal_width_m=nominal_width))
            arm, scripted_width = env.nominal(tick)
            width = scripted_width if tick < 2100 else residual.compose(nominal_width)
            counts['control'] += 1
            write('control', dict(timestamp_ns=now, arm_target_rad=arm.tolist(),
                                  applied_width_target_m=width, residual_m=residual.offset_m,
                                  q=env.data.qpos[:9].tolist(), dq=env.data.qvel[:9].tolist()))
            if tick >= 3800:
                stable_tail.append(bool(speed < .003 and min(forces[:, 0]) > .15 and pos[2] > .49))
            env.step(tick, arm, width)
    summary = dict(seed=seed, mode=mode, sensor_delay_ms=sensor_delay_ms,
                   semantic_delay_ms=semantic_delay_ms, disturbance_scale=disturbance_scale, mass_kg=env.mass,
                   friction=env.friction, disturbance_n=env.disturbance_n,
                   counts=counts, lifted=lifted, dropped=bool(drop),
                   retained=bool(lifted and not drop and all(stable_tail)),
                   recovery_confirmed_ns=recovery_ns,
                   recovery_after_disturbance_ms=None if recovery_ns is None else (recovery_ns-2_800_000_000)/1e6,
                   slip_onset_ns=slip_onset, first_post_hold_correction_ns=first_correction,
                   max_relative_displacement_m=max_displacement, peak_finger_force_n=peak_force,
                   excessive_force=peak_force > c.force_limit_n,
                   max_tactile_age_ns=max(ages, default=0), stale_decisions=stale_count,
                   semantic_channel=channel.stats.as_dict(), wall_seconds=time.perf_counter()-start,
                   simulation_seconds=env.data.time,
                   boundary='Known-pose rigid-contact proxy and rule control; no RGB tactile, learned model, or real-time guarantee.')
    if output is not None:
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/iros2027/contact/config.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/contact-foundation')
    args = parser.parse_args()
    config = load_config(args.config)
    settings = Settings.from_mapping(config)
    run = RunArtifacts.create(root=ROOT, output_root=args.output,
                              experiment_id='contact_foundation', config_path=args.config)
    sources = [Path(__file__), ROOT / 'experiments/iros2027/contact/task.py',
               ROOT / 'src/fr3_sim/fr3_sim/contact_proxy.py',
               ROOT / 'src/fr3_control/fr3_control/tactile_residual.py',
               ROOT / 'src/fr3_description/models/fr3_hand.xml',
               ROOT / 'src/fr3_description/models/scene.xml']
    run.manifest['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    sources.extend([ROOT / 'experiments/iros2027/i002/channel.py',
                    ROOT / 'experiments/iros2027/i002/contracts.py',
                    ROOT / 'experiments/iros2027/contact/history.py',
                    ROOT / 'experiments/iros2027/contact/settings.py'])
    run.manifest['source_sha256'].update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for source in sources:
        target = run.run_path / 'source' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    assets = ROOT / 'src/fr3_description/models/assets'
    run.manifest['asset_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(assets.iterdir()) if p.is_file()}
    run.manifest['resolved_settings'] = asdict(settings)
    run.manifest['mujoco_version'] = mujoco.__version__
    run._write_manifest()
    results = []
    try:
        for seed in config['seeds']:
            for condition in config['conditions']:
                values = {k: v for k, v in condition.items() if k != 'name'}
                result = run_episode(seed=seed, settings=settings, output=run.attempt_path / f"seed-{seed}-{condition['name']}", **values)
                result['condition'] = condition['name']
                results.append(result)
                print(json.dumps(result), flush=True)
        run.finalize(dict(episodes=results, boundary='Development seeds; no statistical or learned-policy claim.'), success=True)
    except BaseException as exc:
        run.finalize(dict(error=str(exc), completed_episodes=results), success=False)
        raise
    print(f'Results: {run.run_path}')


if __name__ == '__main__':
    main()
