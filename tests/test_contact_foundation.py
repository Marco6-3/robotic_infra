import json
from pathlib import Path

import numpy as np
import pytest

from fr3_control.tactile_residual import GripResidual
from experiments.iros2027.contact.history import EpisodeHistory


def test_residual_is_bounded_and_stale_holds_without_oracle():
    control = GripResidual()
    for _ in range(100):
        control.update([2, 0, 1.5, 2, 0, 1.5])
    assert control.offset_m == -.016
    assert control.compose(.032) == .016
    assert control.update(None) == -.016
    control.update([13, 0, 0, 13, 0, 0])
    assert control.offset_m > -.016


def test_history_respects_arrival_time_and_excludes_labels(tmp_path):
    samples = [dict(source_ns=i*5_000_000, available_ns=(i+2)*5_000_000, values=[i]*6) for i in range(5)]
    controls = [dict(timestamp_ns=i*1_000_000, arm_target_rad=[i]*7, applied_width_target_m=.03,
                     q=[0]*9, dq=[0]*9) for i in range(25)]
    for name, rows in [('tactile', samples), ('control', controls)]:
        (tmp_path/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    # A malformed privileged file must not be read by the policy data path.
    (tmp_path/'labels.jsonl').write_text('FORBIDDEN')
    history = EpisodeHistory(tmp_path)
    assert history.context(0, 100)['tactile'].shape == (0, 6)
    c = history.context(20_000_000, 100)
    assert c['source_ns'].tolist() == [0, 5_000_000, 10_000_000]
    assert c['action'][:, 0].tolist() == [0, 5, 10]
    assert history.context(20_000_000, 0)['tactile'].shape == (1, 6)
    assert set(c) == {'source_ns', 'available_ns', 'tactile', 'action', 'proprioception'}


@pytest.mark.integration
def test_actual_contact_closed_loop_and_delayed_failure(tmp_path):
    from experiments.iros2027.contact.run import run_episode
    results = []
    for name, mode, delay in [('nominal', 'nominal', 0), ('fast', 'fast', 0), ('delayed', 'fast', 100)]:
        output = tmp_path/name
        result = run_episode(seed=0, mode=mode, sensor_delay_ms=delay, output=output)
        results.append(result)
        assert result['lifted']
        assert result['counts'] == dict(tactile=800, control=4000, decisions=400, semantic=20)
        assert result['peak_finger_force_n'] > 1
        decisions = [json.loads(x) for x in (output/'decisions.jsonl').read_text().splitlines()]
        for row in decisions:
            if row['tactile_source_ns'] is not None:
                assert row['tactile_source_ns'] <= row['tactile_available_ns'] <= row['timestamp_ns']
            if row['semantic_source_ns'] is not None:
                assert row['semantic_source_ns'] <= row['timestamp_ns']
        assert EpisodeHistory(output).context(2_700_000_000, 100)['tactile'].shape == (21, 6)
    nominal, fast, delayed = results
    assert nominal['dropped'] and delayed['dropped']
    assert fast['retained'] and not fast['excessive_force']
    assert nominal['slip_onset_ns'] is not None
    assert fast['max_relative_displacement_m'] < .01
    assert delayed['max_tactile_age_ns'] == 100_000_000
    assert delayed['first_post_hold_correction_ns'] - fast['first_post_hold_correction_ns'] == 100_000_000


@pytest.mark.integration
def test_semantic_staleness_reaches_real_control():
    from experiments.iros2027.contact.run import run_episode
    result = run_episode(seed=0, mode='fast', semantic_delay_ms=300)
    assert result['semantic_channel']['delivered'] > 0
    assert result['semantic_channel']['stale_reads'] > 0
    assert result['stale_decisions'] == 400
    assert result['simulation_seconds'] == pytest.approx(4.)


@pytest.mark.integration
def test_saved_scene_preserves_randomized_friction_and_force_balance():
    import mujoco
    from experiments.iros2027.contact.task import DisturbedGrasp
    from fr3_sim.contact_proxy import ContactProxy
    root = Path(__file__).resolve().parents[1]
    env = DisturbedGrasp(root, seed=0)
    loaded = mujoco.MjModel.from_xml_string(env.xml)
    np.testing.assert_allclose(loaded.geom_friction, env.model.geom_friction)
    sensor = ContactProxy(env.model, env.object_body)
    for tick in range(2400):
        arm, width = env.nominal(tick)
        env.step(tick, arm, width)
    force, speed, contacts = sensor.truth(env.data)
    assert contacts > 0 and min(force[:, 0]) > 1
    # At the settled, nearly vertical grasp, local z is downward on both fingers.
    assert force[:, 2].sum() == pytest.approx(env.mass*9.81, abs=.15)
    assert speed < .003


def test_shared_configuration_rejects_typos_and_unsupported_rates(tmp_path):
    import yaml
    from experiments.iros2027.contact.settings import Settings, load_config
    with pytest.raises(ValueError, match='divisors'):
        Settings.from_mapping({'sensor': {'hz': 60}})
    with pytest.raises(ValueError):
        Settings.from_mapping({'task': {'mass_range_kg': [-1, .1]}})
    config = {'seeds': [0], 'conditions': [{'name': '../escape', 'mode': 'fast'}]}
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='names'):
        load_config(path)


@pytest.mark.integration
def test_observer_preserves_control_logs_and_config_changes_rates(tmp_path):
    from experiments.iros2027.contact.run import run_episode
    from experiments.iros2027.contact.settings import Settings
    seen = []

    class Observer:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def update(self, env, tick, diagnostics):
            seen.append(tick)

    settings = Settings.from_mapping({'sensor': {'hz': 100, 'noise_std_n': 0.0},
                                      'controller': {'fast_hz': 50}})
    a = run_episode(seed=0, settings=settings, output=tmp_path/'headless')
    b = run_episode(seed=0, settings=settings, output=tmp_path/'observed', observer_factory=lambda env: Observer())
    assert len(seen) == 4000
    assert a['counts']['tactile'] == b['counts']['tactile'] == 400
    assert a['counts']['decisions'] == 200
    for stream in ('control', 'tactile', 'labels', 'decisions'):
        assert (tmp_path/'headless'/f'{stream}.jsonl').read_bytes() == (tmp_path/'observed'/f'{stream}.jsonl').read_bytes()


def test_preview_controls_are_explicit():
    from experiments.iros2027.contact.play import Preview
    preview = Preview()
    preview.key(32)
    assert preview.paused.is_set()
    preview.key(32)
    assert not preview.paused.is_set()
    for key, command in [('R', 'restart'), ('N', 'next'), ('Q', 'quit')]:
        preview.key(ord(key))
        assert preview.command == command
