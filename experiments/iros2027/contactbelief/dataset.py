"""Allowlisted causal inputs; future observations live in separate target arrays."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import yaml
from experiments.iros2027.i001_v2.physics import resolution
from .utils import ROOT, sha, atomic

def source_config(config):
    return yaml.safe_load((ROOT/config['source_run']/'config.yaml').read_text())

def read_sample(path, config, regime):
    c = source_config(config)
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f['metadata']))
        ticks = f['ticks_ns'] // 1_000_000
        obs = f[regime]
        assert obs.shape[1] == 58 and np.isfinite(obs).all()
        indices = np.flatnonzero((ticks >= c['anchor_tick']-config['history_ms']) & (ticks <= c['anchor_tick']))
        assert len(indices) == 61 and np.all(f['available_ns'][indices] <= c['anchor_tick']*1_000_000)
        # x is the ONLY input to the outcome encoder. No metadata/labels arrays enter it.
        x = obs[indices].copy()
        targets, controls = [], []
        for horizon in config['horizons_ms']:
            steps = horizon // c['sample_ms']
            end = indices + steps
            assert np.array_equal(ticks[end], ticks[indices] + horizon)
            targets.append(obs[end, :50].copy())
            # Known commanded controls over [t,t+h), never measured future q/dq.
            # Each episode's open-loop probe schedule was sampled before rollout.
            # Keep mean and final command; the auxiliary head alone receives these.
            planned = np.stack([obs[indices+k, -8:] for k in range(steps)], axis=1)
            controls.append(np.concatenate([planned.mean(1), planned[:, -1]], axis=-1))
        return x, np.stack(targets, axis=1), np.stack(controls, axis=1), meta

def load_split(run, config, regime, split):
    if split not in ['train', 'validation', 'final']:
        raise ValueError('Old I001 test forbidden in training/final loader')
    if split == 'final':
        if not (run/'FROZEN.json').exists():
            raise RuntimeError('Final data access before freeze')
        directory = run/'episodes/final'
    else:
        directory = ROOT/config['source_run']/'episodes'/split
    paths = sorted(directory.glob('*.npz'))
    if not paths:
        raise ValueError('Empty split ' + str(directory))
    examples = [read_sample(p, config, regime) for p in paths]
    keep = [(p, row) for p, row in zip(paths, examples) if row[3]['eligible']]
    return dict(x=np.stack([r[0] for _, r in keep]).astype('float32'),
                future=np.stack([r[1] for _, r in keep]).astype('float32'),
                controls=np.stack([r[2] for _, r in keep]).astype('float32'),
                y=np.array([r[3]['label'] for _, r in keep], dtype='float32'),
                seeds=np.array([r[3]['seed'] for _, r in keep]),
                ids=np.array([p.name for p, _ in keep]),
                meta=[r[3] for _, r in keep])

def fit_normalizer(data, config):
    quantum = resolution(source_config(config)).astype('float32')
    mean = data['x'].mean((0, 1)); scale = np.maximum(data['x'].std((0, 1)), quantum)
    delta = data['future'] - data['x'][:, :, None, :50]
    # Train targets only. No validation/final values contribute to normalization.
    target_scale = np.maximum(delta.std((0, 1, 2)), quantum[:50])
    return dict(mean=mean, scale=scale, target_scale=target_scale)

def transform(data, norm):
    return dict(x=(data['x']-norm['mean'])/norm['scale'],
                future=(data['future']-data['x'][:, :, None, :50])/norm['target_scale'],
                controls=(data['controls']-np.tile(norm['mean'][-8:], 2))/np.tile(norm['scale'][-8:], 2),
                y=data['y'])

def audit_source(run, config):
    source = ROOT/config['source_run']; official = json.loads((source/'manifest.json').read_text())
    hashes = {}; split_seeds = {}
    for split in ['train', 'validation']:
        seeds = set()
        for p in sorted((source/'episodes'/split).glob('*.npz')):
            rel = str(p.relative_to(source)); hashes[rel] = sha(p)
            if hashes[rel] != official['episode_sha256'][rel]:
                raise RuntimeError('I001 source episode changed: '+rel)
            with np.load(p, allow_pickle=False) as f:
                seeds.add(json.loads(str(f['metadata']))['seed'])
        split_seeds[split] = sorted(seeds)
    assert not set(split_seeds['train']) & set(split_seeds['validation'])
    result = dict(source=str(source), episode_sha256=hashes, seeds=split_seeds,
                  old_test_used=False, input_channels=['tactile[32]', 'q[9]', 'dq[9]', 'command[8]'])
    atomic(run/'source_data_audit.json', result)
    return result
