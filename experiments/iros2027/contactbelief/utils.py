from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)

def read(path):
    return json.loads(Path(path).read_text())

def progress(run, stage, **details):
    record = dict(time=now(), stage=stage, **details)
    atomic(run/'progress.json', record)
    with (run/'events.jsonl').open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)

def source_hashes():
    from experiments.iros2027.i001_v2.run import sources
    paths = set(sources()) | set(HERE.glob('*.py')) | set(HERE.glob('*.md')) | set(HERE.glob('*.yaml')) | {ROOT/'tests/test_contactbelief.py'}
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}

def check_frozen(run):
    f = read(run/'FROZEN.json')
    if f['source_sha256'] != source_hashes():
        raise RuntimeError('Frozen source drift: no model/code changes allowed after freeze')
    if sha(run/'config.yaml') != f['config_sha256']:
        raise RuntimeError('Frozen config drift')
    for rel, digest in f['artifacts'].items():
        if sha(run/rel) != digest:
            raise RuntimeError('Frozen artifact drift: ' + rel)
    return f

def command(args):
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

def cluster(values, seeds, config):
    values, seeds = np.asarray(values), np.asarray(seeds)
    unique = np.unique(seeds)
    if len(values) == 0:
        return dict(mean=None, ci95=[None, None], seeds=0, n=0)
    sums = np.array([values[seeds == s].sum() for s in unique])
    counts = np.array([(seeds == s).sum() for s in unique])
    idx = np.random.default_rng(config['bootstrap_seed']).integers(
        len(unique), size=(config['bootstrap_samples'], len(unique)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                seeds=len(unique), n=len(values))
