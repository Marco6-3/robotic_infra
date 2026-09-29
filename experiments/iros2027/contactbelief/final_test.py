"""New random physics seeds are reserved only AFTER model and probe freezing."""
from __future__ import annotations
import argparse, multiprocessing, re, secrets
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import yaml
from experiments.iros2027.i001_v2.physics import simulate
from experiments.iros2027.i001_v2.run import validate_episode
from experiments.iros2027.i001_v2.data import load_rows, stats
from .dataset import source_config
from .utils import ROOT, check_frozen, read, atomic, sha, now, progress

def reserve_seeds(run,config):
    check_frozen(run)
    probes=read(run/'PROBES_FROZEN.json')
    for rel,digest in probes['artifacts'].items():assert sha(run/rel)==digest
    if (run/'FINAL_SEEDS.json').exists():return read(run/'FINAL_SEEDS.json')['seeds']
    used=set()
    # Filename inventory only: old test values and scores are not read.
    for path in (ROOT/'runs').rglob('*.npz'):
        m=re.search(r'-(\d+)-\d+\.npz$',path.name)
        if m:used.add(int(m.group(1)))
    while True:
        first=100_000_000+secrets.randbelow(800_000_000)
        seeds=list(range(first,first+config['final_seed_count']))
        if not used.intersection(seeds):break
    atomic(run/'FINAL_SEEDS.json',dict(time=now(),seeds=seeds,previous_seed_inventory_count=len(used),
                                     disjoint=True,model_freeze_sha256=sha(run/'FROZEN.json'),
                                     probe_freeze_sha256=sha(run/'PROBES_FROZEN.json')))
    return seeds

def collect(run,config):
    seeds=reserve_seeds(run,config);c=source_config(config);directory=run/'episodes/final';directory.mkdir(parents=True,exist_ok=True)
    manifest_path=run/'final_data_manifest.json';manifest=read(manifest_path) if manifest_path.exists() else dict(started=now(),episode_sha256={})
    jobs=[]
    for seed in seeds:
        for v in range(c['episodes_per_seed']):
            name=f'final-{seed}-{v:02}.npz';p=directory/name
            if p.exists():
                validate_episode(p,c)
                if name in manifest['episode_sha256'] and sha(p)!=manifest['episode_sha256'][name]:raise RuntimeError('Final data hash mismatch')
                manifest['episode_sha256'][name]=sha(p)
            else:jobs.append((seed,v,'final',c,str(directory)))
    atomic(manifest_path,manifest)
    with ProcessPoolExecutor(max_workers=config['simulation_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for meta in pool.map(simulate,jobs):
            p=directory/meta['episode'];validate_episode(p,c);manifest['episode_sha256'][p.name]=sha(p)
            if len(manifest['episode_sha256'])%40==0:
                atomic(manifest_path,manifest);progress(run,'collect_final',completed=len(manifest['episode_sha256']),total=len(seeds)*c['episodes_per_seed'])
    manifest['completed']=now();atomic(manifest_path,manifest)
    rows=load_rows(run,c,'final');summary=stats(rows,c)
    assert set(r['seed'] for r in rows)==set(seeds)
    assert summary['episodes']==len(seeds)*c['episodes_per_seed']
    assert summary['positive']>0 and summary['negative']>0
    for seed in seeds:assert len({r['future_hash'] for r in rows if r['seed']==seed})==1
    replay=run/'replay';replay.mkdir(exist_ok=True);meta=simulate((seeds[0],0,'final',c,str(replay)))
    with np.load(directory/meta['episode']) as a,np.load(replay/meta['episode']) as b:
        for k in a.files:
            if k!='metadata':np.testing.assert_array_equal(a[k],b[k])
    atomic(run/'final_data_audit.json',dict(**summary,identical_future_inputs_per_seed=True,deterministic_replay=True,
                                         frozen_before_seeds=read(run/'FROZEN.json')['time']<read(run/'FINAL_SEEDS.json')['time']<manifest['started']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();collect(a.run,yaml.safe_load((a.run/'config.yaml').read_text()))
