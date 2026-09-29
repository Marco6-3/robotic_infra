"""One command: protected source -> train -> freeze -> probes -> new final -> gate."""
from __future__ import annotations
import argparse, fcntl, platform, shutil, sys, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import yaml
from .utils import ROOT, HERE, atomic, read, sha, now, progress, source_hashes, check_frozen, command
from .dataset import audit_source, load_split
from .train import fit_all, setup
from .representation_analysis import fit_probes, analyze
from .final_test import collect
from .evaluate import evaluate
from .report import report

def protected_hashes(config):
    roots=[ROOT/'experiments/iros2027/i001_v2',ROOT/config['source_run']]
    return {str(p.relative_to(ROOT)):sha(p) for r in roots for p in sorted(r.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.name!='.lock'}

def freeze(run,config,selection):
    if (run/'episodes/final').exists():raise RuntimeError('Final data exists before freeze')
    artifacts={}
    for selections in selection.values():
        for model in selections['models'].values():
            for record in model['replicates']:
                p=run/'candidates'/record['candidate']/'best.pt';artifacts[str(p.relative_to(run))]=sha(p)
    for p in (run/'normalizers').glob('*.npz'):artifacts[str(p.relative_to(run))]=sha(p)
    artifacts['selection.json']=sha(run/'selection.json')
    sources=source_hashes()
    for rel in sources:
        target=run/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    raw=load_split(run,config,'finite','train')
    atomic(run/'FROZEN.json',dict(time=now(),config_sha256=sha(run/'config.yaml'),source_sha256=sources,artifacts=artifacts,
                                 selection=selection,source_data_audit_sha256=sha(run/'source_data_audit.json'),
                                 training_positive_rate=float(raw['y'].mean()),architecture='encoder64/GRU64; predictive auxiliary head only',
                                 final_test_plan=dict(seeds=config['final_seed_count'],episodes_per_seed=20,adaptive=False)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=HERE/'config.yaml');p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if a.resume:
        run=a.resume.resolve();config=yaml.safe_load((run/'config.yaml').read_text())
    else:
        config=yaml.safe_load(a.config.read_text());run=ROOT/'runs/contactbelief'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        run.mkdir(parents=True);shutil.copy2(a.config,run/'config.yaml')
    lock=(run/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print('RUN='+str(run),flush=True)
    try:
        if (run/'COMPLETE.json').exists():
            check_frozen(run);print('Already complete; preserved outputs',flush=True);return
        setup(config)
        if not (run/'manifest.json').exists():
            manifest=dict(created=now(),python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
                          cuda=torch.version.cuda,device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                          batch_sizes=[config['train_batch_size'],config['eval_batch_size']],num_workers=config['num_workers'],
                          git=command(['git','status','--short']),source_sha256=source_hashes(),protected_i001=protected_hashes(config))
            atomic(run/'manifest.json',manifest);audit_source(run,config)
            (run/'REPRODUCE.txt').write_text(f'cd {ROOT}\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 .venv-recording/bin/python -m experiments.iros2027.contactbelief.run --config {run}/config.yaml\n\nResume:\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 .venv-recording/bin/python -m experiments.iros2027.contactbelief.run --resume {run}\n')
        if not (run/'FROZEN.json').exists():
            if read(run/'manifest.json')['source_sha256']!=source_hashes():raise RuntimeError('Development source drift; start a fresh run')
            progress(run,'training');selection=fit_all(run,config);freeze(run,config,selection)
        check_frozen(run)
        if not (run/'PROBES_FROZEN.json').exists():
            progress(run,'post_freeze_probe_fitting');fit_probes(run,config)
        if not (run/'final_data_audit.json').exists():collect(run,config)
        progress(run,'frozen_final_evaluation');evaluate(run,config)
        progress(run,'read_only_representation_analysis');analyze(run,config)
        verdict=report(run,config)
        progress(run,'regression_and_preservation',gate=verdict)
        regression=command([sys.executable,'-m','pytest','-q','-o','addopts=']);atomic(run/'regression.json',regression)
        if regression['returncode']:raise RuntimeError('Regression failed: '+regression['stderr'])
        before=read(run/'manifest.json')['protected_i001'];after=protected_hashes(config)
        if before!=after:raise RuntimeError('I001-v2 protection hash changed')
        atomic(run/'preservation_audit.json',dict(all_i001_v2_files_unchanged=True,files=len(before)))
        check_frozen(run)
        plot=command(['env','-u','PYTHONPATH','-u','PYTHONHOME','PYTHONNOUSERSITE=1','/usr/bin/python3',str(HERE/'plot.py'),str(run)])
        atomic(run/'plot_log.json',plot)
        if plot['returncode']:raise RuntimeError('Plotting failed: '+plot['stderr'])
        progress(run,'offline_complete',gate=verdict)
        atomic(run/'COMPLETE.json',dict(time=now(),offline_gate=verdict,
                                       requires_closed_loop=verdict['go'],artifacts={p.name:sha(p) for p in run.iterdir() if p.is_file() and p.name not in ['.lock','COMPLETE.json']}))
        print('COMPLETE='+str(run),flush=True)
    except BaseException as exc:
        progress(run,'failed',error=repr(exc));raise
    finally:lock.close()

if __name__=='__main__':main()
