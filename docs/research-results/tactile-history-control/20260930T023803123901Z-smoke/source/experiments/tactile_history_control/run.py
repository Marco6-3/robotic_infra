"""Timestamped, resumable Pilot lifecycle; fixed protocol before test access."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,os,platform,shutil,subprocess,time,traceback
import mujoco,numpy as np,torch,yaml
from .utils import atomic,config,progress,read,sha
from .dataset import collect
from .train import fit
from .ablation_runner import run_ablations
from .analysis import analyze
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent

def source_manifest():
    paths=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md',ROOT/'experiments/iros2027/contact/task.py',ROOT/'experiments/iros2027/i001_v2/physics.py',ROOT/'src/fr3_sim/fr3_sim/contact_proxy.py',ROOT/'src/fr3_description/models/fr3_hand.xml',ROOT/'src/fr3_description/models/scene.xml']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--full-scale',action='store_true');p.add_argument('--config',type=Path,default=HERE/'config.yaml');a=p.parse_args()
    if a.smoke and a.full_scale:raise ValueError('choose one run budget')
    if a.resume:
        run=a.resume.resolve();c=config(run);manifest=read(run/'manifest.json')
        if manifest['source']!=source_manifest() or manifest['config_sha256']!=sha(run/'config.yaml'):raise RuntimeError('Resume source/config mismatch: start a new run')
    else:
        c=yaml.safe_load(a.config.read_text())
        if a.smoke:
            c.update(stage='Smoke',train_episodes=4,validation_episodes=2,eval_episodes=4,train_steps=3,train_seeds=[17],checkpoint_every=3)
            c['ambiguity'].update(seeds=2,variants=2);c['seed_starts']={k:v+100000 for k,v in c['seed_starts'].items()}
        if a.full_scale:
            c.update(stage='Expanded Pilot',train_episodes=192,validation_episodes=48,eval_episodes=192,train_steps=2000,train_seeds=[17,29,43,59,71])
            c['ambiguity'].update(seeds=64);c['seed_starts']={k:v+200000 for k,v in c['seed_starts'].items()}
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run=ROOT/'docs/research-results/tactile-history-control'/(stamp+('-smoke' if a.smoke else '-pilot'))
        run.mkdir(parents=True,exist_ok=False);(run/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
        shutil.copy(HERE/'PROTOCOL.md',run/'PROTOCOL.md')
        manifest=dict(source=source_manifest(),config_sha256=sha(run/'config.yaml'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        atomic(run/'manifest.json',manifest)
        (run/'git.diff').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT))
        (run/'git-status.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=ROOT))
        for relative in manifest['source']:
            target=run/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy(ROOT/relative,target)
        atomic(run/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,mujoco=mujoco.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,train_batch_size=c['train_batch_size'],num_workers=c['workers'],device=c['device']))
        for name in ['thc-development-v1.json','thc-development-v2.json','thc-ambiguity-development.json','thc-ambiguity-development-v2.json']:
            path=Path('/tmp')/name
            if path.exists():shutil.copy(path,run/name)
    if (run/'FROZEN.json').exists():
        frozen=read(run/'FROZEN.json')
        if frozen['normalizer_sha256']!=sha(run/'normalizer.npz') or any(sha(run/p)!=h for p,h in frozen['checkpoints'].items()):raise RuntimeError('Frozen checkpoint/normalizer changed')
    print('RUN='+str(run),flush=True);timing=read(run/'timing.json') if (run/'timing.json').exists() else {}
    try:
        for name,fn in [('collect',collect),('train',fit),('evaluate',run_ablations)]:
            marker=run/f'{name}.complete.json'
            if marker.exists():continue
            if name=='evaluate':
                freeze=dict(config_sha256=sha(run/'config.yaml'),source=source_manifest(),checkpoints={str(p.relative_to(run)):sha(p) for p in (run/'models').glob('*/final.pt')},normalizer_sha256=sha(run/'normalizer.npz'))
                if (run/'FROZEN.json').exists():
                    if read(run/'FROZEN.json')!=freeze:raise RuntimeError('Frozen evidence changed')
                else:atomic(run/'FROZEN.json',freeze)
            begin=time.perf_counter();progress(run,name+'-start');fn(run,c);timing[name]=time.perf_counter()-begin
            atomic(run/'timing.json',timing);atomic(marker,dict(seconds=timing[name]))
        result=analyze(run,c);atomic(run/'COMPLETE.json',dict(status=result['status'],decision=result['decision']));progress(run,'complete',status=result['status'])
    except Exception as exc:
        atomic(run/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
