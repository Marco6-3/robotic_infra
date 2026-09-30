"""Independent timestamped Pilot; immutable provenance and stage resume."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,platform,shutil,subprocess,sys,time,traceback
import mujoco,numpy as np,torch,yaml
from .utils import atomic,read,sha,config,progress
from .dataset import collect
from .train import fit
from .evaluate import evaluate
from .diagnostics import diagnostics
from .analysis import analyze
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent


def sources():
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[ROOT/'tests/test_active_tactile_insertion.py',ROOT/'experiments/tactile_history_control/analysis.py',ROOT/'experiments/tactile_history_control/utils.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def freeze(run):
    files=list((run/'models').glob('*/final.pt'))+list((run/'data').glob('*/*.npz'))+[run/'normalizer.npz',run/'config.yaml']
    return dict(source=sources(),files={str(p.relative_to(run)):sha(p) for p in files})


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--full-scale',action='store_true');a=p.parse_args()
    if a.resume:
        run=a.resume.resolve();c=config(run);m=read(run/'manifest.json')
        if m['source']!=sources() or m['config_sha256']!=sha(run/'config.yaml'):raise RuntimeError('Resume source/config changed')
        if (run/'FROZEN.json').exists() and read(run/'FROZEN.json')!=freeze(run):raise RuntimeError('Frozen data/model changed')
    else:
        c=yaml.safe_load((HERE/'config.yaml').read_text())
        if a.smoke:
            c.update(stage='Smoke',train_episodes=4,validation_episodes=2,eval_episodes=4,train_steps=4,checkpoint_every=4,train_seeds=[17]);c['ambiguity']['seeds']=2
            c['seed_starts']={k:v+100000 for k,v in c['seed_starts'].items()}
        if a.full_scale:
            c.update(stage='Expanded Pilot',train_episodes=192,validation_episodes=48,eval_episodes=192,train_steps=2000,train_seeds=[17,29,43,59,71]);c['ambiguity']['seeds']=64
            c['seed_starts']={k:v+200000 for k,v in c['seed_starts'].items()}
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run=ROOT/'docs/research-results/active-tactile-insertion'/(stamp+('-smoke' if a.smoke else '-pilot'));run.mkdir(parents=True,exist_ok=False)
        (run/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
        for name in ['PROTOCOL.md','LITERATURE.md']:shutil.copy(HERE/name,run/name)
        atomic(run/'manifest.json',dict(source=sources(),config_sha256=sha(run/'config.yaml'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        for rel in sources():
            dst=run/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy(ROOT/rel,dst)
        (run/'git.diff').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT));(run/'git-status.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=ROOT))
        (run/'pip-freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']));(run/'nvidia-smi.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
        atomic(run/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,mujoco=mujoco.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),device=c['device'],collection_processes=c['workers'],train_data_loader_workers=0,train_batch_size=c['train_batch_size']))
        for path in Path('/tmp').glob('insertion-development-*.json'):shutil.copy(path,run/path.name)
        for name in ['insertion-cylinder-reproduced.json','insertion-environment-v1.py']:
            if (Path('/tmp')/name).exists():shutil.copy(Path('/tmp')/name,run/name)
    print('RUN='+str(run),flush=True);timing=read(run/'timing.json') if (run/'timing.json').exists() else {}
    try:
        for stage,fn in [('collect',collect),('train',fit),('evaluate',evaluate),('diagnostics',diagnostics)]:
            if (run/f'{stage}.complete.json').exists():continue
            if stage=='evaluate':
                f=freeze(run)
                if (run/'FROZEN.json').exists():assert read(run/'FROZEN.json')==f
                else:atomic(run/'FROZEN.json',f)
            begin=time.perf_counter();progress(run,stage+'-start');fn(run,c);timing[stage]=time.perf_counter()-begin
            atomic(run/'timing.json',timing);atomic(run/f'{stage}.complete.json',dict(seconds=timing[stage]))
        r=analyze(run,c);atomic(run/'COMPLETE.json',dict(status=r['status'],go_no_go=r['go_no_go']));progress(run,'complete',status=r['status'],go_no_go=r['go_no_go'])
    except Exception as exc:
        atomic(run/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
