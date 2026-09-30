"""Fixed steps/final checkpoint; validation is logging only, no model selection."""
from pathlib import Path
import argparse,time
import numpy as np
import torch
from .policy import Policy,normalize
from .dataset import load_windows
from .utils import atomic,config,progress,save_pt,sha,setup

def fit(run,c):
    setup(c)
    if (run/'FROZEN.json').exists():raise RuntimeError('Training forbidden after evaluation freeze')
    norm=np.load(run/'normalizer.npz');raw,y=load_windows(run,c,'train');vraw,vy=load_windows(run,c,'validation')
    x=torch.tensor(normalize(raw,norm['mean'],norm['scale']),device=c['device']);y=torch.tensor(y,device=c['device'])
    vx=torch.tensor(normalize(vraw,norm['mean'],norm['scale']),device=c['device']);vy=torch.tensor(vy,device=c['device'])
    for seed in c['train_seeds']:
        for method in ['M0','M1','M2']:
            directory=run/'models'/f'{method}-{seed}';directory.mkdir(parents=True,exist_ok=True)
            if (directory/'complete.json').exists():continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=Policy(c,method).to(c['device']);opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
            start=0;curve=[];begin=time.perf_counter()
            if (directory/'latest.pt').exists():
                state=torch.load(directory/'latest.pt',weights_only=False,map_location=c['device'])
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['step'];curve=state['curve']
            for step in range(start,c['train_steps']):
                # Identical minibatches for all methods; no dropout -> exact resume RNG.
                ix=np.random.default_rng([seed,step]).integers(0,len(x),c['train_batch_size'])
                model.train();opt.zero_grad(set_to_none=True)
                pred=model(x[ix]);loss=(pred-y[ix]).square().mean()
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip']);opt.step()
                if (step+1)%c['checkpoint_every']==0 or step==0 or step+1==c['train_steps']:
                    model.eval();errs=[]
                    with torch.no_grad():
                        for j in range(0,len(vx),256):errs.append((model(vx[j:j+256])-vy[j:j+256]).square().sum().item())
                    rec=dict(step=step+1,train_mse=loss.item(),validation_mse=sum(errs)/len(vx));curve.append(rec)
                    save_pt(directory/'latest.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,curve=curve))
                    atomic(directory/'curve.json',curve);progress(run,'train',method=method,seed=seed,**rec)
            save_pt(directory/'final.pt',dict(model=model.state_dict(),method=method,seed=seed,step=c['train_steps']))
            atomic(directory/'complete.json',dict(parameters=sum(p.numel() for p in model.parameters()),steps=c['train_steps'],samples=len(x),
                 seed=seed,method=method,seconds_this_attempt=time.perf_counter()-begin,sha256=sha(directory/'final.pt'),final_validation_mse=curve[-1]['validation_mse']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();fit(a.run,config(a.run))
