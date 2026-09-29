"""Read-only audit of frozen auxiliary heads; does not alter gate or model selection."""
from pathlib import Path
import numpy as np, torch, yaml
from experiments.iros2027.contactbelief.utils import check_frozen, atomic
from experiments.iros2027.contactbelief.dataset import load_split,transform
from experiments.iros2027.contactbelief.train import setup,tensor_data,load_model
run=Path(__file__).resolve().parent
assert (run/'COMPLETE.json').exists(), 'Run only after the registered final evaluation completes'
c=yaml.safe_load((run/'config.yaml').read_text());frozen=check_frozen(run);setup(c);result={}
for regime in c['regimes']:
 raw=load_split(run,c,regime,'final')
 with np.load(run/'normalizers'/f'{regime}.npz') as f:norm=dict(f)
 data=tensor_data(transform(raw,norm),c['device'])
 models=[load_model(run,r,c) for r in frozen['selection'][regime]['models']['B4']['replicates']]
 totals={}
 with torch.no_grad():
  for start in range(0,len(raw['y']),c['eval_batch_size']):
   stop=min(start+c['eval_batch_size'],len(raw['y']));x=data['x'][start:stop];u=data['controls'][start:stop];target=data['future'][start:stop]
   prediction=torch.stack([m.predictive(m.encode(x)[0],u) for m in models]).mean(0)
   for period,sl in [('all_prefixes',slice(None)),('anchor',slice(-1,None))]:
    for h,horizon in enumerate(c['horizons_ms']):
     for name,features in [('tactile',slice(0,32)),('q',slice(32,41)),('dq',slice(41,50))]:
      error=(prediction[:,sl,h,features]-target[:,sl,h,features]).square().mean().item()
      persistence=target[:,sl,h,features].square().mean().item()
      key=f'{period}/{horizon}ms/{name}'
      if key not in totals:totals[key]=np.zeros(2)
      totals[key]+=np.array([error,persistence])*(stop-start)
 result[regime]={k:dict(contactbelief_normalized_mse=float(v[0]/len(raw['y'])),persistence_normalized_mse=float(v[1]/len(raw['y']))) for k,v in totals.items()}
atomic(run/'auxiliary_prediction_audit.json',dict(scope='post-hoc read-only; not used for gates, selection or model updates',metrics=result))
print(result)
