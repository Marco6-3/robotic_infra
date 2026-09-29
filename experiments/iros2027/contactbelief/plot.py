"""Standalone host-Matplotlib renderer; no ML/simulator imports."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

run=Path(sys.argv[1]);out=run/'figures';out.mkdir(exist_ok=True)
metrics=json.loads((run/'metrics.json').read_text());selection=json.loads((run/'selection.json').read_text())
names=['B0','B1','B2','B3','B4'];labels=['Current','Stack','GRU','Transformer','ContactBelief']
fig,axes=plt.subplots(1,3,figsize=(14,4))
for offset,(regime,color) in enumerate([('finite','#2364aa'),('exact','#e59500')]):
    models=metrics[regime]['models']
    for ax,key in zip(axes,['brier','auroc','pair']):
        vals=[models[n]['matched_pair']['mean'] if key=='pair' else models[n][key] for n in names]
        cis=[models[n]['matched_pair']['ci95'] if key=='pair' else models[n]['ci95'][key] for n in names]
        err=np.array([[v-ci[0],ci[1]-v] for v,ci in zip(vals,cis)]).T
        ax.bar(np.arange(5)+(offset-.5)*.36,vals,width=.35,color=color,label=regime,yerr=np.maximum(err,0),capsize=2)
        ax.set_xticks(range(5),labels,rotation=23,ha='right');ax.set_title(key);ax.grid(axis='y',alpha=.2)
axes[0].legend();fig.suptitle('Frozen FINAL TEST: seed-clustered 95% CI');fig.tight_layout()
fig.savefig(out/'baselines.png',dpi=160);fig.savefig(out/'baselines.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for name in ['B2','B4']:
    records=selection['finite']['models'][name]['replicates']
    curves=[json.loads((run/'candidates'/r['candidate']/'curve.json').read_text()) for r in records]
    vb=np.array([[r['validation_brier'] for r in curve] for curve in curves])
    axes[0].plot(range(1,vb.shape[1]+1),vb.mean(0),label=name)
    if name=='B4':
        for key in ['loss_future_tactile','loss_future_q','loss_future_dq']:
            vals=np.array([[r[key] for r in curve] for curve in curves]);axes[1].plot(range(1,vals.shape[1]+1),vals.mean(0),label=key)
axes[0].set_title('Validation Brier / fixed training budget');axes[1].set_title('Predictive loss components (train)')
for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'training.png',dpi=160);plt.close(fig)
