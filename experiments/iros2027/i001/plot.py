"""Standalone publication-readable figures from saved evidence (no fitting)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('analysis',type=Path);a=p.parse_args()
    s=json.loads((a.analysis/'summary.json').read_text());run=Path(s['run'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for flag,label in [('True','With probe'),('False','No probe')]:
        g=s['groups'][flag];ms=[0,20,50,100,200,250]
        axs[0].plot(ms,[g['models'][f'tactile_{m}ms']['brier'] for m in ms],'o-',label=label)
    axs[0].set(xlabel='History (ms)',ylabel='Test Brier loss (lower is better)',title='Held-out physics seeds');axs[0].legend()
    models=['tactile_0ms','tactile_250ms','mean_last_100ms','shuffle_time_250ms','mismatch_past_250ms','current_tactile_proprio','history_tactile_proprio']
    labels=['Single frame','250 ms history','100 ms mean','Time shuffled','Past mismatched','Current + q/dq','History + q/dq']
    axs[1].barh(labels,[s['groups']['True']['models'][n]['brier'] for n in models],color='#287d8e')
    axs[1].invert_yaxis();axs[1].set(xlabel='Test Brier loss',title='With probe: controls and alternatives')
    fig.savefig(a.analysis/'benchmark.png',dpi=180);fig.savefig(a.analysis/'benchmark.pdf');plt.close(fig)
    pairs=json.loads((a.analysis/'alias_pairs.json').read_text())
    pairs=[p for p in pairs if p['probe'] and p['opposite'] and p['proprio_match'] and p['force_max_n']<=.03]
    if not pairs:return
    pair=min(pairs,key=lambda p:p['force_max_n'])
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for name,color in [(pair['a'],'#c45432'),(pair['b'],'#287d8e')]:
        with np.load(run/'episodes'/name) as z:
            m=json.loads(str(z['metadata']));t=z['ticks']-2650;keep=t<=0
            label=f"friction={m['friction']:.3f}, future slip={m['label']}"
            axs[0,0].plot(t[keep],z['tactile'][keep][:,[0,3]].sum(1),label=label,color=color)
            axs[0,1].plot(t[keep],z['tactile'][keep][:,[2,5]].sum(1),color=color)
            index=np.where(t==0)[0][0]
            axs[1,0].plot(range(6),z['force_truth'][index],'o-',color=color,label=label)
            truth=z['labels'];axs[1,1].plot(truth[:,0]-2650,truth[:,1],color=color)
    axs[0,0].set(title='Past normal force (noisy observation)',ylabel='Sum over fingers (N)',xlabel='Time relative to prediction (ms)')
    axs[0,0].legend(fontsize=8)
    axs[0,1].set(title='Past shear force (noisy observation)',ylabel='Sum over fingers (N)',xlabel='Time relative to prediction (ms)')
    axs[1,0].set(title=f"Current truth force: max difference {pair['force_max_n']:.4f} N",xlabel='Force channel',ylabel='Force (N)')
    axs[1,1].set(title='Privileged contact speed (evaluation only)',xlabel='Time relative to prediction (ms)',ylabel='Speed (m/s)',xlim=(-20,50))
    axs[1,1].axvline(0,color='gray',linestyle='--');axs[1,1].axhline(.01,color='gray',linestyle=':')
    fig.suptitle(f"Same mass, gripper command and future load | held-out seed {pair['seed']}")
    fig.savefig(a.analysis/'alias_pair.png',dpi=180);fig.savefig(a.analysis/'alias_pair.pdf');plt.close(fig)
    (a.analysis/'figure_pair.json').write_text(json.dumps(pair,indent=2)+'\n')

if __name__=='__main__':main()
