"""Render inspectable static figures from already evaluated artifacts."""
import csv,json,sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    run=Path(sys.argv[1]);out=run/'figures';out.mkdir(exist_ok=True)
    m=json.loads((run/'metrics.json').read_text());boot=json.loads((run/'bootstrap.json').read_text())
    c=yaml.safe_load((run/'config.yaml').read_text());o=c['observation'];anchor=c['anchor_tick']
    resolution=np.r_[np.full(32,o['tactile_quantum_m']),np.full(7,o['arm_q_quantum_rad']),np.full(2,o['finger_q_quantum_m']),np.full(7,o['arm_dq_quantum_rad_s']),np.full(2,o['finger_dq_quantum_m_s']),np.full(7,o['action_arm_quantum_rad']),o['action_width_quantum_m']]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['B0','B1','B2','B3','B4','N1','N2','N3'];fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for regime,color in [('exact','#c35635'),('finite','#287d8e')]:
        ax[0].plot(names,[m[regime]['models'][n]['brier'] for n in names],'o-',label=regime,color=color)
        values=[m[regime]['models'][n]['matched_pair']['mean'] for n in names]
        ax[1].plot(names,[np.nan if v is None else v for v in values],'o-',label=regime,color=color)
    ax[0].set(title='Frozen unseen seeds',ylabel='Brier loss (lower is better)');ax[1].set(title='Opposite-outcome matched pairs',ylabel='Pair ranking accuracy',ylim=(0,1));ax[1].axhline(.5,color='gray',ls=':');ax[0].legend();ax[1].legend()
    fig.savefig(out/'baselines.png',dpi=180);fig.savefig(out/'baselines.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained');ax.plot([0,1],[0,1],':',color='gray')
    for name in ['B1','B4']:
        bins=m['finite']['models'][name]['calibration'];ax.plot([b['prediction'] for b in bins],[b['frequency'] for b in bins],'o-',label=name)
    ax.set(xlabel='Predicted slip probability',ylabel='Observed frequency',title='Finite-resolution calibration');ax.legend();fig.savefig(out/'calibration.png',dpi=180);plt.close(fig)
    pairs=list(csv.DictReader((run/'pairs.csv').open()));valid=[p for p in pairs if p['regime']=='finite' and p['opposite']=='True']
    selected=[]
    for subset in [[p for p in valid if p['fixed']=='True'],valid]:
        for pair in sorted(subset,key=lambda p:float(p['rms'])):
            if pair not in selected:selected.append(pair)
            if len(selected)>=3:break
        if len(selected)>=3:break
    for index,pair in enumerate(selected):
        fig,axs=plt.subplots(3,2,figsize=(12,9),layout='constrained');current=[]
        for key,color in [('a','#c35635'),('b','#287d8e')]:
            with np.load(run/'episodes/test'/pair[key]) as z:
                meta=json.loads(str(z['metadata']));t=z['ticks_ns']/1e6-anchor;keep=(t>=-300)&(t<=0);obs=z['finite'][keep];ts=t[keep];truth=z['labels'];label=f"mu={meta['friction']:.3f}, slip={meta['label']}"
                axs[0,0].plot(ts,obs[:,:32].sum(1)*1e6,color=color,label=label)
                axs[0,1].plot(ts,obs[:,-1]*1e3,color=color)
                axs[1,0].plot(ts,obs[:,39]*1e6,color=color,label='left '+label)
                axs[1,1].plot(ts,obs[:,48]*1e3,color=color)
                current.append(obs[-1].copy())
                axs[2,1].plot(truth[:,0]-anchor,truth[:,1],color=color)
        axs[0,0].set(title='Tactile geometry (sum)',ylabel='Depth sum (um)',xlabel='Time to anchor (ms)');axs[0,0].legend(fontsize=8)
        axs[0,1].set(title='Actual commanded probing action',ylabel='Width target (mm)',xlabel='Time to anchor (ms)')
        axs[1,0].set(title='Current and historical finger q',ylabel='Left finger q (um)',xlabel='Time to anchor (ms)')
        axs[1,1].set(title='Current and historical finger dq',ylabel='Left finger dq (mm/s)',xlabel='Time to anchor (ms)')
        axs[2,0].plot((current[0]-current[1])/resolution,'.',color='#287d8e');axs[2,0].axhline(1,color='gray',ls=':');axs[2,0].axhline(-1,color='gray',ls=':')
        axs[2,0].set(title=f"Current difference; normalized RMS={float(pair['rms']):.3f}",xlabel='Sensor feature index',ylabel='Difference / resolution',ylim=(-1.2,1.2))
        axs[2,1].set(title='Privileged contact slip velocity',xlabel='Time to anchor (ms)',ylabel='m/s',xlim=(-100,100));axs[2,1].axvline(0,color='gray',ls='--');axs[2,1].axhline(.01,color='gray',ls=':')
        fig.suptitle(f"Seed {pair['seed']}; fixed friction={pair['fixed']}; same future commands and load")
        fig.savefig(out/f'pair-{index}.png',dpi=180);fig.savefig(out/f'pair-{index}.pdf');plt.close(fig)
    (out/'selected_pairs.json').write_text(json.dumps(selected,indent=2)+'\n')

if __name__=='__main__':main()
