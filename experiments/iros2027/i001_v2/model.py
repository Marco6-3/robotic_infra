"""Fixed-budget Nyström RBF regression. Select hyperparameters on validation only."""
import time
import numpy as np
BASELINES=['B0','B1','B2','B3','B4','N1','N2','N3']


def distances(a,b):
    return np.maximum(((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T)/a.shape[1],0.)


def train(x,y,xv,yv,c):
    begin=time.perf_counter();cfg=c['models'];mean=x.mean(0);scale=np.maximum(x.std(0),1e-12)
    x=(x-mean)/scale;xv=(xv-mean)/scale
    rng=np.random.default_rng(cfg['random_seed']);indices=rng.choice(len(x),min(cfg['landmarks'],len(x)),replace=False)
    centers=x[indices];dt=distances(x,centers);dv=distances(xv,centers);dc=distances(centers,centers)
    target_mean=float(y.mean());best=None
    for length in cfg['lengths']:
        kc=np.exp(-dc/(2*length**2));eig,vec=np.linalg.eigh(kc)
        whitening=(vec/np.sqrt(np.maximum(eig,1e-6)))@vec.T
        phi=np.exp(-dt/(2*length**2))@whitening;pv=np.exp(-dv/(2*length**2))@whitening
        gram=phi.T@phi;target=phi.T@(y-target_mean)
        for penalty in cfg['penalties']:
            coef=np.linalg.solve(gram+penalty*np.eye(len(centers)),target)
            pred=np.clip(target_mean+pv@coef,0,1);loss=float(np.mean((pred-yv)**2))
            if best is None or loss<best['validation_brier']:
                best=dict(mean=mean,scale=scale,centers=centers,whitening=whitening,coef=coef,target_mean=target_mean,
                    length=length,penalty=penalty,validation_brier=loss,validation_predictions=pred)
    best['fit_seconds']=time.perf_counter()-begin;return best


def predict(m,x):
    out=[];begin=time.perf_counter()
    for i in range(0,len(x),256):
        z=(x[i:i+256]-m['mean'])/m['scale']
        phi=np.exp(-distances(z,m['centers'])/(2*float(m['length'])**2))@m['whitening']
        out.append(np.clip(float(m['target_mean'])+phi@m['coef'],0,1))
    return np.concatenate(out),time.perf_counter()-begin
