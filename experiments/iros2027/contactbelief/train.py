"""Fixed training budget, validation-only selection, resumable optimizer state."""
from __future__ import annotations
import argparse, copy, itertools, time
from pathlib import Path
import numpy as np
import torch
import yaml
from .dataset import load_split, fit_normalizer, transform
from .models import Model, parameter_report
from .utils import atomic, read, progress, sha

def setup(config):
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(config['device'])

def save_torch(path, obj):
    temp = path.with_suffix('.tmp'); torch.save(obj, temp); temp.replace(path)

def tensor_data(data, device):
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}

def infer(model, data, config, representations=False):
    model.eval(); out = []; latent = []; embedding = []
    with torch.no_grad():
        for x in data['x'].split(config['eval_batch_size']):
            z, e = model.encode(x)
            out.append(torch.sigmoid(model.outcome(z[:, -1]).squeeze(-1)).cpu().numpy())
            if representations:
                latent.append(z[:, -1].cpu().numpy()); embedding.append(e[:, -1].cpu().numpy())
    if representations:
        return np.concatenate(out), np.concatenate(latent), np.concatenate(embedding)
    return np.concatenate(out)

def train_candidate(run, config, regime, method, lr, weight, seed, tr, va):
    key = f'{regime}-{method}-lr{lr:g}-lam{weight:g}-s{seed}'
    directory = run/'candidates'/key; directory.mkdir(parents=True, exist_ok=True)
    if (directory/'complete.json').exists():
        return read(directory/'complete.json')
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = Model(method, config).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
    start = 0; best = float('inf'); logs = []; best_epoch = None
    if (directory/'latest.pt').exists():
        state = torch.load(directory/'latest.pt', weights_only=False, map_location=config['device'])
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start = state['epoch']; best = state['best']; logs = state['logs']; best_epoch = state['best_epoch']
    begin = time.perf_counter()
    for epoch in range(start, config['epochs']):
        model.train()
        # Independent epoch permutation makes uninterrupted/resumed training identical.
        order = np.random.default_rng([seed, epoch]).permutation(len(tr['y']))
        losses = np.zeros(4); count = 0
        for offset in range(0, len(order), config['train_batch_size']):
            ix = order[offset:offset+config['train_batch_size']]
            optimizer.zero_grad(set_to_none=True)
            z, _ = model.encode(tr['x'][ix])
            logits = model.outcome(z[:, -1]).squeeze(-1)
            outcome = torch.nn.functional.binary_cross_entropy_with_logits(logits, tr['y'][ix])
            future_parts = [outcome.new_zeros(()) for _ in range(3)]
            if weight > 0:
                prediction = model.predictive(z, tr['controls'][ix])
                future_parts = [(prediction[..., sl]-tr['future'][ix][..., sl]).square().mean()
                                for sl in [slice(0, 32), slice(32, 41), slice(41, 50)]]
            loss = outcome+weight*sum(future_parts)/3
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss '+key)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip']); optimizer.step()
            losses += np.array([outcome.item()]+[v.item() for v in future_parts])*len(ix); count += len(ix)
        p = infer(model, va, config); vb = float(np.mean((p-va['y'].cpu().numpy())**2))
        record = dict(epoch=epoch+1, validation_brier=vb,
                      **dict(zip(['loss_outcome', 'loss_future_tactile', 'loss_future_q', 'loss_future_dq'], (losses/count).tolist())))
        logs.append(record)
        if vb < best:
            best = vb; best_epoch = epoch+1
            save_torch(directory/'best.pt', dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                               method=method, lr=lr, lambda_future=weight, seed=seed, best_epoch=best_epoch))
            np.save(directory/'validation_predictions.npy', p)
        # Save every epoch, including the first: interruption loses at most one epoch.
        save_torch(directory/'latest.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                              epoch=epoch+1, best=best, best_epoch=best_epoch, logs=logs))
        atomic(directory/'curve.json', logs)
        if (epoch+1)%10 == 0:
            progress(run, 'train', candidate=key, **record)
    result = dict(candidate=key, method=method, regime=regime, lr=lr, lambda_future=weight, seed=seed,
                  validation_brier=best, best_epoch=best_epoch, trained_epochs=config['epochs'],
                  optimizer_steps=config['epochs']*((len(tr['y'])+config['train_batch_size']-1)//config['train_batch_size']),
                  parameters=parameter_report(model), seconds_this_attempt=time.perf_counter()-begin,
                  checkpoint_sha256=sha(directory/'best.pt'))
    atomic(directory/'complete.json', result)
    return result

def fit_all(run, config):
    if (run/'FROZEN.json').exists():
        raise RuntimeError('Training forbidden after FROZEN.json')
    device = setup(config); selections = {}; all_records = []
    (run/'normalizers').mkdir(exist_ok=True)
    for regime in config['regimes']:
        raw_train = load_split(run, config, regime, 'train'); raw_val = load_split(run, config, regime, 'validation')
        norm_path = run/'normalizers'/f'{regime}.npz'
        if norm_path.exists():
            with np.load(norm_path) as f: norm = dict(f)
        else:
            norm = fit_normalizer(raw_train, config); np.savez(norm_path, **norm)
        tr = tensor_data(transform(raw_train, norm), device); va = tensor_data(transform(raw_val, norm), device)
        chosen = {}
        for method in config['methods']:
            weights = config['lambda_future'] if method == 'B4' else [0.]
            candidates = []
            for lr, weight in itertools.product(config['learning_rates'], weights):
                replicates = [train_candidate(run, config, regime, method, lr, weight, seed, tr, va)
                              for seed in config['training_seeds']]
                all_records.extend(replicates)
                ensemble = np.mean([np.load(run/'candidates'/r['candidate']/'validation_predictions.npy') for r in replicates], axis=0)
                candidates.append(dict(lr=lr, lambda_future=weight, replicates=replicates,
                                       validation_brier=float(np.mean((ensemble-raw_val['y'])**2))))
            chosen[method] = min(candidates, key=lambda r:r['validation_brier'])
            atomic(run/'selection_in_progress.json', dict(regime=regime, selected=chosen))
        # Strict same-learning-rate lambda=0 comparator as well as best validation GRU.
        matching = [r for r in all_records if r['regime']==regime and r['method']=='B2' and r['lr']==chosen['B4']['lr']]
        p = np.mean([np.load(run/'candidates'/r['candidate']/'validation_predictions.npy') for r in matching], axis=0)
        chosen['B2matched'] = dict(lr=chosen['B4']['lr'], lambda_future=0., replicates=matching,
                                   validation_brier=float(np.mean((p-raw_val['y'])**2)))
        selections[regime] = dict(models=chosen,
                                 strongest_generic=min(['B1','B2','B3'], key=lambda k:chosen[k]['validation_brier']),
                                 strongest_recurrent_attention=min(['B2','B3'], key=lambda k:chosen[k]['validation_brier']))
        atomic(run/'selection.json', selections)
        del tr, va
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    atomic(run/'training_summary.json', all_records)
    return selections

def load_model(run, record, config):
    state = torch.load(run/'candidates'/record['candidate']/'best.pt', map_location='cpu', weights_only=False)
    model = Model(record['method'], config)
    model.load_state_dict(state['model']); return model.to(config['device']).eval()

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path); a = p.parse_args()
    fit_all(a.run, yaml.safe_load((a.run/'config.yaml').read_text()))
