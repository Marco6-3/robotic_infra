"""Small matched GRUs; only the auxiliary objective separates B2 and B4."""
import math
import torch
from torch import nn

NAMES = {'B0': 'Current-MLP', 'B1': 'Stack-MLP', 'B2': 'GRU-History',
         'B3': 'Temporal-Transformer', 'B4': 'ContactBelief'}

class Model(nn.Module):
    def __init__(self, method, config):
        super().__init__(); self.method = method; self.config = config
        e, h = config['encoder_dim'], config['hidden_dim']
        self.encoder = nn.Sequential(nn.Linear(58, e), nn.ReLU())
        if method in ['B2', 'B4']:
            self.temporal = nn.GRU(e, h, batch_first=True)
        elif method == 'B3':
            layer = nn.TransformerEncoderLayer(e, config['transformer_heads'], dim_feedforward=2*e,
                                               dropout=0., activation='gelu', batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, config['transformer_layers'], enable_nested_tensor=False)
            position = torch.arange(61)[:, None]
            freq = torch.exp(torch.arange(0, e, 2)*(-math.log(10000.)/e))
            pe = torch.zeros(61, e); pe[:, 0::2] = torch.sin(position*freq); pe[:, 1::2] = torch.cos(position*freq)
            self.register_buffer('positions', pe)
        elif method == 'B1':
            # All input channels, all 61 observations; no bottleneck via encoder.
            self.encoder = nn.Identity()
            self.temporal = nn.Sequential(nn.Linear(61*58, config['stack_hidden']), nn.ReLU(),
                                          nn.Linear(config['stack_hidden'], h), nn.ReLU())
        elif method == 'B0':
            self.temporal = nn.Sequential(nn.Linear(e, h), nn.ReLU())
        else:
            raise ValueError(method)
        self.outcome = nn.Sequential(nn.Linear(h, h//2), nn.ReLU(), nn.Linear(h//2, 1))
        # Same allocated parameters/initialization for B2 and B4. B2 head is dormant.
        if method in ['B2', 'B4']:
            self.dynamics = nn.Sequential(nn.Linear(h+16+1, 32), nn.ReLU(), nn.Linear(32, 50))

    def encode(self, x):
        if self.method == 'B0':
            e = self.encoder(x[:, -1:]); return self.temporal(e), e
        if self.method == 'B1':
            return self.temporal(x.flatten(1))[:, None], x[:, -1:]
        e = self.encoder(x)
        if self.method in ['B2', 'B4']:
            z, _ = self.temporal(e)
        else:
            length = x.shape[1]
            mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
            z = self.temporal(e+self.positions[:length], mask=mask)
        return z, e

    def forward(self, x):
        z, _ = self.encode(x)
        return self.outcome(z[:, -1]).squeeze(-1)

    def predictive(self, z, controls):
        horizons = torch.tensor(self.config['horizons_ms'], device=z.device, dtype=z.dtype)/100.
        h = horizons[None, None, :, None].expand(z.shape[0], z.shape[1], -1, -1)
        state = z[:, :, None].expand(-1, -1, len(horizons), -1)
        return self.dynamics(torch.cat([state, controls, h], -1))

def future_loss(predicted, target):
    # Equal weight per modality, avoiding 32 tactile cells dominating 9 q / 9 dq.
    return sum((predicted[..., sl]-target[..., sl]).square().mean()
               for sl in [slice(0, 32), slice(32, 41), slice(41, 50)])/3

def parameter_report(model):
    total = sum(p.numel() for p in model.parameters())
    auxiliary = sum(p.numel() for p in model.dynamics.parameters()) if hasattr(model, 'dynamics') else 0
    return dict(allocated_parameters=total, outcome_inference_parameters=total-auxiliary,
                parameters_receiving_training_gradients=total-(auxiliary if model.method == 'B2' else 0),
                auxiliary_head_parameters=auxiliary,
                channels=58, history_ms=0 if model.method == 'B0' else 300, samples=1 if model.method == 'B0' else 61)
