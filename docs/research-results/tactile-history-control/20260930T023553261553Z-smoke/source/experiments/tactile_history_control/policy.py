"""Exactly identical parameter tensors across M0, M1 and M2."""
import math
import torch
from torch import nn

class Policy(nn.Module):
    def __init__(self,c,method):
        super().__init__();self.method=method
        m=c['model'];d=m['hidden'];h=c['history_ms']//c['sample_ms']+1
        self.project=nn.Linear(42,d)
        layer=nn.TransformerEncoderLayer(d,m['heads'],m['ff'],m['dropout'],batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,m['layers'],enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
        position=torch.arange(h)[:,None];frequency=torch.exp(torch.arange(0,d,2)*(-math.log(10000.)/d))
        pe=torch.zeros(h,d);pe[:,0::2]=torch.sin(position*frequency);pe[:,1::2]=torch.cos(position*frequency)
        self.register_buffer('position',pe)
        self.register_buffer('mask',torch.triu(torch.ones(h,h,dtype=torch.bool),1))

    def content(self,x):
        if self.method=='M0':x=x[:,-1:,:].expand(-1,x.shape[1],-1)
        if self.method in ('M0','M1'):x=torch.cat([x[...,:-1],torch.zeros_like(x[...,-1:])],-1)
        return x

    def encode(self,x):
        x=self.content(x);n=x.shape[1]
        return self.transformer(self.project(x)+self.position[:n],mask=self.mask[:n,:n])

    def forward(self,x):return self.head(self.encode(x)[:,-1]).squeeze(-1)

def normalize(x,mean,scale):return ((x-mean)/scale).clip(-8,8)
