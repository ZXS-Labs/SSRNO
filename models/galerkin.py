import torch
import torch.nn as nn

from models import register
from .galerkin_core import galerkin_core
from tempo.inplace_gelu import InplaceGelu
@register('galerkin') 
class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.core = galerkin_core(self.midc, self.heads)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)
        # self.act = InplaceGelu()
        self.act = nn.GELU()
        #self.act = nn.ReLU()
    def forward(self, x, name='0'):
        bias = x
        ret = self.core(x) + bias 
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias
    