import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from math import sqrt 
from models import register
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = out * self.weight + self.bias
        
        return out
# class galerkin_sampled(nn.Module):
#     def __init__(self, core):
#         super().__init__()
#         self.qkv_base = core.qkv_proj.weight.data
#         self.qkv_bias = core.qkv_proj.bias.data
#         self.klnbase = core.kln.weight.data
#         self.klnbias = core.kln.bias.data
#         self.vlnbase = core.vln.weight.data
#         self.vlnbias = core.vln.bias.data
#     def sample(self,x,grid):
#         grid = torch.stack(torch.meshgrid(grid[::-1], indexing="ij"), dim=-1)
#         grid = grid.unsqueeze(0)
#         return F.grid_sample(x,grid,mode="bicubic",padding_mode="border",align_corners=True)
#     def forward(self, x):
#         B, C, H, W = x.shape
#         kl = random.randint(4,16)
#         kr = random.randint(4,16)
#         out_size = 16*kl+kl*kr+kr*16

#         qkv = F.conv2d(x,F.interpolate(self.qkv_base,(out_size,self.qkv_base[1],self.qkv_base[2],self.qkv_base[3]),mode="bicubic"),\
#                        F.interpolate(self.qkv_bias.unsqueeze().unsqueeze().unsqueeze(),(out_size,1,1,1),mode="bicubic")).permute(0,2,3,1).reshape(B,H*W,-1)

#         q = qkv[:,:,:kl*16].reshape(B,H*W,16,-1)
#         k = qkv[:,:,kl*16:kl*kr].reshape(B,H*W,kl,kr)
#         v = qkv[:,:,kl*kr:].reshape(B,H*W,16,-1)

        
class galerkin_core(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()
        headc = midc // heads
        self.cp = False
        self.midc = midc
        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.kln = LayerNorm((heads, headc))
        self.vln = LayerNorm((heads, headc))
    def add_core_proj(self):
        self.cp = True
        self.core_proj = nn.Conv2d(self.midc,self.midc,1)
    def forward(self, x):
        B, C, H, W = x.shape
        # tmp = int(sqrt(C))
        # max = 1
        # for i in range(tmp):
        #     if C % (i+1) == 0:
        #         max = i+1
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, 16, -1)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, -1)

        ret = v.permute(0, 3, 1, 2)
        if self.cp:
            ret = self.core_proj(ret)
        return ret