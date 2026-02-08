import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from arch_utils import CondMLP, ResBlockCond
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class TinyUNetCond(nn.Module):
    def __init__(self, in_channels=4, base_channels=64, time_emb_dim=128, d_s=8, d_e=1, cond_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.cond_mlp = CondMLP(time_emb_dim=time_emb_dim, d_s=d_s, d_e=d_e, cond_dim=cond_dim)

        self.down1 = ResBlockCond(in_channels, base_channels, cond_dim)
        self.down2 = ResBlockCond(base_channels, base_channels * 2, cond_dim)
        self.mid  = ResBlockCond(base_channels * 2, base_channels * 2, cond_dim)
        self.up1  = ResBlockCond(base_channels * 2 + base_channels, base_channels, cond_dim)

        self.out = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t, s_t, e_t):
        """
        x: [B,C,H,W]
        t: [B] in [0,1]
        s_t: [B,d_s] (unit sphere)
        e_t: [B,d_e]
        """
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)                 # [B, time_emb_dim]
        cond  = self.cond_mlp(t_emb, s_t, e_t)        # [B, cond_dim]

        h1 = self.down1(x, cond)
        h2 = self.down2(self.pool(h1), cond)
        h3 = self.mid(h2, cond)
        h4 = self.upsample(h3)
        h4 = torch.cat([h4, h1], dim=1)
        h4 = self.up1(h4, cond)
        return self.out(h4)
