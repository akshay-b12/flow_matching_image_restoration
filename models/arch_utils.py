import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(tdim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.conv1(x)
        h = self.gn1(h)
        h = h + self.time_proj(temb)[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        h = self.gn2(h)
        return self.act(h + self.skip(x))
    
class CondMLP(nn.Module):
    """
    Builds a conditioning vector c from time embedding + manifold state.
    c is used to generate AdaGN scale/shift in each ResBlock.
    """
    def __init__(self, time_emb_dim=128, d_s=8, d_e=1, cond_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim + d_s + d_e, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

    def forward(self, t_emb, s_t, e_t):
        # t_emb: [B,time_emb_dim], s_t:[B,d_s], e_t:[B,d_e]
        x = torch.cat([t_emb, s_t, e_t], dim=-1)
        return self.mlp(x)  # [B,cond_dim]
