import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaGN(nn.Module):
    """
    Adaptive GroupNorm: y = GN(x) * (1 + scale) + shift
    scale, shift from emb.
    """
    def __init__(self, num_groups, num_channels, emb_dim):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=1e-6, affine=False)
        self.to_ss = nn.Linear(emb_dim, 2 * num_channels)

        # identity at init
        nn.init.zeros_(self.to_ss.weight)
        nn.init.zeros_(self.to_ss.bias)

    def forward(self, x, emb):
        h = self.gn(x)
        ss = self.to_ss(emb)
        scale, shift = ss.chunk(2, dim=1)
        return h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlockAdaGN(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, num_groups=8):
        super().__init__()
        self.act = nn.SiLU()

        self.adagn1 = AdaGN(num_groups, in_ch, emb_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.adagn2 = AdaGN(num_groups, out_ch, emb_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # stable residual init
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.adagn1(x, emb)
        h = self.act(h)
        h = self.conv1(h)

        h = self.adagn2(h, emb)
        h = self.act(h)
        h = self.conv2(h)

        return self.skip(x) + h


class TinyUNetCondOnManifold(nn.Module):
    """
    TinyUNet with AdaGN conditioning, using JOINT fusion of time+cond:

        emb = joint_mlp([t_sin, cond])      # strongest / most flexible

    This matches your earlier "collective MLP" idea but in a clean way.
    """
    def __init__(
        self,
        in_channels=4,
        base_channels=64,
        time_emb_dim=128,
        cond_dim=0,
        num_groups=8,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim

        # Joint fusion MLP:
        # input = t_sin (time_emb_dim) + cond (cond_dim)
        joint_in = time_emb_dim + (cond_dim if cond_dim > 0 else 0)
        self.joint_mlp = nn.Sequential(
            nn.Linear(joint_in, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        emb_dim = time_emb_dim

        self.down1 = ResBlockAdaGN(in_channels, base_channels, emb_dim, num_groups=num_groups)
        self.down2 = ResBlockAdaGN(base_channels, base_channels * 2, emb_dim, num_groups=num_groups)
        self.mid  = ResBlockAdaGN(base_channels * 2, base_channels * 2, emb_dim, num_groups=num_groups)
        self.up1  = ResBlockAdaGN(base_channels * 2 + base_channels, base_channels, emb_dim, num_groups=num_groups)

        self.out  = nn.Conv2d(base_channels, in_channels, 3, padding=1)

        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # stable output init
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond=None, return_feats: bool = False):
        t_sin = timestep_embedding(t, self.time_emb_dim)

        if self.cond_dim > 0:
            if cond is None:
                cond = torch.zeros(x.size(0), self.cond_dim, device=x.device, dtype=t_sin.dtype)
            emb_in = torch.cat([t_sin, cond], dim=1)
        else:
            emb_in = t_sin

        emb = self.joint_mlp(emb_in)

        h1 = self.down1(x, emb)
        h2 = self.down2(self.pool(h1), emb)
        h3 = self.mid(h2, emb)

        h4 = self.upsample(h3)
        h4 = torch.cat([h4, h1], dim=1)
        h4 = self.up1(h4, emb)

        out = self.out(h4)

        if return_feats:
            return out, {"mid": h3}
        return out