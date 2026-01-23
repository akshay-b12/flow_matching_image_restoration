import torch
import torch.nn as nn

class MetricDiagHead(nn.Module):
    """
    Predicts positive per-pixel weights w in R^{C x H x W} for a diagonal metric.
    Input: concat([zt, z0]) and t embedding (optional).
    Output: w >= eps
    """
    def __init__(self, in_ch=8, out_ch=4, hidden=32, tdim=128, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.time = nn.Sequential(
            SinusoidalTimeEmb(tdim),
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        self.tproj = nn.Linear(tdim, hidden)

        self.conv1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, out_ch, 1)
        self.act = nn.SiLU()
        self.gn1 = nn.GroupNorm(8, hidden)
        self.gn2 = nn.GroupNorm(8, hidden)

    def forward(self, zt, t, z0):
        x = torch.cat([zt, z0], dim=1)  # [B,8,H,W]
        temb = self.time(t)            # [B,tdim]
        h = self.conv1(x)
        h = self.gn1(h)
        h = h + self.tproj(temb)[:, :, None, None]
        h = self.act(h)

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h)

        raw = self.conv3(h)            # [B,4,H,W]
        w = F.softplus(raw) + self.eps
        return w
