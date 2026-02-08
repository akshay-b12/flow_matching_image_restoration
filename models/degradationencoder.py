import torch
import torch.nn as nn
from utils.manifold_utils import normalize

class DegradationEncoder(nn.Module):
    """
    Predicts m1 = (s1, e1) from degraded latent z_src.
      s1 in S^{d_s-1}, e1 in R^{d_e}
    """
    def __init__(self, z_channels=4, d_s=8, d_e=1, hidden=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(z_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head_s = nn.Linear(hidden, d_s)
        self.head_e = nn.Linear(hidden, d_e)

        # Bias sphere output towards identity at init to avoid early collapse/instability.
        nn.init.zeros_(self.head_s.weight)
        nn.init.zeros_(self.head_s.bias)
        with torch.no_grad():
            self.head_s.bias[0] = 1.0

    def forward(self, z):
        # z: [B,C,H,W]
        h = self.pool(z).flatten(1)  # [B,C]
        h = self.mlp(h)              # [B,hidden]
        s_raw = self.head_s(h)       # [B,d_s]
        e = self.head_e(h)           # [B,d_e]
        s = normalize(s_raw)
        return s, e
