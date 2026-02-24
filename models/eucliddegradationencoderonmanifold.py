import torch
import torch.nn as nn

class EuclidDegradationEncoderOnManifold(nn.Module):
    """
    Simple baseline: outputs m1 in R^d and severity e.
    """
    def __init__(self, z_channels=4, d=8, d_e=1, hidden=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(z_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head_m = nn.Linear(hidden, d)
        self.head_e = nn.Linear(hidden, d_e)
        nn.init.zeros_(self.head_m.weight); nn.init.zeros_(self.head_m.bias)

    def forward(self, z):
        h = self.pool(z).flatten(1)
        h = self.mlp(h)
        m = self.head_m(h)
        e = self.head_e(h)
        return m, e