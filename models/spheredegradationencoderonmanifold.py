import torch
import torch.nn as nn

# ============================================================
# Degradation encoders (produce manifold endpoint m1 + severity e)
# ============================================================
    
class SphereDegradationEncoderOnManifold(nn.Module):
    """
    Outputs m1 on unit sphere S^{d-1} and Euclidean severity e.
    """
    """
    Minor stability tweak:
    - ensure we never divide by 0 in normalization at init / early epochs.
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
        self.head_m = nn.Linear(hidden, d_s)
        self.head_e = nn.Linear(hidden, d_e)

        # stable init: bias to north pole, weights small
        nn.init.normal_(self.head_m.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head_m.bias)
        with torch.no_grad():
            self.head_m.bias[0] = 1.0

    def forward(self, z):
        h = self.pool(z).flatten(1)
        h = self.mlp(h)
        m_raw = self.head_m(h)
        m = m_raw / (m_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        e = self.head_e(h)
        return m, e