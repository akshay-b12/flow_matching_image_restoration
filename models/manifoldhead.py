import torch
import torch.nn as nn

class ManifoldHead(nn.Module):
    def __init__(self, mid_ch, cross_dim, m_out_dim, hidden=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(mid_ch + cross_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, m_out_dim),
        )
        # stable start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, mid_feat, tokens):
        # mid_feat: [B, mid_ch, h, w]
        # tokens:   [B, seq, cross_dim]
        B = mid_feat.size(0)
        f = self.pool(mid_feat).view(B, -1)     # [B, mid_ch]
        c = tokens.mean(dim=1)                  # [B, cross_dim]
        return self.mlp(torch.cat([f, c], dim=-1))  # [B, m_out_dim]