import torch
import torch.nn as nn

# ============================================================
# Vector-field model on product space: outputs (v_z, v_m_raw)
#    - v_m_raw is ambient; we project it to tangent at m_t
#    - conditioning: concat([m_t, e_t]) -> embed -> add to time embedding
# ============================================================

class ProductVFModel(nn.Module):
    """
    Minor stability tweak:
    - initialize m_head last layer to near-zero so early v_m is small;
      prevents early manifold "kicks" from dominating before z-field stabilizes.
    """
    """
    Patched ProductVFModel that uses TinyUNetCondJoint as trunk.

    - Conditioning is handled inside the trunk via JOINT fusion MLP on [t_sin, cond].
    - We use cond = [m_t, e] as requested.
    - Outputs:
        v_z:   [B, z_channels, H, W]
        v_m_raw: [B, m_out_dim]  (ambient manifold velocity; project to tangent outside)
    """
    def __init__(
        self,
        *,
        trunk,                 # instance of TinyUNetCondJoint
        z_channels=4,
        base_channels=64,
        m_out_dim=8,
    ):
        super().__init__()
        self.trunk = trunk

        # manifold head from a mid feature map (channels = base_channels*2)
        self.m_pool = nn.AdaptiveAvgPool2d(1)
        self.m_head = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, m_out_dim),
        )

        # Stability: start with small manifold velocities
        nn.init.zeros_(self.m_head[-1].weight)
        nn.init.zeros_(self.m_head[-1].bias)

    def forward(self, z_t, t_scalar, cond_vec):
        """
        z_t:      [B, C, H, W]
        t_scalar: [B]  (float in [0,1])
        cond_vec: [B, cond_dim]  where cond_dim = dim(m_t) + dim(e)

        Returns:
          v_z:     [B, C, H, W]
          v_m_raw: [B, m_out_dim]
        """
        # --- trunk forward with conditioning ---
        # We need mid features for v_m head, so we slightly augment the trunk call:
        # If your TinyUNetCondJoint currently returns only out, modify it to optionally return mid features.
        #
        # Here we assume trunk.forward returns (v_z, feats) when return_feats=True.
        v_z, feats = self.trunk(z_t, t_scalar, cond=cond_vec, return_feats=True)

        # use mid feature map (the one with base*2 channels)
        h_mid = feats["mid"]  # [B, base*2, H/2, W/2] in this tiny UNet
        pooled = self.m_pool(h_mid).flatten(1)
        v_m_raw = self.m_head(pooled)
        return v_z, v_m_raw