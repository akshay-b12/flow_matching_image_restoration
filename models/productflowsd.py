import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class ProductFlowSD(nn.Module):
    """
    Iteration-2: predicts (v_z, v_m_raw)
    Conditioning tokens = [null_tokens, deg_tokens(m_t,e)].
    """
    def __init__(self, unet: UNet2DConditionModel, null_text, deg_cond, m_out_dim: int):
        super().__init__()
        self.unet = unet
        self.null_text = null_text
        self.deg_cond = deg_cond

        # capture mid features
        self._mid_cache = None
        def _save_mid(module, inp, out):
            self._mid_cache = out
        self.unet.mid_block.register_forward_hook(_save_mid)

        mid_ch = self.unet.config.block_out_channels[-1]
        cross_dim = self.unet.config.cross_attention_dim
        self.m_head = ManifoldHead(mid_ch, cross_dim, m_out_dim)

    def forward(self, z_t, t, m_t, e):
        B = z_t.size(0)

        null_tokens = self.null_text(B, device=z_t.device, dtype=z_t.dtype)   # [B,77,D]
        deg_tokens  = self.deg_cond(m_t, e).to(dtype=z_t.dtype)               # [B,T,D]
        tokens = torch.cat([null_tokens, deg_tokens], dim=1)                  # [B,77+T,D]

        t_scaled = t * 999.0  # float timesteps

        out = self.unet(
            sample=z_t,
            timestep=t_scaled,
            encoder_hidden_states=tokens,
            return_dict=True,
        )
        v_z = out.sample

        mid_feat = self._mid_cache
        if mid_feat is None:
            raise RuntimeError("mid_block hook did not capture features.")
        v_m_raw = self.m_head(mid_feat, tokens)  # [B, m_out_dim]

        return v_z, v_m_raw