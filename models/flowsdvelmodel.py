import torch
from diffusers import UNet2DConditionModel


class FlowSDVelModel(nn.Module):
    """
    Iteration-1: SD UNet predicts v_z only.
    Conditioning tokens = [null text tokens, degradation tokens].
    """
    def __init__(self, unet: UNet2DConditionModel, null_text: NullTextTokens, deg_cond: DegradationConditioner):
        super().__init__()
        self.unet = unet
        self.null_text = null_text
        self.deg_cond = deg_cond

    def forward(self, z_t, t, m_t, e):
        """
        z_t: [B,4,H,W]
        t:   [B] in [0,1] float
        m_t: [B, m_dim]
        e:   [B, 1]
        """
        B = z_t.size(0)

        null_tokens = self.null_text(B)       # [B,77,768]
        deg_tokens  = self.deg_cond(m_t, e)   # [B,T,768]
        tokens = torch.cat([null_tokens, deg_tokens], dim=1)  # [B,77+T,768]

        # SD-style continuous timestep scaling
        t_scaled = t * 999.0  # float is OK

        out = self.unet(
            sample=z_t,
            timestep=t_scaled,
            encoder_hidden_states=tokens,
            return_dict=True,
        )
        return out.sample  # v_z_pred: [B,4,H,W]