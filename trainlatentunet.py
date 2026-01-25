import torch
import torch.nn.functional as F
from torch.optim import AdamW

from utils.vae_utils import encode_latents, decode_latents
from utils.utils import sample_t
from models.tinylatentunet import TinyLatentUNet

def train_one_step(x_deg, x_clean, vae, v_theta, opt):
    B = x_deg.shape[0]
    device = x_deg.device

    # Encode endpoints (frozen VAE)
    with torch.no_grad():
        z0 = encode_latents(vae, x_deg, deterministic=False)    # [B,4,64,64]
        z1 = encode_latents(vae, x_clean, deterministic=False)  # [B,4,64,64]

    # Sample time + construct triple
    t = sample_t(B, device)                 # [B]
    tv = t[:, None, None, None]             # [B,1,1,1]
    zt = (1 - tv) * z0 + tv * z1
    ut = (z1 - z0)

    # Predict velocity (condition on z0)
    uhat = v_theta(zt, t, cond=z0)

    loss = F.mse_loss(uhat, ut)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(v_theta.parameters(), 1.0)  # helps early stability
    opt.step()

    return loss.item()

# Setup
def setup(vae, device="cuda"):
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    v_theta = TinyLatentUNet(in_ch=8, out_ch=4, base=64, tdim=256).to(device)
    opt = AdamW(v_theta.parameters(), lr=1e-4, weight_decay=1e-2)
    return v_theta, opt
