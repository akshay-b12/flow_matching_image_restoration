import torch 
from utils.vae_utils import encode_latents, decode_latents

@torch.no_grad()
def heun_transport(z0, v_theta, steps=20):
    B = z0.shape[0]
    device = z0.device
    z = z0
    ts = torch.linspace(0, 1, steps + 1, device=device)

    for i in range(steps):
        t0 = ts[i].repeat(B)
        t1 = ts[i+1].repeat(B)
        dt = (ts[i+1] - ts[i]).item()

        k1 = v_theta(z, t0, cond=z0)
        z_pred = z + dt * k1
        k2 = v_theta(z_pred, t1, cond=z0)

        z = z + dt * 0.5 * (k1 + k2)
    return z

@torch.no_grad()
def restore(x_deg, vae, v_theta, steps=20):
    # x_deg in [-1,1]
    z0 = encode_latents(vae, x_deg, deterministic=True)
    zT = heun_transport(z0, v_theta, steps=steps)
    x_hat = decode_latents(vae, zT)
    return x_hat
