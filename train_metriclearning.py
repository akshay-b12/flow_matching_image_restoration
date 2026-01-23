import torch.nn.functional as F

def train_step_metric(x_deg, x_clean, vae, v_theta, m_phi, opt, lam_norm=1.0, lam_tv=1e-4):
    B = x_deg.shape[0]
    device = x_deg.device

    with torch.no_grad():
        z0 = encode_latents(vae, x_deg, deterministic=False)    # [B,4,64,64]
        z1 = encode_latents(vae, x_clean, deterministic=False)  # [B,4,64,64]

    t = sample_t(B, device)                     # [B]
    tv = t[:, None, None, None]
    zt = (1 - tv) * z0 + tv * z1
    ut = (z1 - z0)

    uhat = v_theta(zt, t, cond=z0)

    # metric weights
    w = m_phi(zt, t, z0)                         # [B,4,64,64], positive
    err2 = (uhat - ut) ** 2
    loss_metric = (w * err2).mean()

    loss_norm = (w.mean() - 1.0) ** 2
    loss_tv = total_variation(w)

    loss = loss_metric + lam_norm * loss_norm + lam_tv * loss_tv

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(v_theta.parameters()) + list(m_phi.parameters()), 1.0)
    opt.step()

    return {
        "loss": float(loss.item()),
        "loss_metric": float(loss_metric.item()),
        "loss_norm": float(loss_norm.item()),
        "loss_tv": float(loss_tv.item()),
        "w_mean": float(w.mean().item()),
        "w_min": float(w.min().item()),
        "w_max": float(w.max().item()),
    }
