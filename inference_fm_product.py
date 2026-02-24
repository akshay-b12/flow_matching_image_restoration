# ============================================================
# Inference: integrate learned ODE on Z x M (Euler / Heun)
# ============================================================

@torch.no_grad()
def integrate_product_ode(
    *,
    vf_model: ProductVFModel,
    deg_enc: nn.Module,
    ops: GeometryOps,
    vae,
    scaling_factor: float,
    degraded: torch.Tensor,
    steps: int = 30,
    method: Literal["euler", "heun"] = "heun",
):
    """
    Restores degraded -> clean by integrating t:0->1
    state: (z_t, m_t), start at (z0, m1), end should approach (z1, m0)

    NOTE: We do not explicitly force endpoint m(1)=m0 during inference; the learned vf should do it.
          We still use m0 as conditioning anchor by including it implicitly through dynamics learned.
    """
    device = degraded.device
    B = degraded.size(0)

    z = vae.encode(degraded).latent_dist.sample() * scaling_factor  # z0
    m, e = deg_enc(z)  # start manifold state m1

    # fixed time grid
    t0, t1 = 0.0, 1.0
    ts = torch.linspace(t0, t1, steps + 1, device=device)

    def eval_v(z_state, m_state, t_scalar):
        # conditioning uses current m_t and fixed e
        cond = torch.cat([m_state, e], dim=-1)
        v_z, v_m_raw = vf_model(z_state, t_scalar, cond_vec=cond)
        v_m = ops.proj_tangent(m_state, v_m_raw)
        return v_z, v_m

    for i in range(steps):
        t_a = ts[i].expand(B)
        t_b = ts[i + 1].expand(B)
        dt = (ts[i + 1] - ts[i]).item()

        v_z_a, v_m_a = eval_v(z, m, t_a)

        if method == "euler":
            z = z + dt * v_z_a
            m = ops.proj_manifold(m + dt * v_m_a) if ops.cfg.kind != "euclid" else (m + dt * v_m_a)
        else:
            # Heun (RK2)
            z_e = z + dt * v_z_a
            m_e = m + dt * v_m_a
            m_e = ops.proj_manifold(m_e) if ops.cfg.kind != "euclid" else m_e

            v_z_b, v_m_b = eval_v(z_e, m_e, t_b)

            z = z + 0.5 * dt * (v_z_a + v_z_b)
            m = m + 0.5 * dt * (v_m_a + v_m_b)
            m = ops.proj_manifold(m) if ops.cfg.kind != "euclid" else m

    # decode final z
    z_out = z / scaling_factor
    out = vae.decode(z_out).sample
    return out, {"m_final": m, "e": e}