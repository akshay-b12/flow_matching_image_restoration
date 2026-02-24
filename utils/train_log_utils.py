# ============================================================
# What to log (tangent-bundle alignment checks)
# ============================================================

@torch.no_grad()
def tangent_alignment_metrics(ops: GeometryOps, m_t: torch.Tensor, v_m: torch.Tensor):
    """
    Returns a scalar 'tangent violation' that should be ~0 if v_m \in T_{m_t}M.
    """
    if ops.cfg.kind == "euclid":
        return {"tangent_violation": 0.0}

    if ops.cfg.kind == "sphere":
        # <m, v> should be ~0
        viol = (m_t * v_m).sum(dim=-1).abs().mean().item()
        # manifold constraint ||m||=1
        norm_err = (m_t.norm(dim=-1) - 1.0).abs().mean().item()
        return {"tangent_violation": viol, "manifold_residual": norm_err}

    if ops.cfg.kind == "hyperboloid":
        viol = ops.hyp_lorentz_inner(m_t, v_m).abs().mean().item()
        # residual |<h,h>_L + 1/c|
        res = (ops.hyp_lorentz_inner(m_t, m_t) + (1.0 / ops.cfg.c)).abs().mean().item()
        return {"tangent_violation": viol, "manifold_residual": res}

    raise ValueError(ops.cfg.kind)