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

import torch
import torch.nn as nn
import math


@torch.no_grad()
def log_iteration2_stats(
    writer,
    global_step: int,
    *,
    ops,
    v_m_raw: torch.Tensor,
    v_m_pred: torch.Tensor,
    m_t: torch.Tensor,
    v_m_star: torch.Tensor,
    loss_m: torch.Tensor,
    prefix: str = "iter2",
    eps: float = 1e-8,
):
    """
    Logs the 3 high-signal diagnostics to confirm manifold branch is not collapsing:

    1) proj_ratio = ||proj_tangent(v_m_raw)|| / ||v_m_raw||
       - If this is ~1e-5, you're annihilating almost all of v_m_raw (collapse).
       - Healthy values typically: 0.1 ~ 1.0 depending on geometry/training stage.

    2) raw_tangent_violation (before projection)
       - Sphere: |<m_t, v_m_raw>|
       - Hyperboloid: |<m_t, v_m_raw>_L|
       - Euclid: 0
       This tells whether v_m_raw is mostly normal (large violation) or already tangent-ish.

    3) v_m_star_norm_mean (target manifold speed)
       - If this is ~0, your supervision for v_m_star is wrong or vanishing.
       - Should be non-trivial if your geodesic endpoints differ meaningfully.

    Also logs loss_m for context.
    """

    kind = ops.cfg.kind

    # -----------------------
    # 1) projection retention
    # -----------------------
    raw_norm = v_m_raw.flatten(1).norm(dim=1).clamp_min(eps)   # [B]
    pred_norm = v_m_pred.flatten(1).norm(dim=1)                # [B]
    proj_ratio = (pred_norm / raw_norm)                        # [B]

    writer.add_scalar(f"{prefix}/proj_ratio_mean", proj_ratio.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/proj_ratio_std", proj_ratio.std().item(), global_step)
    writer.add_scalar(f"{prefix}/v_m_raw_norm_mean", raw_norm.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/v_m_pred_norm_mean", pred_norm.mean().item(), global_step)

    # ---------------------------------------
    # 2) raw tangent violation (pre-projection)
    # ---------------------------------------
    if kind == "sphere":
        # Euclidean dot should be ~0 for tangent vectors; large means normal-ish
        viol = (v_m_raw * m_t).sum(dim=-1).abs()  # [B]
    elif kind == "hyperboloid":
        # Lorentz dot should be ~0 for tangent vectors
        viol = ops.hyp_lorentz_inner(m_t, v_m_raw).abs()  # [B]
    else:
        viol = torch.zeros_like(raw_norm)

    writer.add_scalar(f"{prefix}/raw_tangent_violation_mean", viol.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/raw_tangent_violation_std", viol.std().item(), global_step)

    # ---------------------------------------
    # 3) target manifold speed magnitude
    # ---------------------------------------
    if kind == "hyperboloid":
        # intrinsic norm sqrt(<v,v>_L) on tangent
        vv = ops.hyp_lorentz_inner(ops.hyp_proj_tangent(m_t, v_m_star), ops.hyp_proj_tangent(m_t, v_m_star))
        vstar_norm = torch.sqrt(torch.clamp(vv, min=ops.cfg.eps))
    else:
        vstar_norm = v_m_star.flatten(1).norm(dim=1)

    writer.add_scalar(f"{prefix}/v_m_star_norm_mean", vstar_norm.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/v_m_star_norm_std", vstar_norm.std().item(), global_step)

    # context
    writer.add_scalar(f"{prefix}/loss_m", float(loss_m.detach().cpu()), global_step)