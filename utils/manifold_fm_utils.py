# ============================================================
# Iteration-2: Product-space Flow Matching on Z x M
# - Learn vector fields (v_z, v_m) with v_m in tangent bundle T_{m_t}M
# - Supports: Euclidean M=R^d, Sphere S^{d-1}, Hyperboloid H_c^d (Lorentz model)
#
# Drop-in style: matches your VAE latent FM loop + TinyUNet spirit
# ============================================================

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# ============================================================
# Geometry ops: Euclidean / Sphere / Hyperboloid (Lorentz)
# ============================================================

@dataclass
class GeoConfig:
    kind: Literal["euclid", "sphere", "hyperboloid"]
    d: int                 # sphere: d_s; euclid: d_euclid; hyperboloid: d_h (spatial dim, ambient is d+1)
    c: float = 1.0         # curvature parameter for hyperboloid (and used in formulas)
    eps: float = 1e-6


class GeometryOps:
    def __init__(self, cfg: GeoConfig):
        self.cfg = cfg

    # ---- helpers ----
    def _safe_norm(self, x, dim=-1, keepdim=True):
        return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=self.cfg.eps))

    # =========================================================
    # EUCLIDEAN: M=R^d (no constraints)
    # =========================================================
    def euclid_origin(self, B, device):
        return torch.zeros(B, self.cfg.d, device=device)

    def euclid_geodesic(self, m1, m0, t):  # (1-t)m1 + t m0
        return (1.0 - t) * m1 + t * m0

    def euclid_velocity(self, m1, m0, t):
        return (m0 - m1)

    def euclid_proj_tangent(self, m, w):
        return w

    def euclid_proj_manifold(self, x):
        return x

    def euclid_metric_norm2(self, m, v):
        return (v * v).sum(dim=-1)

    # =========================================================
    # SPHERE: M=S^{d-1} embedded in R^d
    # =========================================================
    def sphere_origin(self, B, device):
        # north pole
        m0 = torch.zeros(B, self.cfg.d, device=device)
        m0[:, 0] = 1.0
        return m0

    def sphere_normalize(self, x):
        return x / (self._safe_norm(x, dim=-1, keepdim=True))

    def sphere_inner(self, a, b):
        return (a * b).sum(dim=-1)

    def sphere_proj_tangent(self, m, w):
        # w - <w,m> m
        dot = (w * m).sum(dim=-1, keepdim=True)
        return w - dot * m

    def sphere_log(self, m, q):
        # log_m(q) in T_m S
        ip = torch.clamp(self.sphere_inner(m, q), -1.0 + self.cfg.eps, 1.0 - self.cfg.eps)
        theta = torch.acos(ip)  # [B]
        # q_perp = q - cos(theta) m
        q_perp = q - ip[:, None] * m
        denom = torch.clamp(torch.sin(theta), min=self.cfg.eps)
        v = (theta / denom)[:, None] * q_perp
        return self.sphere_proj_tangent(m, v)

    def sphere_exp(self, m, v):
        # exp_m(v) on S
        v = self.sphere_proj_tangent(m, v)
        nv = self._safe_norm(v, dim=-1, keepdim=True)  # [B,1]
        nv_s = nv.squeeze(-1)
        # handle small nv with series-like safe ratio
        cos = torch.cos(nv_s)[:, None]
        sin_over = (torch.sin(nv_s) / torch.clamp(nv_s, min=self.cfg.eps))[:, None]
        out = cos * m + sin_over * v
        return self.sphere_normalize(out)

    def sphere_geodesic(self, m1, m0, t):
        # m_t = exp_{m1}(t * log_{m1}(m0))
        u = self.sphere_log(m1, m0)  # [B,d]
        return self.sphere_exp(m1, t * u)

    def sphere_velocity(self, m1, m0, t):
        # Analytic derivative of exp_{m1}(t u) with u=log_{m1}(m0)
        u = self.sphere_log(m1, m0)
        theta = self._safe_norm(u, dim=-1, keepdim=True)  # [B,1]
        u_hat = u / theta
        th = theta.squeeze(-1)
        # dm/dt = theta*(-sin(t theta) m1 + cos(t theta) u_hat)
        dm = th[:, None] * (-torch.sin(t.squeeze(-1) * th)[:, None] * m1 + torch.cos(t.squeeze(-1) * th)[:, None] * u_hat)
        # ensure tangent at m_t
        m_t = self.sphere_exp(m1, t * u)
        return self.sphere_proj_tangent(m_t, dm)

    def sphere_proj_manifold(self, x):
        return self.sphere_normalize(x)

    def sphere_metric_norm2(self, m, v):
        # induced Euclidean on tangent
        v = self.sphere_proj_tangent(m, v)
        return (v * v).sum(dim=-1)

    # =========================================================
    # HYPERBOLOID: Lorentz model H_c^d embedded in R^{d+1}
    # <h,h>_L = -1/c, with <a,b>_L = -a0 b0 + sum_{i>=1} ai bi
    # =========================================================
    def hyp_origin(self, B, device):
        h0 = torch.zeros(B, self.cfg.d + 1, device=device)
        h0[:, 0] = 1.0 / math.sqrt(self.cfg.c)
        return h0

    def hyp_lorentz_inner(self, a, b):
        # returns [B]
        return -a[:, 0] * b[:, 0] + (a[:, 1:] * b[:, 1:]).sum(dim=-1)

    def hyp_proj_tangent(self, h, w):
        # Pi_{T_h}(w) = w + c <h,w>_L h
        hw = self.hyp_lorentz_inner(h, w)  # [B]
        return w + (self.cfg.c * hw)[:, None] * h

    def hyp_metric_norm2(self, h, v):
        v = self.hyp_proj_tangent(h, v)
        # On tangent, <v,v>_L is positive
        return torch.clamp(self.hyp_lorentz_inner(v, v), min=0.0)

    def hyp_exp(self, h, u):
        # exp_h(u), u in T_h
        u = self.hyp_proj_tangent(h, u)
        nu2 = torch.clamp(self.hyp_lorentz_inner(u, u), min=self.cfg.eps)  # [B]
        nu = torch.sqrt(nu2)  # [B]
        k = math.sqrt(self.cfg.c) * nu  # [B]
        # cosh(k) h + sinh(k)/(k) u
        cosh = torch.cosh(k)[:, None]
        sinh_over = (torch.sinh(k) / torch.clamp(k, min=self.cfg.eps))[:, None]
        out = cosh * h + sinh_over * u
        return self.hyp_proj_manifold(out)

    def hyp_log(self, h, q):
        # log_h(q) in T_h
        # alpha = -c <h,q>_L >= 1
        ip = self.hyp_lorentz_inner(h, q)
        alpha = torch.clamp(-self.cfg.c * ip, min=1.0 + self.cfg.eps)
        dist = (1.0 / math.sqrt(self.cfg.c)) * torch.acosh(alpha)  # [B]
        # v = q - alpha h
        v = q - alpha[:, None] * h
        v = self.hyp_proj_tangent(h, v)
        denom = torch.clamp(torch.sinh(math.sqrt(self.cfg.c) * dist), min=self.cfg.eps)
        out = (dist / denom)[:, None] * v
        return self.hyp_proj_tangent(h, out)

    def hyp_geodesic(self, h1, h0, t):
        u = self.hyp_log(h1, h0)
        return self.hyp_exp(h1, t * u)

    def hyp_velocity(self, h1, h0, t):
        # Differentiate h_t = cosh(k t) h1 + sinh(k t)/(k) u
        u = self.hyp_log(h1, h0)  # tangent at h1
        nu2 = torch.clamp(self.hyp_lorentz_inner(u, u), min=self.cfg.eps)
        nu = torch.sqrt(nu2)
        k = math.sqrt(self.cfg.c) * nu  # [B]
        kt = k * t.squeeze(-1)          # [B]
        # dh/dt = k*sinh(k t) h1 + cosh(k t) u
        dh = (k * torch.sinh(kt))[:, None] * h1 + (torch.cosh(kt))[:, None] * u
        # ensure tangent at current point
        h_t = self.hyp_exp(h1, t * u)
        return self.hyp_proj_tangent(h_t, dh)

    def hyp_proj_manifold(self, x):
        # Project arbitrary ambient x to the upper sheet: set time coord to satisfy constraint
        # Keep spatial part, fix x0 = sqrt(1/c + ||x_spatial||^2)
        spatial = x[:, 1:]
        s2 = (spatial * spatial).sum(dim=-1)  # [B]
        x0 = torch.sqrt(torch.clamp((1.0 / self.cfg.c) + s2, min=self.cfg.eps))
        out = x.clone()
        out[:, 0] = x0
        return out

    # =========================================================
    # Unified interface
    # =========================================================
    def origin(self, B, device):
        if self.cfg.kind == "euclid":
            return self.euclid_origin(B, device)
        if self.cfg.kind == "sphere":
            return self.sphere_origin(B, device)
        if self.cfg.kind == "hyperboloid":
            return self.hyp_origin(B, device)
        raise ValueError(self.cfg.kind)

    def geodesic(self, m1, m0, t):
        if self.cfg.kind == "euclid":
            return self.euclid_geodesic(m1, m0, t)
        if self.cfg.kind == "sphere":
            return self.sphere_geodesic(m1, m0, t)
        if self.cfg.kind == "hyperboloid":
            return self.hyp_geodesic(m1, m0, t)
        raise ValueError(self.cfg.kind)

    def velocity(self, m1, m0, t):
        if self.cfg.kind == "euclid":
            return self.euclid_velocity(m1, m0, t)
        if self.cfg.kind == "sphere":
            return self.sphere_velocity(m1, m0, t)
        if self.cfg.kind == "hyperboloid":
            return self.hyp_velocity(m1, m0, t)
        raise ValueError(self.cfg.kind)

    def proj_tangent(self, m, w):
        if self.cfg.kind == "euclid":
            return self.euclid_proj_tangent(m, w)
        if self.cfg.kind == "sphere":
            return self.sphere_proj_tangent(m, w)
        if self.cfg.kind == "hyperboloid":
            return self.hyp_proj_tangent(m, w)
        raise ValueError(self.cfg.kind)

    def proj_manifold(self, x):
        if self.cfg.kind == "euclid":
            return self.euclid_proj_manifold(x)
        if self.cfg.kind == "sphere":
            return self.sphere_proj_manifold(x)
        if self.cfg.kind == "hyperboloid":
            return self.hyp_proj_manifold(x)
        raise ValueError(self.cfg.kind)

    def metric_norm2(self, m, v):
        if self.cfg.kind == "euclid":
            return self.euclid_metric_norm2(m, v)
        if self.cfg.kind == "sphere":
            return self.sphere_metric_norm2(m, v)
        if self.cfg.kind == "hyperboloid":
            return self.hyp_metric_norm2(m, v)
        raise ValueError(self.cfg.kind)

@torch.no_grad()
def pack_cond_mt_e(
    *,
    ops,
    m_t: torch.Tensor,
    e: Optional[torch.Tensor],
    expect_m_dim: Optional[int] = None,
    expect_e_dim: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Unified conditioning packer for cond = [m_t, e].

    - Works for all geometries:
        * euclid:      m_t is [B, d]
        * sphere:      m_t is [B, d] (unit norm)
        * hyperboloid: m_t is [B, d+1] (Lorentz ambient)
    - Ensures:
        * correct device + dtype alignment
        * e can be None (filled with zeros)
        * safe shape checks (optional)
    Returns:
        cond_vec: [B, m_dim + e_dim]
        info: dict with dimensions and offsets for debugging/logging
    """
    if dtype is None:
        dtype = m_t.dtype

    device = m_t.device
    B = m_t.shape[0]
    m_dim = m_t.shape[1]

    if expect_m_dim is not None:
        assert m_dim == expect_m_dim, f"m_t dim mismatch: got {m_dim}, expected {expect_m_dim}"

    # If e missing, create zeros of expected dimension (or infer 1)
    if e is None:
        e_dim = 1 if expect_e_dim is None else expect_e_dim
        e = torch.zeros(B, e_dim, device=device, dtype=dtype)
    else:
        e = e.to(device=device, dtype=dtype)
        if e.dim() == 1:
            e = e.view(B, 1)
        e_dim = e.shape[1]
        if expect_e_dim is not None:
            assert e_dim == expect_e_dim, f"e dim mismatch: got {e_dim}, expected {expect_e_dim}"

    m_t = m_t.to(device=device, dtype=dtype)

    # Optional: geometry safety clamps (no-ops for euclid)
    # Sphere: keep on unit norm
    # Hyperboloid: keep on manifold constraint (upper sheet projection)
    if hasattr(ops, "proj_manifold") and ops.cfg.kind != "euclid":
        m_t = ops.proj_manifold(m_t)

    cond = torch.cat([m_t, e], dim=-1)

    info = {
        "B": B,
        "m_dim": m_dim,
        "e_dim": e_dim,
        "cond_dim": m_dim + e_dim,
        "m_offset": 0,
        "e_offset": m_dim,
    }
    return cond, info