import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------
# 1) Basic geometry utilities
# -----------------------------
def normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def safe_acos(x, eps=1e-7):
    return torch.acos(torch.clamp(x, -1 + eps, 1 - eps))

def slerp(s0, s1, t, eps=1e-7):
    """
    s0, s1: [B, d] unit vectors
    t: [B, 1] in [0,1]
    returns: st [B, d] unit
    """
    s0 = normalize(s0)
    s1 = normalize(s1)
    dot = (s0 * s1).sum(dim=-1, keepdim=True)
    omega = safe_acos(dot, eps=eps)
    sin_omega = torch.sin(omega).clamp_min(eps)
    a = torch.sin((1 - t) * omega) / sin_omega
    b = torch.sin(t * omega) / sin_omega
    out = a * s0 + b * s1
    return normalize(out)

def make_sphere_identity(B, d_s, device):
    s0 = torch.zeros(B, d_s, device=device)
    s0[:, 0] = 1.0
    return s0

# Sphere geodesic distance
def sphere_dist(a, b, eps=1e-7):
    """
    a,b: [...,d] unit
    returns: [...,1] angle distance
    """
    a = normalize(a)
    b = normalize(b)
    dot = (a * b).sum(dim=-1, keepdim=True)
    return safe_acos(dot, eps=eps)

# -----------------------------
# 2) Lorentz hyperboloid utilities (for analysis)
# -----------------------------
def arcosh(x, eps=1e-7):
    x = torch.clamp(x, min=1.0 + eps)
    return torch.log(x + torch.sqrt(x * x - 1.0))

def sinhc(x, eps=1e-7):
    x_abs = torch.abs(x)
    return torch.where(x_abs < 1e-4, 1.0 + (x * x) / 6.0, torch.sinh(x) / torch.clamp(x, min=eps))

def lorentz_dot(x, y):
    """
    x,y: (..., d+1)
    returns (...,1)
    """
    xy = x * y
    return (-xy[..., :1] + xy[..., 1:].sum(dim=-1, keepdim=True))

def hyperboloid_origin(d, device, c=1.0):
    o = torch.zeros(d + 1, device=device)
    o[0] = 1.0 / math.sqrt(c)
    return o

def expmap0(u_euc, c=1.0, eps=1e-7):
    """
    u_euc: (..., d) tangent at origin
    returns y: (..., d+1) on hyperboloid
    """
    v_norm = torch.norm(u_euc, dim=-1, keepdim=True).clamp_min(eps)
    sqrt_c = math.sqrt(c)
    u = sqrt_c * v_norm
    y0 = torch.cosh(u) / math.sqrt(c)
    ysp = sinhc(u, eps=eps) * u_euc
    return torch.cat([y0, ysp], dim=-1)

def hyperbolic_dist0_from_u(u_euc, c=1.0, eps=1e-7):
    """
    Distance from origin to expmap0(u) is ||u|| in the hyperbolic metric scaled by curvature:
    Actually: d(o, exp_o(u)) = ||u|| (when u is tangent length in Riemannian metric at origin).
    Our u_euc is Euclidean tangent coords; for origin, that matches.
    We'll compute explicitly via Lorentz distance for robustness.
    """
    device = u_euc.device
    d = u_euc.shape[-1]
    o = hyperboloid_origin(d, device, c=c).view(*([1] * (u_euc.dim() - 1)), d + 1)
    y = expmap0(u_euc, c=c, eps=eps)
    # d(o,y)=arcosh(-c<o,y>_L)/sqrt(c)
    alpha = -c * lorentz_dot(o, y)
    return arcosh(alpha, eps=eps) / math.sqrt(c)

def hyperbolic_dist(x, y, c=1.0, eps=1e-7):
    alpha = -c * lorentz_dot(x, y)
    return arcosh(alpha, eps=eps) / math.sqrt(c)

# -----------------------------
# 3) Image-quality metrics (optional, but useful)
# -----------------------------
def psnr_torch(x, y, eps=1e-8):
    """
    x,y: [B,3,H,W] assumed in [0,1] or [-1,1] consistently.
    """
    mse = torch.mean((x - y) ** 2, dim=(1,2,3)).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)

# -----------------------------
# 4) PCA helper (no seaborn)
# -----------------------------
def pca_project(X: np.ndarray, out_dim=2) -> np.ndarray:
    """
    X: [N, D]
    returns: [N, out_dim]
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD PCA
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:out_dim].T  # [D,out_dim]
    return Xc @ W

# -----------------------------
# 5) Data structures for analysis
# -----------------------------
@dataclass
class TrajData:
    t: torch.Tensor                  # [T]
    # Sphere
    s_traj: Optional[torch.Tensor] = None   # [B,T,d_s]
    e_traj_s: Optional[torch.Tensor] = None # [B,T,d_e]
    # Hyperbolic tangent
    u_traj: Optional[torch.Tensor] = None   # [B,T,d_h]
    e_traj_h: Optional[torch.Tensor] = None # [B,T,d_e]
    # Hyperboloid points (for analysis)
    h_traj: Optional[torch.Tensor] = None   # [B,T,d_h+1]
    # Optional labels
    sev: Optional[torch.Tensor] = None      # [B,1] or [B]
    deg_id: Optional[torch.Tensor] = None   # [B]

# -----------------------------
# 6) Collect trajectories from models
# -----------------------------
@torch.no_grad()
def collect_trajs(
    dataloader,
    vae,
    scaling_factor: float,
    # Spherical
    deg_enc_sphere=None,
    model_sphere=None,
    d_s=8,
    d_e=1,
    # Hyperbolic
    deg_enc_hyp=None,
    model_hyp=None,
    d_h=8,
    c=1.0,
    # Sampling
    T=11,
    max_batches=10,
    device="cuda",
):
    """
    Collect trajectories m_t inferred from degraded samples.
    This does NOT integrate the ODE; it only evaluates manifold states across timesteps.
    """
    t_grid = torch.linspace(0.0, 1.0, T, device=device)  # [T]
    all_s, all_es, all_u, all_eh, all_h = [], [], [], [], []
    all_sev, all_deg_id = [], []

    for bi, batch in enumerate(dataloader):
        if bi >= max_batches:
            break

        # batch could be tuple or dict
        if isinstance(batch, (list, tuple)):
            degraded, clean = batch[0].to(device), batch[1].to(device)
            extra = {}
        else:
            degraded, clean = batch["degraded"].to(device), batch["clean"].to(device)
            extra = batch

        with torch.no_grad():
            z_src = vae.encode(degraded).latent_dist.sample() * scaling_factor

        B = z_src.size(0)

        # optional labels
        if "sev" in extra:
            sev = extra["sev"].to(device)
            all_sev.append(sev.detach().float().view(B, -1))
        if "deg_id" in extra:
            deg_id = extra["deg_id"].to(device)
            all_deg_id.append(deg_id.detach().view(-1))

        # Sphere trajectories (constructed from s1,e1)
        if deg_enc_sphere is not None and model_sphere is not None:
            s1, e1 = deg_enc_sphere(z_src)  # s1 [B,d_s], e1 [B,d_e]
            s0 = make_sphere_identity(B, d_s, device)
            e0 = torch.zeros(B, d_e, device=device)

            s_traj = []
            e_traj = []
            for t in t_grid:
                t1 = t.view(1,1).expand(B,1)  # [B,1]
                st = slerp(s0, s1, t1)        # [B,d_s]
                et = (1 - t1) * e0 + t1 * e1  # [B,d_e]
                s_traj.append(st)
                e_traj.append(et)
            s_traj = torch.stack(s_traj, dim=1)  # [B,T,d_s]
            e_traj = torch.stack(e_traj, dim=1)  # [B,T,d_e]
            all_s.append(s_traj.cpu())
            all_es.append(e_traj.cpu())

        # Hyperbolic tangent trajectories (u_t = t * u)
        if deg_enc_hyp is not None and model_hyp is not None:
            u, e1h = deg_enc_hyp(z_src)  # u [B,d_h], e1 [B,d_e]
            e0 = torch.zeros(B, d_e, device=device)

            u_traj = []
            e_traj = []
            h_traj = []
            for t in t_grid:
                t1 = t.view(1,1).expand(B,1)  # [B,1]
                ut = t1 * u
                et = (1 - t1) * e0 + t1 * e1h
                u_traj.append(ut)
                e_traj.append(et)

                # analysis: map to hyperboloid point h_t = expmap0(u_t)
                ht = expmap0(ut, c=c)
                h_traj.append(ht)

            u_traj = torch.stack(u_traj, dim=1)  # [B,T,d_h]
            e_traj = torch.stack(e_traj, dim=1)  # [B,T,d_e]
            h_traj = torch.stack(h_traj, dim=1)  # [B,T,d_h+1]

            all_u.append(u_traj.cpu())
            all_eh.append(e_traj.cpu())
            all_h.append(h_traj.cpu())

    # concat batches
    def cat_or_none(xs):
        return torch.cat(xs, dim=0) if len(xs) > 0 else None

    traj = TrajData(
        t=t_grid.cpu(),
        s_traj=cat_or_none(all_s),
        e_traj_s=cat_or_none(all_es),
        u_traj=cat_or_none(all_u),
        e_traj_h=cat_or_none(all_eh),
        h_traj=cat_or_none(all_h),
        sev=cat_or_none(all_sev),
        deg_id=cat_or_none(all_deg_id),
    )
    return traj

# -----------------------------
# 7) Metrics
# -----------------------------
def sphere_on_manifold_residual(s_traj: torch.Tensor) -> torch.Tensor:
    """
    s_traj: [N,T,d]
    returns: [N,T] | ||s||-1 |
    """
    nrm = torch.norm(s_traj, dim=-1)
    return torch.abs(nrm - 1.0)

def hyperboloid_on_manifold_residual(h_traj: torch.Tensor, c=1.0) -> torch.Tensor:
    """
    h_traj: [N,T,d+1]
    residual: | <h,h>_L + 1/c |
    """
    # compute lorentz dot per point
    x0 = h_traj[..., :1]
    xs = h_traj[..., 1:]
    ll = -(x0 * x0) + (xs * xs).sum(dim=-1, keepdim=True)  # [N,T,1]
    return torch.abs(ll.squeeze(-1) + (1.0 / c))

def sphere_geodesic_linearity(s_traj: torch.Tensor, eps=1e-7) -> torch.Tensor:
    """
    Checks | d(s0,st) - t*d(s0,s1) | per sample per t.
    Here s0 is assumed to be [1,0,0...]
    """
    N, T, d = s_traj.shape
    device = s_traj.device
    s0 = torch.zeros(N, d, device=device)
    s0[:, 0] = 1.0
    s1 = s_traj[:, -1, :]  # t=1 endpoint (constructed)
    d01 = sphere_dist(s0, s1, eps=eps).squeeze(-1)  # [N]

    # for each t
    t_grid = torch.linspace(0, 1, T, device=device).view(1, T)
    d0t = sphere_dist(s0[:,None,:].expand(N,T,d), s_traj, eps=eps).squeeze(-1)  # [N,T]
    return torch.abs(d0t - t_grid * d01.view(N,1))

def hyperbolic_geodesic_linearity(h_traj: torch.Tensor, c=1.0, eps=1e-7) -> torch.Tensor:
    """
    Checks | d(o,ht) - t*d(o,h1) | using Lorentz distance.
    """
    N, T, Dp1 = h_traj.shape
    d = Dp1 - 1
    device = h_traj.device
    o = hyperboloid_origin(d, device, c=c).view(1,1,d+1).expand(N,T,d+1)
    h1 = h_traj[:, -1, :].unsqueeze(1)  # [N,1,d+1]
    o1 = o[:, :1, :]                    # [N,1,d+1]
    d01 = hyperbolic_dist(o1, h1, c=c, eps=eps).squeeze(-1).squeeze(-1)  # [N]

    t_grid = torch.linspace(0, 1, T, device=device).view(1, T)
    d0t = hyperbolic_dist(o, h_traj, c=c, eps=eps).squeeze(-1)  # [N,T]
    return torch.abs(d0t - t_grid * d01.view(N,1))

def metric_speed(traj: torch.Tensor, dist_fn, t: torch.Tensor) -> torch.Tensor:
    """
    traj: [N,T,dim]
    dist_fn: callable(a,b)->[N,1] or [N]
    t: [T] times in [0,1]
    returns: [N,T-1] speed approx = d(m_{i+1},m_i)/dt
    """
    N, T, _ = traj.shape
    dt = (t[1:] - t[:-1]).view(1, T-1).to(traj.device)  # [1,T-1]
    a = traj[:, :-1, :]
    b = traj[:, 1:, :]
    d = dist_fn(a, b)  # should return [N,T-1,1] or [N,T-1]
    if d.dim() == 3:
        d = d.squeeze(-1)
    return d / dt

def sphere_pairwise_dist(a, b, eps=1e-7):
    # a,b: [N,T,dim] -> [N,T,1]
    dot = (normalize(a) * normalize(b)).sum(dim=-1, keepdim=True)
    return safe_acos(dot, eps=eps)

def hyperbolic_pairwise_dist(a, b, c=1.0, eps=1e-7):
    # a,b: [N,T,dim+1] -> [N,T,1]
    alpha = -c * lorentz_dot(a, b)
    return arcosh(alpha, eps=eps) / math.sqrt(c)

# -----------------------------
# 8) Optional analyses
# -----------------------------
def content_invariance_ratio(m1: torch.Tensor, deg_id: torch.Tensor) -> Dict[str, float]:
    """
    m1: [N,D] endpoints
    deg_id: [N] integer groups (same degradation)
    Compute intra-group variance vs inter-group variance ratio.
    """
    deg_id = deg_id.cpu().numpy()
    m1_np = m1.cpu().numpy()

    # global variance
    global_var = float(np.mean(np.var(m1_np, axis=0)))

    # intra-group variance (average over groups)
    intra_vars = []
    for gid in np.unique(deg_id):
        idx = np.where(deg_id == gid)[0]
        if len(idx) < 2:
            continue
        intra_vars.append(float(np.mean(np.var(m1_np[idx], axis=0))))
    intra = float(np.mean(intra_vars)) if len(intra_vars) > 0 else float("nan")

    # ratio: lower is better (more invariant)
    return {
        "intra_var": intra,
        "global_var": global_var,
        "intra_over_global": intra / (global_var + 1e-12)
    }

def correlation_with_severity(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Pearson correlation between x and y (1D tensors).
    """
    x = x.detach().cpu().numpy().reshape(-1)
    y = y.detach().cpu().numpy().reshape(-1)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

# -----------------------------
# 9) Plotting helpers
# -----------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def plot_curve_family(curves: np.ndarray, t: np.ndarray, title: str, outpath: str, max_lines=30):
    """
    curves: [N,T]
    """
    plt.figure()
    N = curves.shape[0]
    for i in range(min(N, max_lines)):
        plt.plot(t, curves[i], linewidth=1)
    plt.plot(t, curves.mean(axis=0), linewidth=3)
    plt.xlabel("t")
    plt.ylabel("value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_hist(x: np.ndarray, title: str, outpath: str, bins=50):
    plt.figure()
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_pca_trajectories(traj: torch.Tensor, t: torch.Tensor, title: str, outpath: str, max_lines=30):
    """
    traj: [N,T,D]
    PCA on all points; plot polylines for a subset of N.
    """
    N, T, D = traj.shape
    X = traj.reshape(N*T, D).cpu().numpy()
    Y = pca_project(X, out_dim=2).reshape(N, T, 2)
    tt = t.cpu().numpy()

    plt.figure()
    for i in range(min(N, max_lines)):
        plt.plot(Y[i,:,0], Y[i,:,1], marker="o", linewidth=1)
        # optionally annotate endpoints
        plt.text(Y[i,0,0], Y[i,0,1], "0", fontsize=6)
        plt.text(Y[i,-1,0], Y[i,-1,1], "1", fontsize=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -----------------------------
# 10) Main analysis runner
# -----------------------------
@torch.no_grad()
def run_analysis(
    traj: TrajData,
    outdir: str,
    c_hyp: float = 1.0,
):
    ensure_dir(outdir)
    t = traj.t.numpy()

    results = {}

    # ---------- Sphere ----------
    if traj.s_traj is not None:
        s_traj = traj.s_traj  # [N,T,d_s]
        N, Tn, ds = s_traj.shape

        # on-manifold
        sph_res = sphere_on_manifold_residual(s_traj).numpy()  # [N,T]
        results["sphere_norm_res_mean"] = float(sph_res.mean())
        results["sphere_norm_res_std"] = float(sph_res.std())

        plot_curve_family(sph_res, t, "Sphere | ||s_t||-1 | vs t", os.path.join(outdir, "sphere_norm_res_vs_t.png"))

        # geodesic linearity
        sph_geo = sphere_geodesic_linearity(s_traj).numpy()
        results["sphere_geo_res_mean"] = float(sph_geo.mean())
        results["sphere_geo_res_std"] = float(sph_geo.std())
        plot_curve_family(sph_geo, t, "Sphere geodesic linearity residual vs t", os.path.join(outdir, "sphere_geo_res_vs_t.png"))

        # speed constancy
        sph_speed = metric_speed(s_traj, sphere_pairwise_dist, traj.t).numpy()  # [N,T-1]
        results["sphere_speed_mean"] = float(sph_speed.mean())
        results["sphere_speed_std"] = float(sph_speed.std())
        plot_curve_family(sph_speed, t[:-1], "Sphere metric speed vs t", os.path.join(outdir, "sphere_speed_vs_t.png"))

        # info content: endpoint angle to identity
        s0 = torch.zeros(N, ds); s0[:,0]=1.0
        s1 = s_traj[:, -1, :]
        theta = sphere_dist(s0, s1).squeeze(-1).numpy()
        plot_hist(theta, "Sphere endpoint angle d(s0,s1) distribution", os.path.join(outdir, "sphere_theta_hist.png"))
        results["sphere_theta_mean"] = float(theta.mean())
        results["sphere_theta_std"] = float(theta.std())

        # visualization: distance-to-identity vs t
        d0t = sphere_dist(s0[:,None,:].expand(N,Tn,ds), s_traj).squeeze(-1).numpy()
        plot_curve_family(d0t, t, "Sphere distance to identity vs t", os.path.join(outdir, "sphere_d0t_vs_t.png"))

        # PCA trajectories
        plot_pca_trajectories(s_traj, traj.t, "Sphere trajectories (PCA projection)", os.path.join(outdir, "sphere_pca_trajs.png"))

    # ---------- Hyperbolic ----------
    if traj.u_traj is not None and traj.h_traj is not None:
        u_traj = traj.u_traj  # [N,T,d_h]
        h_traj = traj.h_traj  # [N,T,d_h+1]
        N, Tn, dh = u_traj.shape

        # on-manifold residual in hyperboloid
        hyp_res = hyperboloid_on_manifold_residual(h_traj, c=c_hyp).numpy()  # [N,T]
        results["hyp_hyperboloid_res_mean"] = float(hyp_res.mean())
        results["hyp_hyperboloid_res_std"] = float(hyp_res.std())
        plot_curve_family(hyp_res, t, "Hyperboloid | <h,h>_L + 1/c | vs t", os.path.join(outdir, "hyp_hyperboloid_res_vs_t.png"))

        # hyperbolic geodesic linearity residual
        hyp_geo = hyperbolic_geodesic_linearity(h_traj, c=c_hyp).numpy()
        results["hyp_geo_res_mean"] = float(hyp_geo.mean())
        results["hyp_geo_res_std"] = float(hyp_geo.std())
        plot_curve_family(hyp_geo, t, "Hyperbolic geodesic linearity residual vs t", os.path.join(outdir, "hyp_geo_res_vs_t.png"))

        # speed constancy
        hyp_speed = metric_speed(h_traj, lambda a,b: hyperbolic_pairwise_dist(a,b,c=c_hyp), traj.t).numpy()
        results["hyp_speed_mean"] = float(hyp_speed.mean())
        results["hyp_speed_std"] = float(hyp_speed.std())
        plot_curve_family(hyp_speed, t[:-1], "Hyperbolic metric speed vs t", os.path.join(outdir, "hyp_speed_vs_t.png"))

        # info content: tangent radius and hyperbolic distance
        u1 = u_traj[:, -1, :]  # equals u if u_t=t*u
        r = torch.norm(u1, dim=-1).numpy()
        plot_hist(r, "Hyperbolic tangent radius ||u|| distribution", os.path.join(outdir, "hyp_u_radius_hist.png"))
        results["hyp_u_radius_mean"] = float(r.mean())
        results["hyp_u_radius_std"] = float(r.std())

        # distance-to-origin vs t
        d0t = hyperbolic_dist(h_traj[:, :1, :].expand_as(h_traj), h_traj, c=c_hyp).squeeze(-1).numpy()
        # Actually above computes d(h0, ht) where h0=first element at t=0 (origin if u_t=t*u); good.
        plot_curve_family(d0t, t, "Hyperbolic distance to origin vs t", os.path.join(outdir, "hyp_d0t_vs_t.png"))

        # PCA trajectories on tangent u_t (often cleaner)
        plot_pca_trajectories(u_traj, traj.t, "Hyperbolic tangent trajectories u_t (PCA)", os.path.join(outdir, "hyp_tangent_pca_trajs.png"))

        # PCA trajectories on hyperboloid points h_t (distorted but OK visually)
        plot_pca_trajectories(h_traj, traj.t, "Hyperboloid trajectories h_t (PCA)", os.path.join(outdir, "hyp_hyperboloid_pca_trajs.png"))

    # ---------- Optional: severity monotonicity + correlation ----------
    if traj.sev is not None:
        sev = traj.sev.squeeze(-1)  # [N] or [N,k]
        if sev.dim() > 1:
            sev = sev[:, 0]

        # For sphere: correlate endpoint angle with severity
        if traj.s_traj is not None:
            N, Tn, ds = traj.s_traj.shape
            s0 = torch.zeros(N, ds); s0[:,0]=1.0
            theta = sphere_dist(s0, traj.s_traj[:, -1, :]).squeeze(-1)
            corr = correlation_with_severity(theta, sev)
            results["corr_sphere_theta_vs_sev"] = corr

        # For hyperbolic: correlate ||u|| with severity
        if traj.u_traj is not None:
            u1 = traj.u_traj[:, -1, :]
            r = torch.norm(u1, dim=-1)
            corr = correlation_with_severity(r, sev)
            results["corr_hyp_radius_vs_sev"] = corr

    # ---------- Optional: content invariance ----------
    if traj.deg_id is not None:
        deg_id = traj.deg_id
        # Use endpoints as representation
        if traj.s_traj is not None:
            s1 = traj.s_traj[:, -1, :]
            results["sphere_content_invariance"] = content_invariance_ratio(s1, deg_id)
        if traj.u_traj is not None:
            u1 = traj.u_traj[:, -1, :]
            results["hyp_content_invariance"] = content_invariance_ratio(u1, deg_id)

    # Save results summary
    with open(os.path.join(outdir, "metrics_summary.txt"), "w") as f:
        for k,v in results.items():
            f.write(f"{k}: {v}\n")

    print(f"[Analysis] Saved plots and metrics to: {outdir}")
    return results
