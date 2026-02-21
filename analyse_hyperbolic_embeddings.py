import os
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------
# Hyperbolic geometry utils
# ----------------------------
def lorentz_dot(x, y):
    """
    Lorentz inner product <x,y>_L with signature (-,+,+,...,+).
    x,y: (..., d+1)
    returns: (...,)
    """
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def arcosh(x, eps=1e-7):
    # stable arcosh; assumes x >= 1
    x = torch.clamp(x, min=1.0 + eps)
    return torch.log(x + torch.sqrt(x * x - 1.0))


def hyp_dist(x, y, c=1.0, eps=1e-7):
    """
    Hyperbolic distance on hyperboloid model of curvature -c.
    d(x,y) = arcosh(-c * <x,y>_L) / sqrt(c)
    """
    alpha = -c * lorentz_dot(x, y)
    # Exact 0 for near-identical points:
    mask = alpha <= (1.0 + eps)
    out = torch.zeros_like(alpha)
    out[~mask] = arcosh(alpha[~mask], eps=eps) / math.sqrt(c)
    return out


def poincare_project(h):
    """
    Hyperboloid -> Poincaré ball (d-dim):
      x = h_spatial / (h0 + 1)
    h: [N, d+1]
    returns: [N, d]
    """
    denom = h[:, :1] + 1.0
    denom = torch.clamp(denom, min=1e-8)
    return h[:, 1:] / denom


# ----------------------------
# Stats utils (no SciPy)
# ----------------------------
def pearsonr_torch(x, y, eps=1e-12):
    x = x.float().flatten()
    y = y.float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = torch.sqrt((x * x).sum() + eps) * torch.sqrt((y * y).sum() + eps)
    return (num / den).item()


def rankdata(x):
    """
    Simple rankdata with average ranks for ties (numpy-like).
    x: 1D torch tensor
    returns: 1D float torch tensor ranks starting at 1
    """
    x = x.detach().cpu().numpy()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)

    # handle ties: average ranks among equal values
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1

    # convert to 1..N ranks
    return torch.from_numpy(ranks + 1.0).float()


def spearmanr_torch(x, y):
    rx = rankdata(torch.as_tensor(x))
    ry = rankdata(torch.as_tensor(y))
    return pearsonr_torch(rx, ry)


# ----------------------------
# PCA (simple SVD, no sklearn)
# ----------------------------
def pca_2d(X):
    """
    X: [N, D] torch
    returns: [N,2] torch
    """
    X = X.float()
    Xc = X - X.mean(dim=0, keepdim=True)
    # SVD: Xc = U S V^T
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    # take first 2 PCs
    return Xc @ Vt[:2].T


# ----------------------------
# Analysis: correlation, plots, heatmap
# ----------------------------
@torch.no_grad()
def run_checks(u, h1, severity, outdir, c=1.0, max_points_heatmap=256):
    """
    u: [N, d] tangent vectors at origin
    h1: [N, d+1] hyperboloid endpoints (t=1)
    severity: [N] float severity labels (sigma/stage/etc.)
    """
    os.makedirs(outdir, exist_ok=True)

    device = h1.device
    severity = severity.to(device).float().flatten()
    u = u.to(device).float()
    h1 = h1.to(device).float()

    # -------------------------
    # (1) Severity correlation with radius
    # -------------------------
    radius = torch.norm(u, dim=-1)  # [N]
    pear = pearsonr_torch(radius, severity)
    spear = spearmanr_torch(radius, severity)

    print(f"[Correlation] Pearson(radius, severity) = {pear:.4f}")
    print(f"[Correlation] Spearman(radius, severity) = {spear:.4f}")

    # Scatter plot
    plt.figure()
    plt.scatter(severity.detach().cpu().numpy(),
                radius.detach().cpu().numpy(),
                s=12, alpha=0.7)
    plt.xlabel("severity")
    plt.ylabel("||u|| (radius)")
    plt.title(f"Radius vs severity (Pearson={pear:.3f}, Spearman={spear:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "radius_vs_severity.png"), dpi=200)
    plt.close()

    # -------------------------
    # (2) Poincaré projection visualization
    # -------------------------
    x_ball = poincare_project(h1)        # [N, d]
    x2 = pca_2d(x_ball)                  # [N, 2]
    x2_np = x2.detach().cpu().numpy()
    sev_np = severity.detach().cpu().numpy()

    plt.figure()
    sc = plt.scatter(x2_np[:, 0], x2_np[:, 1], c=sev_np, s=14, alpha=0.85)
    plt.colorbar(sc, label="severity")
    plt.title("Poincaré-ball projection (PCA-2D) colored by severity")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "poincare_pca2d_colored_by_severity.png"), dpi=200)
    plt.close()

    # Optional: also color by radius
    rad_np = radius.detach().cpu().numpy()
    plt.figure()
    sc = plt.scatter(x2_np[:, 0], x2_np[:, 1], c=rad_np, s=14, alpha=0.85)
    plt.colorbar(sc, label="||u|| (radius)")
    plt.title("Poincaré-ball projection (PCA-2D) colored by radius")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "poincare_pca2d_colored_by_radius.png"), dpi=200)
    plt.close()

    # -------------------------
    # (3) Hyperbolic distance heatmap (sorted by severity)
    # -------------------------
    N = h1.shape[0]
    if N > max_points_heatmap:
        # subsample evenly after sorting
        idx_sort = torch.argsort(severity)
        idx = idx_sort[torch.linspace(0, N - 1, max_points_heatmap).long()]
        h_sub = h1[idx]
        sev_sub = severity[idx]
    else:
        idx = torch.argsort(severity)
        h_sub = h1[idx]
        sev_sub = severity[idx]

    M = h_sub.shape[0]
    # compute pairwise distances
    # D[i,j] = d(h_i, h_j)
    D = torch.empty((M, M), device=device, dtype=torch.float32)
    for i in range(M):
        D[i] = hyp_dist(h_sub[i].unsqueeze(0).expand(M, -1), h_sub, c=c)

    D_np = D.detach().cpu().numpy()
    plt.figure(figsize=(7, 6))
    plt.imshow(D_np, aspect="auto")
    plt.colorbar(label="hyperbolic distance")
    plt.title("Pairwise hyperbolic distance heatmap (sorted by severity)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hyperbolic_distance_heatmap_sorted.png"), dpi=200)
    plt.close()

    # Simple “bandedness” diagnostics:
    # mean distance to k-nearest neighbors along sorted order should be small
    k = max(1, M // 50)  # ~2% neighbors
    # distances to near diagonal bands
    diag_band = []
    far_band = []
    for i in range(M):
        j0 = max(0, i - k)
        j1 = min(M, i + k + 1)
        diag_band.append(D[i, j0:j1].mean().item())
        far_band.append(D[i, :].mean().item())
    diag_mean = float(np.mean(diag_band))
    far_mean = float(np.mean(far_band))
    print(f"[Heatmap] mean local-band distance (k={k}) = {diag_mean:.4f}")
    print(f"[Heatmap] mean global distance             = {far_mean:.4f}")
    if far_mean > 1e-8:
        print(f"[Heatmap] band/global ratio              = {diag_mean / far_mean:.4f}")

    # save a small plot of severity along sorted index (sanity)
    plt.figure()
    plt.plot(sev_sub.detach().cpu().numpy())
    plt.title("Severity after sorting (sanity check)")
    plt.xlabel("sorted index")
    plt.ylabel("severity")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "severity_sorted_curve.png"), dpi=200)
    plt.close()


def load_tensors(pt_path):
    """
    Expects a dict with keys: 'u', 'h1', 'severity'
    - u: [N, d]
    - h1: [N, d+1]
    - severity: [N]
    """
    data = torch.load(pt_path, map_location="cpu")
    for k in ["u", "h1", "severity"]:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {pt_path}. Found keys: {list(data.keys())}")
    return data["u"], data["h1"], data["severity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pt", type=str, required=True,
                    help="Path to .pt file containing {'u','h1','severity'} tensors.")
    ap.add_argument("--outdir", type=str, default="manifold_checks_out",
                    help="Output directory for plots.")
    ap.add_argument("--c", type=float, default=1.0, help="Curvature parameter c (space curvature = -c).")
    ap.add_argument("--max_heatmap", type=int, default=256, help="Max points to use for distance heatmap.")
    args = ap.parse_args()

    u, h1, sev = load_tensors(args.in_pt)
    run_checks(u=u, h1=h1, severity=sev, outdir=args.outdir, c=args.c, max_points_heatmap=args.max_heatmap)
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()

'''
python analyze_hyperbolic_embeddings.py \
  --in_pt hyp_epoch7_embeddings.pt \
  --outdir hyp_epoch7_checks \
  --c 1.0 \
  --max_heatmap 256
'''

'''
### Save embeddings and severity scores
torch.save({"u": u_cpu, "h1": h1_cpu, "severity": severity_cpu}, "hyp_epoch7_embeddings.pt")
'''