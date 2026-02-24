import torch
import torch.nn as nn
    
class HyperbolicDegradationEncoderOnManifold(nn.Module):
    """
    Outputs h1 on Lorentz hyperboloid H^d_c via exp at origin using bounded tangent u,
    and Euclidean severity e.

    Key stability tweaks vs iteration-1:
    - avoid forcing u_dir when u_raw~0 (use smooth clamp)
    - use softplus for radius (>=0) instead of abs(tanh(.)) which can kill gradients near 0
    - keep radius reasonably bounded but not saturated early (r_max + softplus scaling)
    """
    """
    Corrected stability-focused version.

    Fixes vs the previous snippet:
    1) head_u MUST NOT be all-zeros (avoids dead / directionless start).
    2) radius head init chosen to start near the origin but with non-zero gradients.
    3) safer direction normalization when u_raw is tiny.
    4) avoids any silent saturation early (still bounded by r_max).
    """
    def __init__(self, z_channels=4, d_h=8, d_e=1, hidden=256, c=1.0, r_max=6.0, u_init_std=1e-3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(z_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head_u = nn.Linear(hidden, d_h)  # tangent direction (spatial at origin)
        self.head_r = nn.Linear(hidden, 1)    # radius scalar
        self.head_e = nn.Linear(hidden, d_e)

        self.c = float(c)
        self.r_max = float(r_max)

        # --- IMPORTANT: non-zero head_u init ---
        # Small random weights: gives a direction signal from the start.
        nn.init.normal_(self.head_u.weight, mean=0.0, std=u_init_std)
        nn.init.zeros_(self.head_u.bias)

        # Radius: start near 0 (close to origin), but keep gradients alive.
        # sigmoid(r_raw) with r_raw ~ -5 => small radius ~ r_max * 0.0067
        nn.init.zeros_(self.head_r.weight)
        nn.init.constant_(self.head_r.bias, -5.0)

        # severity head: keep default init (or mild)
        # nn.init.zeros_(self.head_e.bias)  # optional

    def forward(self, z):
        B = z.size(0)
        device = z.device
        dtype = z.dtype

        h = self.pool(z).flatten(1)
        h = self.mlp(h)

        u_raw = self.head_u(h)        # [B, d_h]
        r_raw = self.head_r(h)        # [B, 1]
        e = self.head_e(h)            # [B, d_e]

        # Safe direction normalization
        u_norm = torch.norm(u_raw, dim=-1, keepdim=True).clamp_min(1e-6)
        u_dir = u_raw / u_norm

        # Bounded radius
        r = self.r_max * torch.sigmoid(r_raw)   # [B,1] in [0, r_max]
        u = r * u_dir                           # [B, d_h]

        # Ambient tangent at origin: (0, u)
        u_amb = torch.cat([torch.zeros(B, 1, device=device, dtype=dtype), u], dim=-1)  # [B, d_h+1]

        # Origin on hyperboloid: [1/sqrt(c), 0, ..., 0]  (NO in-place)
        h0 = torch.zeros(B, self.head_u.out_features + 1, device=device, dtype=dtype)
        h0 = torch.cat(
            [torch.full((B, 1), 1.0 / math.sqrt(self.c), device=device, dtype=dtype),
            torch.zeros(B, self.head_u.out_features, device=device, dtype=dtype)],
            dim=-1
        )  # [B, d_h+1]

        # exp_{h0}(u) closed-form
        nu = torch.norm(u, dim=-1).clamp_min(1e-6)  # [B]
        k = math.sqrt(self.c) * nu                  # [B]
        cosh = torch.cosh(k)[:, None]
        sinh_over = (torch.sinh(k) / k)[:, None]
        h1 = cosh * h0 + sinh_over * u_amb          # [B, d_h+1]

        # Enforce manifold constraint without in-place:
        spatial = h1[:, 1:]                         # [B, d_h]
        x0 = torch.sqrt((1.0 / self.c) + (spatial * spatial).sum(dim=-1)).clamp_min(1e-6)  # [B]
        h1 = torch.cat([x0[:, None], spatial], dim=-1)  # [B, d_h+1]

        return h1, e