import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

from models.eucliddegradationencoderonmanifold import EuclidDegradationEncoderOnManifold
from models.spheredegradationencoderonmanifold import SphereDegradationEncoderOnManifold
from models.hyperbolicdegradationencoderonmanifold import HyperbolicDegradationEncoderOnManifold
from utils.manifold_fm_utils import GeometryOps, GeoConfig, pack_cond_mt_e
from models.productvfmodel import ProductVFModel

def build_iteration2_components(
    kind: Literal["euclid", "sphere", "hyperboloid"],
    z_channels=4,
    d_m=8,     # sphere d_s or euclid dim or hyperbolic spatial dim d_h
    d_e=1,
    c=1.0,
    base_channels=64,
    time_emb_dim=128,
):
    ops = GeometryOps(GeoConfig(kind=kind, d=d_m, c=c))

    if kind == "euclid":
        deg_enc = EuclidDegradationEncoderOnManifold(z_channels=z_channels, d=d_m, d_e=d_e)
        m_out_dim = d_m
        cond_dim = d_m + d_e
    elif kind == "sphere":
        deg_enc = SphereDegradationEncoderOnManifold(z_channels=z_channels, d_s=d_m, d_e=d_e)
        m_out_dim = d_m
        cond_dim = d_m + d_e
    elif kind == "hyperboloid":
        deg_enc = HyperbolicDegradationEncoderOnManifold(z_channels=z_channels, d_h=d_m, d_e=d_e, c=c)
        m_out_dim = d_m + 1  # ambient (time + spatial)
        cond_dim = (d_m + 1) + d_e
    else:
        raise ValueError(kind)

    vf_model = ProductVFModel(
        z_channels=z_channels,
        time_emb_dim=time_emb_dim,
        base_channels=base_channels,
        cond_dim=cond_dim,
        m_out_dim=m_out_dim,
    )

    return ops, deg_enc, vf_model

# ============================================================
# Training step patch: learn (v_z, v_m) and enforce v_m in tangent bundle
# ============================================================

@dataclass
class TrainCfg:
    lambda_m: float = 0.5
    use_amp: bool = True


@torch.no_grad()
def make_m0(ops: GeometryOps, B: int, device):
    return ops.origin(B, device)

def fm_product_train_step(
    *,
    vae,
    scaling_factor: float,
    vf_model: ProductVFModel,
    deg_enc: nn.Module,
    ops: GeometryOps,
    degraded: torch.Tensor,
    clean: torch.Tensor,
    optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: TrainCfg,
):
    """
    One training step consistent with your loop style.
    """
    device = degraded.device
    B = degraded.size(0)

    with torch.no_grad():
        z0 = vae.encode(degraded).latent_dist.sample() * scaling_factor
        z1 = vae.encode(clean).latent_dist.sample() * scaling_factor

    # sample t in [0,1], broadcastable for z interpolation
    t = torch.rand(B, device=device)
    t_img = t.view(B, 1, 1, 1)

    # z path + target velocity
    z_t = (1.0 - t_img) * z0 + t_img * z1
    v_z_star = (z1 - z0)

    # manifold endpoints
    m1, e = deg_enc(z0)        # m1 on manifold (or R^d for euclid), e in R^{d_e}
    m0 = make_m0(ops, B, device)

    # geodesic point m_t and its true velocity \dot m_t
    tt = t.view(B, 1)  # [B,1] for vector ops
    m_t = ops.geodesic(m1, m0, tt)           # [B, d or d+1]
    v_m_star = ops.velocity(m1, m0, tt)      # [B, d or d+1]

    # conditioning vector = [m_t, e]
    # cond_vec = torch.cat([m_t, e], dim=-1)

    cond_vec, cond_info = pack_cond_mt_e(
                                ops=ops,
                                m_t=m_t,
                                e=e,
                                expect_m_dim=(ops.cfg.d + 1 if ops.cfg.kind == "hyperboloid" else ops.cfg.d),
                                expect_e_dim=1,
                                dtype=z_t.dtype,
                            )
    # predict
    with torch.cuda.amp.autocast(enabled=cfg.use_amp):
        v_z_pred, v_m_raw = vf_model(z_t, t, cond_vec=cond_vec)

        # project to tangent at current point
        v_m_pred = ops.proj_tangent(m_t, v_m_raw)

        # losses
        loss_z = F.mse_loss(v_z_pred, v_z_star)
        # manifold loss in metric: mean ||v_pred - v_star||^2_g
        dv = v_m_pred - v_m_star
        loss_m = ops.metric_norm2(m_t, dv).mean()

        loss = loss_z + cfg.lambda_m * loss_m

    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "loss_z": float(loss_z.detach().cpu()),
        "loss_m": float(loss_m.detach().cpu()),
    }
