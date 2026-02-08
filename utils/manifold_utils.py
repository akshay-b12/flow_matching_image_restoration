import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def safe_acos(x, eps=1e-7):
    return torch.acos(torch.clamp(x, -1 + eps, 1 - eps))

def slerp(s0, s1, t, eps=1e-7):
    """
    s0, s1: [B, d] unit vectors
    t: [B, 1] in [0,1]
    returns st: [B, d] unit vectors
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

def make_identity(B, d_s, d_e, device):
    s0 = torch.zeros(B, d_s, device=device)
    s0[:, 0] = 1.0
    e0 = torch.zeros(B, d_e, device=device)
    return s0, e0
