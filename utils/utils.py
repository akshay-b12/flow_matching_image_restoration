import torch

def to_minus1_1(x):  # if x in [0,1]
    return x * 2.0 - 1.0

def sample_t(B, device, eps=1e-3):
    return torch.rand(B, device=device) * (1 - 2*eps) + eps