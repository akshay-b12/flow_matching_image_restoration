import torch

@torch.no_grad()
def encode_latents(vae, x, deterministic=False):
    sf = float(vae.config.scaling_factor)
    post = vae.encode(x).latent_dist
    z = post.mean if deterministic else post.sample()
    return z * sf  # scaled latents

@torch.no_grad()
def decode_latents(vae, z):
    sf = float(vae.config.scaling_factor)
    return vae.decode(z / sf).sample