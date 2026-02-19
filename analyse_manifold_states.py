# Example usage

device = "cuda"

# Load your models
vae.eval().to(device)

# Sphere models
deg_enc_sphere.eval().to(device)   # outputs (s1,e1)
model_sphere.eval().to(device)     # expects (x_t, t_flat, s_t, e_t)

# Hyperbolic models
deg_enc_hyp.eval().to(device)      # outputs (u,e1)
model_hyp.eval().to(device)        # expects (x_t, t_flat, u_t, e_t)

traj = collect_trajs(
    dataloader=test_loader,
    vae=vae,
    scaling_factor=scaling_factor,
    deg_enc_sphere=deg_enc_sphere,
    model_sphere=model_sphere,
    d_s=8,
    d_e=1,
    deg_enc_hyp=deg_enc_hyp,
    model_hyp=model_hyp,
    d_h=8,
    c=1.0,
    T=11,                 # t grid points
    max_batches=20,        # increase for better stats
    device=device
)

results = run_analysis(traj, outdir="./manifold_analysis", c_hyp=1.0)
print(results)
