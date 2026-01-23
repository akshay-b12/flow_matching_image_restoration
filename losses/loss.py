def metric_weighted_loss(uhat, ut, w):
    # w: [B,C,H,W] positive
    err2 = (uhat - ut) ** 2
    return (w * err2).mean()

def total_variation(x):
    # x: [B,C,H,W]
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w
