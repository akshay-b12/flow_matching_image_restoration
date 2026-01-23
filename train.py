import argparse
import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import yaml
from tqdm import tqdm

from diffusers import AutoencoderKL

from models.tinyunet import TinyUNet
from data.paireddataset import ImagePairDataset

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for name in os.listdir(root):
        if os.path.splitext(name)[1].lower() in exts:
            files.append(name)
    files.sort()
    return files

@dataclass
class TrainState:
    step: int
    epoch: int


def save_checkpoint(path, model, optimizer, state):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": state.step,
        "epoch": state.epoch,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return TrainState(step=payload.get("step", 0), epoch=payload.get("epoch", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["train"].get("seed", 42))

    dataset = ImagePairDataset(
        clean_dir=cfg["dataset"]["clean_dir"],
        degraded_dir=cfg["dataset"].get("degraded_dir"),
        image_size=cfg["dataset"].get("image_size", 256),
        center_crop=cfg["dataset"].get("center_crop", True),
        degrade_cfg=cfg["dataset"].get("degrade_on_the_fly", {}),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"].get("batch_size", 4),
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    vae = AutoencoderKL.from_pretrained(
        cfg["vae"]["pretrained_path"],
        subfolder=cfg["vae"].get("subfolder"),
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

    model = TinyUNet(
        in_channels=cfg["model"].get("in_channels", 4),
        base_channels=cfg["model"].get("base_channels", 64),
        time_emb_dim=cfg["model"].get("time_emb_dim", 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-2),
    )

    use_amp = cfg["train"].get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    state = TrainState(step=0, epoch=0)
    resume = cfg["train"].get("resume")
    if resume and os.path.exists(resume):
        state = load_checkpoint(resume, model, optimizer)

    max_epochs = cfg["train"].get("epochs", 10)
    log_every = cfg["train"].get("log_every", 50)
    save_every = cfg["train"].get("save_every", 1)
    out_dir = cfg["train"].get("output_dir", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(state.epoch, max_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{max_epochs}", leave=True)
        for degraded, clean in pbar:
            degraded = degraded.to(device)
            clean = clean.to(device)

            with torch.no_grad():
                z_src = vae.encode(degraded).latent_dist.sample() * scaling_factor
                z_tgt = vae.encode(clean).latent_dist.sample() * scaling_factor

            t = torch.rand(z_src.size(0), device=device).view(-1, 1, 1, 1)
            x_t = (1.0 - t) * z_src + t * z_tgt
            v = z_tgt - z_src

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x_t, t.squeeze(-1).squeeze(-1))
                loss = nn.functional.mse_loss(pred, v)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if state.step % log_every == 0:
                pbar.set_postfix(step=state.step, loss=f"{loss.item():.6f}")

            state.step += 1

        state.epoch = epoch + 1
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(out_dir, f"flowmatch_epoch_{epoch+1}.pt")
            save_checkpoint(ckpt_path, model, optimizer, state)


if __name__ == "__main__":
    main()
