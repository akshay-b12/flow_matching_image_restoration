import argparse
import os

import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import yaml

from diffusers import AutoencoderKL

from models.tinyunet import TinyUNet


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_image(path, image_size, center_crop):
    img = Image.open(path).convert("RGB")
    resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC)
    crop = transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size)
    base = transforms.Compose(
        [
            resize,
            crop,
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    return base(img).unsqueeze(0)


def load_full_image(path):
    img = Image.open(path).convert("RGB")
    base = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    return base(img).unsqueeze(0)


def save_image(tensor, path):
    img = (tensor.clamp(-1, 1) + 1.0) * 0.5
    img = img.squeeze(0).detach().cpu()
    img = transforms.ToPILImage()(img)
    img.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--patch_size", type=int, default=512)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    vae = AutoencoderKL.from_pretrained(
        cfg["vae"]["pretrained_path"],
        subfolder=cfg["vae"].get("subfolder"),
    ).to(device)
    vae.eval()
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

    model = TinyUNet(
        in_channels=cfg["model"].get("in_channels", 4),
        base_channels=cfg["model"].get("base_channels", 64),
        time_emb_dim=cfg["model"].get("time_emb_dim", 128),
    ).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()

    if args.patch_size % 8 != 0:
        raise ValueError("patch_size must be divisible by 8 for the VAE downsampling.")

    degraded = load_full_image(args.input).to(device)
    _, _, h, w = degraded.shape
    pad_h = (args.patch_size - h % args.patch_size) % args.patch_size
    pad_w = (args.patch_size - w % args.patch_size) % args.patch_size
    degraded = F.pad(degraded, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, hp, wp = degraded.shape
    output = torch.zeros_like(degraded)

    with torch.no_grad():
        dt = 1.0 / args.steps
        for y in range(0, hp, args.patch_size):
            for x in range(0, wp, args.patch_size):
                patch = degraded[:, :, y : y + args.patch_size, x : x + args.patch_size]
                z = vae.encode(patch).latent_dist.sample() * scaling_factor
                for i in range(args.steps):
                    t = torch.full((z.size(0),), i * dt, device=device)
                    v = model(z, t)
                    z = z + v * dt
                rec = vae.decode(z / scaling_factor).sample
                output[:, :, y : y + args.patch_size, x : x + args.patch_size] = rec

    rec = output[:, :, :h, :w]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_image(rec, args.output)


if __name__ == "__main__":
    main()
