import argparse
import os

import torch
from torchvision import transforms
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

    degraded = load_image(
        args.input,
        image_size=cfg["dataset"].get("image_size", 256),
        center_crop=cfg["dataset"].get("center_crop", True),
    ).to(device)

    with torch.no_grad():
        z = vae.encode(degraded).latent_dist.sample() * scaling_factor
        dt = 1.0 / args.steps
        for i in range(args.steps):
            t = torch.full((z.size(0),), i * dt, device=device)
            v = model(z, t)
            z = z + v * dt
        rec = vae.decode(z / scaling_factor).sample

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_image(rec, args.output)


if __name__ == "__main__":
    main()
