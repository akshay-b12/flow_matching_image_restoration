import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

class ImagePairDataset(Dataset):
    def __init__(self, clean_dir, degraded_dir, image_size, center_crop, degrade_cfg):
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.clean_files = list_images(clean_dir)
        if degraded_dir:
            self.degraded_files = list_images(degraded_dir)
            self.degraded_map = {n: n for n in self.degraded_files}
        else:
            self.degraded_files = None
            self.degraded_map = None

        resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC)
        crop = transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size)
        self.base_transform = transforms.Compose(
            [
                resize,
                crop,
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )

        self.degrade_cfg = degrade_cfg or {}
        self.blur = transforms.GaussianBlur(
            kernel_size=self.degrade_cfg.get("blur_kernel", 5),
            sigma=self.degrade_cfg.get("blur_sigma", (0.1, 2.0)),
        )

    def __len__(self):
        return len(self.clean_files)

    def _load_image(self, root, name):
        path = os.path.join(root, name)
        img = Image.open(path).convert("RGB")
        return img

    def _degrade(self, clean_img):
        img = clean_img
        if self.degrade_cfg.get("use_blur", True):
            img = self.blur(img)
        degraded = self.base_transform(img)
        if self.degrade_cfg.get("use_noise", True):
            noise_std = self.degrade_cfg.get("noise_std", 0.05)
            noise = torch.randn_like(degraded) * noise_std
            degraded = (degraded + noise).clamp(-1.0, 1.0)
        return degraded

    def __getitem__(self, idx):
        clean_name = self.clean_files[idx]
        clean_img = self._load_image(self.clean_dir, clean_name)
        clean = self.base_transform(clean_img)

        if self.degraded_dir:
            degraded_name = self.degraded_map.get(clean_name)
            if degraded_name is None:
                raise FileNotFoundError(f"Missing degraded image for {clean_name}")
            degraded_img = self._load_image(self.degraded_dir, degraded_name)
            degraded = self.base_transform(degraded_img)
        else:
            degraded = self._degrade(clean_img)

        return degraded, clean
