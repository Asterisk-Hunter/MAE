"""CIFAR-10 data loading for MAE pretraining and the demo."""
from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

NATIVE_SIZE = 32


def _transform(img_size: int) -> transforms.Compose:
    ops: list = []
    if img_size != NATIVE_SIZE:
        ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC,
                                     antialias=True))
    ops.append(transforms.ToTensor())  # CIFAR-10 is uint8 -> [0, 1]
    return transforms.Compose(ops)


def build_dataset(root: str = "data", train: bool = True, img_size: int = NATIVE_SIZE,
                  download: bool = True) -> CIFAR10:
    """CIFAR-10 with images as float tensors in [0, 1] (no mean/std normalization,
    so the MAE pixel loss and the demo share one consistent pixel space)."""
    return CIFAR10(root=root, train=train, download=download, transform=_transform(img_size))


def build_loader(root: str = "data", img_size: int = NATIVE_SIZE, batch_size: int = 256,
                 num_workers: int = 4, download: bool = True) -> DataLoader:
    ds = build_dataset(root, train=True, img_size=img_size, download=download)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
