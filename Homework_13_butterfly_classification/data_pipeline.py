from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _natural_sort_key(path_like: str | Path):
    file_name = Path(path_like).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", file_name)]


def build_transforms(image_size: int = IMAGE_SIZE):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, valid_transform


class ButterflyDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, cache_in_memory: bool = True):
        self.samples: list[tuple[str, int]] = []
        self.transform = transform
        self.cache_in_memory = cache_in_memory

        root_path = Path(root_dir)
        for class_dir in sorted(root_path.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = int(class_dir.name.rsplit("_", 1)[-1])
            for file_path in sorted(class_dir.iterdir()):
                if file_path.is_file():
                    self.samples.append((str(file_path), class_name))

        self.labels = np.array([label for _, label in self.samples], dtype=np.int64)
        self._cached_images: list[np.ndarray] | None = None
        if self.cache_in_memory:
            self._cached_images = self._preload_images()

    def __len__(self):
        return len(self.samples)

    def _preload_images(self) -> list[np.ndarray]:
        cached_images: list[np.ndarray] = []
        for file_path, _ in self.samples:
            with Image.open(file_path) as img:
                cached_images.append(np.array(img.convert("RGB"), dtype=np.uint8))
        return cached_images

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        cached_images = getattr(self, "_cached_images", None)
        if cached_images is not None:
            image = Image.fromarray(cached_images[idx], mode="RGB")
        else:
            with Image.open(file_path) as img:
                image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TestButterflyDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, cache_in_memory: bool = True):
        self.transform = transform
        root_path = Path(root_dir)
        self.files = sorted(
            [str(path) for path in root_path.iterdir() if path.is_file()],
            key=_natural_sort_key,
        )
        self.cache_in_memory = cache_in_memory
        self._cached_images: list[np.ndarray] | None = None
        if self.cache_in_memory:
            self._cached_images = self._preload_images()

    def __len__(self):
        return len(self.files)

    def _preload_images(self) -> list[np.ndarray]:
        cached_images: list[np.ndarray] = []
        for file_path in self.files:
            with Image.open(file_path) as img:
                cached_images.append(np.array(img.convert("RGB"), dtype=np.uint8))
        return cached_images

    def __getitem__(self, idx):
        cached_images = getattr(self, "_cached_images", None)
        if cached_images is not None:
            image = Image.fromarray(cached_images[idx], mode="RGB")
        else:
            with Image.open(self.files[idx]) as img:
                image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class CachedTransformDataset(Dataset):
    """
    Dataset wrapper that applies a transform to samples from a base dataset.

    If cache_transformed=True, transformed tensors are computed once and then
    reused in subsequent epochs (useful for deterministic val/test transforms).
    """

    def __init__(
        self,
        base_dataset: ButterflyDataset | TestButterflyDataset,
        transform,
        *,
        indices: Sequence[int] | None = None,
        include_labels: bool = True,
        cache_transformed: bool = False,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.indices = list(indices) if indices is not None else list(range(len(base_dataset)))
        self.include_labels = include_labels
        self.cache_transformed = cache_transformed
        self._cached_outputs: list[torch.Tensor] | None = None

        if self.cache_transformed:
            self._cached_outputs = [self._transform_at(i) for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def _load_pil_image(self, source_idx: int):
        if isinstance(self.base_dataset, ButterflyDataset):
            file_path, _ = self.base_dataset.samples[source_idx]
            cached_images = getattr(self.base_dataset, "_cached_images", None)
            if cached_images is not None:
                return Image.fromarray(cached_images[source_idx], mode="RGB")
            with Image.open(file_path) as img:
                return img.convert("RGB")

        cached_images = getattr(self.base_dataset, "_cached_images", None)
        if cached_images is not None:
            return Image.fromarray(cached_images[source_idx], mode="RGB")
        with Image.open(self.base_dataset.files[source_idx]) as img:
            return img.convert("RGB")

    def _transform_at(self, source_idx: int):
        image = self._load_pil_image(source_idx)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        source_idx = self.indices[idx]
        image = (
            self._cached_outputs[idx]
            if self._cached_outputs is not None
            else self._transform_at(source_idx)
        )

        if self.include_labels:
            return image, int(self.base_dataset.labels[source_idx])
        return image


def _stratified_split_indices(labels: np.ndarray, seed: int, train_ratio: float):
    indices_by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_by_class[int(label)].append(idx)

    rng = random.Random(seed)
    train_indices: list[int] = []
    valid_indices: list[int] = []

    for class_indices in indices_by_class.values():
        rng.shuffle(class_indices)
        split_idx = int(train_ratio * len(class_indices))
        train_indices.extend(class_indices[:split_idx])
        valid_indices.extend(class_indices[split_idx:])

    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)
    return train_indices, valid_indices


def _resolve_num_workers(
    device: torch.device,
    num_workers: int | None,
    *,
    cache_in_memory: bool,
) -> int:
    if num_workers is not None:
        return max(0, int(num_workers))
    if cache_in_memory and os.name == "nt":
        # On Windows, DataLoader uses spawn; workers can duplicate RAM cache.
        # Keep default at 0 to avoid multiplying cached dataset memory.
        return 0
    cpu_count = os.cpu_count() or 2
    if device.type == "cuda":
        return max(2, min(6, cpu_count // 2))
    return max(0, min(2, cpu_count // 4))


def build_dataloaders(
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
    image_size: int = IMAGE_SIZE,
    train_root: str = "data/train_split",
    test_root: str = "data/valid",
    train_ratio: float = 0.8,
    num_workers: int | None = None,
    cache_in_memory: bool = True,
    cache_eval_tensors: bool = True,
):
    train_transform, valid_transform = build_transforms(image_size=image_size)
    base_dataset = ButterflyDataset(
        train_root,
        transform=None,
        cache_in_memory=cache_in_memory,
    )
    num_classes = int(base_dataset.labels.max()) + 1

    train_indices, valid_indices = _stratified_split_indices(
        base_dataset.labels,
        seed=seed,
        train_ratio=train_ratio,
    )

    full_train_dataset = CachedTransformDataset(
        base_dataset,
        valid_transform,
        include_labels=True,
        cache_transformed=cache_eval_tensors,
    )
    train_dataset = CachedTransformDataset(
        base_dataset,
        train_transform,
        indices=train_indices,
        include_labels=True,
        cache_transformed=False,
    )
    valid_dataset = CachedTransformDataset(
        base_dataset,
        valid_transform,
        indices=valid_indices,
        include_labels=True,
        cache_transformed=cache_eval_tensors,
    )
    test_base_dataset = TestButterflyDataset(
        test_root,
        transform=None,
        cache_in_memory=cache_in_memory,
    )
    test_dataset = CachedTransformDataset(
        test_base_dataset,
        valid_transform,
        include_labels=False,
        cache_transformed=cache_eval_tensors,
    )

    loader_workers = _resolve_num_workers(
        device=device,
        num_workers=num_workers,
        cache_in_memory=cache_in_memory,
    )
    pin_memory = device.type == "cuda"

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": loader_workers,
        "pin_memory": pin_memory,
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    metadata = {
        "num_classes": num_classes,
        "train_len": len(train_dataset),
        "valid_len": len(valid_dataset),
        "test_len": len(test_dataset),
        "num_workers": loader_workers,
        "cache_in_memory": cache_in_memory,
        "cache_eval_tensors": cache_eval_tensors,
        "test_file_names": [Path(file_path).name for file_path in test_base_dataset.files],
    }

    return full_train_dataset, train_loader, valid_loader, test_loader, metadata
