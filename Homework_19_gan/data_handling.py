"""Tiny image-folder dataset + DataLoader factory for the GAN homework.

The dataset has only ~141 photos, so we lean fairly hard on augmentation
to give the GAN more "effective" training samples per epoch.  Heavy
augmentation in GANs is a real research direction (see DiffAugment,
Karras et al. 2020 ADA), but for our purposes a modest set of crops,
flips and color jitter is a big improvement over plain resize.

Notes on the pipeline:

* We pre-load all images into memory and resize them once on disk read.
  At 64x64 / 141 images that's <2 MB total in PIL form, which is fine.
* Random transforms are applied in __getitem__, so each epoch sees a
  fresh randomized version of every image.
* The Normalize(0.5, 0.5) maps pixel values from [0, 1] (after ToTensor)
  to roughly [-1, 1], which lines up with the Tanh output of the
  generator.  Both nets must agree on this convention.
"""

from __future__ import annotations

import os

import PIL
from PIL import ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class CustomDataset(Dataset):
    def __init__(self, data_path: str, img_size: int, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.img_size = img_size
        self.data = self._load_data()

    def _load_data(self) -> list[PIL.Image.Image]:
        # Resize on load to ~2x the target size so RandomResizedCrop has
        # something to actually crop from.
        load_size = self.img_size * 2
        data: list[PIL.Image.Image] = []
        files = [
            f
            for f in sorted(os.listdir(self.data_path))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        for file in tqdm(files, desc="Loading images"):
            image = PIL.Image.open(os.path.join(self.data_path, file))
            # Phone cameras (like the Samsung Galaxy that produced these
            # files) store the raw landscape pixel buffer and record the
            # actual capture orientation in the EXIF "Orientation" tag.
            # PIL.Image.open does NOT honor that tag automatically, so a
            # portrait photo loads as a sideways landscape image.
            # ImageOps.exif_transpose reads the tag, applies the matching
            # rotation/flip, and clears the tag so downstream code can
            # treat the image as plain top-left-origin pixels.
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((load_size, load_size))
            data.append(image)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        image = self.data[idx]
        if self.transform is not None:
            return self.transform(image)
        return image


class DataHandler:
    def __init__(self, data_path: str, img_size: int, batch_size: int):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def get_loader(self) -> DataLoader:
        dataset = CustomDataset(self.data_path, self.img_size, self.transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
