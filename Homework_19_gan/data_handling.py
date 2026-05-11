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
        load_size = self.img_size * 2
        data: list[PIL.Image.Image] = []
        files = [
            f
            for f in sorted(os.listdir(self.data_path))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        for file in tqdm(files, desc="Loading images"):
            image = PIL.Image.open(os.path.join(self.data_path, file))
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
