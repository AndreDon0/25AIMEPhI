import torch
from torch.utils.data import Dataset
from dataprovider import RandomRgbDataProvider
import numpy as np


class DataGenerator:
    def __init__(self):
        self.data = None
        self.labels = None

    def generate_data(self, x_size: int, y_size: int, samples: int, **kwargs):
        data_provider = RandomRgbDataProvider(x_size, y_size, **kwargs)
        self.data, self.labels = data_provider(samples)
        self.data = self.data.astype(np.uint8)
        self.labels = self.labels.astype(bool)
    
    def save_data(self, path: str):
        if self.data is None or self.labels is None:
            raise ValueError("Data and labels are not generated. Call generate_data method first.")
        np.savez_compressed(path, data=self.data, labels=self.labels)
        print(f"Data saved to {path}")

    def load_data(self, path: str):
        self.data = np.load(path)["data"]
        self.labels = np.load(path)["labels"]
        print(f"Data loaded from {path}")


class TrainDataset(Dataset):
    def __init__(self, path: str):
        loaded = np.load(path)
        self.data = loaded["data"]
        labels = loaded["labels"]
        self.labels_idx = np.argmax(labels, axis=-1).astype(np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels_idx[idx]
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).to(torch.float32) / 255.0
        y = torch.from_numpy(np.ascontiguousarray(y))
        return x, y


class TestDataset(Dataset):
    def __init__(self, path: str):
        self.data = np.load(path)["data"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).to(torch.float32) / 255.0
        return x