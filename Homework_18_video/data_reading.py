import torch
import pandas as pd
import os
from torch.utils.data import Dataset
import av
import numpy as np

CLIP_NUM_FRAMES = 16
CROP_SIZE = 224
RESIZE_SIZE = 256

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)


def _videomae_pixel_values(video_hwc_uint8: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> float tensor (T, 3, H, W) rescaled and ImageNet-normalized."""
    x = video_hwc_uint8.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(x))


def read_all_frames_rgb(
    path: str,
    width: int = RESIZE_SIZE,
    height: int = RESIZE_SIZE,
) -> np.ndarray:
    """Decode **all** video frames at (height, width), return (T, H, W, 3) uint8 RGB.

    Temporal sampling is NOT done here — that's the transform's job.
    Results are cached as .npy files so repeated epochs skip the PyAV decode.
    """
    basename = os.path.splitext(os.path.basename(path))[0]
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(path)), ".video_cache")
    cache_file = os.path.join(cache_dir, f"{basename}_{width}x{height}_all.npy")

    if os.path.isfile(cache_file):
        return np.load(cache_file)

    frames = []
    with av.open(path, metadata_errors="replace") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frame = frame.reformat(width=width, height=height, format="rgb24")
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")

    video = np.stack(frames, axis=0)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_file, video)
    return video


def process_label_csv(label_path: str) -> pd.DataFrame:
    """Read train.csv and return a DataFrame with 'path' + one-hot label columns."""
    df = pd.read_csv(label_path)
    df["labels"] = df["labels"].str.replace(".", ",")
    dummies = df["labels"].str.get_dummies(sep=", ")
    return pd.concat([df, dummies], axis=1).drop(columns=["labels"])


class TrainDataset(Dataset):
    """Supports two construction modes:

    1. ``TrainDataset(data_path, label_path="train.csv", transform=...)``
       — reads and processes the CSV internally.
    2. ``TrainDataset(data_path, labels_df=pre_split_df, transform=...)``
       — uses an already-processed DataFrame (for train/val splits).
    """

    def __init__(
        self,
        data_path: str,
        label_path: str | None = None,
        labels_df: pd.DataFrame | None = None,
        transform=None,
    ):
        if labels_df is not None:
            self.labels = labels_df.reset_index(drop=True)
        elif label_path is not None:
            self.labels = process_label_csv(label_path)
        else:
            raise ValueError("Provide either label_path or labels_df")
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        video_path = os.path.join(self.data_path, self.labels["path"][index])
        video = read_all_frames_rgb(video_path)
        if self.transform is not None:
            video = self.transform(video)
        pixel_values = _videomae_pixel_values(video)
        label = torch.tensor(
            self.labels.iloc[index].values.tolist()[1:], dtype=torch.float32
        )
        return pixel_values, label

    def get_classes(self):
        return self.labels.columns.tolist()[1:]


class TestDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.labels = [p for p in os.listdir(data_path) if p.endswith(".mp4")]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        video_path = os.path.join(self.data_path, self.labels[index])
        video = read_all_frames_rgb(video_path)
        if self.transform is not None:
            video = self.transform(video)
        return _videomae_pixel_values(video)
