import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import av
import numpy as np

TARGET_VIDEO_WIDTH = 224
TARGET_VIDEO_HEIGHT = 224
CLIP_NUM_FRAMES = 16

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)


def _videomae_pixel_values(video_hwc_uint8: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 → float tensor (T, 3, H, W) rescaled and ImageNet-normalized."""
    x = video_hwc_uint8.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(x))


def read_all_frames_rgb(
    path: str,
    width: int = TARGET_VIDEO_WIDTH,
    height: int = TARGET_VIDEO_HEIGHT,
    num_frames: int = CLIP_NUM_FRAMES,
) -> np.ndarray:
    """Decode video, resize each frame to (height, width), return (num_frames, H, W, 3) uint8 RGB.

    Uniformly samples ``num_frames`` frames along the video timeline. Shorter videos are padded
    by repeating the last frame until ``num_frames`` is reached.
    """
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
    t = video.shape[0]
    if t < num_frames:
        pad = np.repeat(video[-1:], num_frames - t, axis=0)
        return np.concatenate([video, pad], axis=0)
    if t == num_frames:
        return video
    idx = np.linspace(0, t - 1, num_frames, dtype=int)
    return video[idx]


class TrainDataset(Dataset):
    def __init__(self, data_path: str, label_path: str, transform=None):
        self.labels = pd.read_csv(label_path)

        self.labels["labels"] = self.labels["labels"].str.replace(".", ",")
        dummies = self.labels['labels'].str.get_dummies(sep=', ')

        self.labels = pd.concat([self.labels, dummies], axis=1)
        self.labels.drop(columns=["labels"], inplace=True)
        self.data_path = data_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        video_path = os.path.join(self.data_path, self.labels['path'][index])
        video = read_all_frames_rgb(video_path)
        return _videomae_pixel_values(video), torch.tensor(self.labels.iloc[index].values.tolist()[1:], dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.labels = [path for path in os.listdir(data_path) if path.endswith('.mp4')]
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        video_path = os.path.join(self.data_path, self.labels[index])
        video = read_all_frames_rgb(video_path)
        return _videomae_pixel_values(video)