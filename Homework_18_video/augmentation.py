import numpy as np
import random

CLIP_NUM_FRAMES = 16
CROP_SIZE = 224


def sample_frames(video: np.ndarray, num_frames: int, *, jitter: bool = False) -> np.ndarray:
    """Temporally sample *num_frames* from video ``(T, H, W, 3)``.

    With ``jitter=True`` the timeline is split into *num_frames* equal bins and one
    random frame is drawn from each bin — every epoch sees a slightly different clip.
    With ``jitter=False`` frames are spaced uniformly (deterministic).
    Short videos are padded by repeating the last frame.
    """
    t = video.shape[0]
    if t <= num_frames:
        if t < num_frames:
            pad = np.repeat(video[-1:], num_frames - t, axis=0)
            video = np.concatenate([video, pad], axis=0)
        return video

    if jitter:
        stride = t / num_frames
        indices = [
            random.randint(int(i * stride), min(int((i + 1) * stride), t) - 1)
            for i in range(num_frames)
        ]
        return video[np.array(indices)]

    return video[np.linspace(0, t - 1, num_frames, dtype=int)]


def _random_crop(video: np.ndarray, crop_size: int) -> np.ndarray:
    _, h, w, _ = video.shape
    if h == crop_size and w == crop_size:
        return video
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return video[:, top : top + crop_size, left : left + crop_size, :]


def _center_crop(video: np.ndarray, crop_size: int) -> np.ndarray:
    _, h, w, _ = video.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return video[:, top : top + crop_size, left : left + crop_size, :]


def _random_horizontal_flip(video: np.ndarray, p: float = 0.5) -> np.ndarray:
    if random.random() < p:
        return np.ascontiguousarray(np.flip(video, axis=2))
    return video


def _color_jitter(
    video: np.ndarray, brightness: float = 0.3, contrast: float = 0.2
) -> np.ndarray:
    x = video.astype(np.float32)
    b = random.uniform(max(0.0, 1 - brightness), 1 + brightness)
    x = x * b
    c = random.uniform(max(0.0, 1 - contrast), 1 + contrast)
    mean = x.mean(axis=(0, 1, 2), keepdims=True)
    x = (x - mean) * c + mean
    return np.clip(x, 0, 255).astype(np.uint8)


class VideoTrainTransform:
    """Training augmentation: temporal jitter -> random crop -> flip -> color jitter.

    Input:  all decoded frames ``(T_all, H_resize, W_resize, 3)`` uint8.
    Output: augmented clip ``(num_frames, crop_size, crop_size, 3)`` uint8.
    """

    def __init__(
        self,
        num_frames: int = CLIP_NUM_FRAMES,
        crop_size: int = CROP_SIZE,
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
    ):
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter

    def __call__(self, video: np.ndarray) -> np.ndarray:
        video = sample_frames(video, self.num_frames, jitter=True)
        video = _random_crop(video, self.crop_size)
        video = _random_horizontal_flip(video, self.hflip_prob)
        if self.color_jitter:
            video = _color_jitter(video)
        return video


class VideoTestTransform:
    """Deterministic transform for val / test: uniform sampling -> center crop.

    Input:  all decoded frames ``(T_all, H_resize, W_resize, 3)`` uint8.
    Output: clip ``(num_frames, crop_size, crop_size, 3)`` uint8.
    """

    def __init__(
        self,
        num_frames: int = CLIP_NUM_FRAMES,
        crop_size: int = CROP_SIZE,
    ):
        self.num_frames = num_frames
        self.crop_size = crop_size

    def __call__(self, video: np.ndarray) -> np.ndarray:
        video = sample_frames(video, self.num_frames, jitter=False)
        video = _center_crop(video, self.crop_size)
        return video
