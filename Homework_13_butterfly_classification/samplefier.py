import random
from typing import Callable

import cv2
import numpy as np


class Samplefier:
    """Generate stochastic OpenCV-based augmentations for one input image."""

    def __init__(self, seed: int, sample_rate: float) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be in [0, 1]")
        self.sample_rate = float(sample_rate)
        self._rng = random.Random(seed)

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        """Return sampled augmented copies while preserving image shape."""
        if not self._is_valid_image(image):
            return []

        transforms: list[Callable[[np.ndarray], np.ndarray]] = [
            self._horizontal_flip,
            self._rotate_90_cw,
            self._rotate_90_ccw,
            self._rotate_180,
            self._center_crop_resize,
            self._scale_down_resize,
            self._scale_up_resize,
            self._blur_3x3,
            self._blur_7x7,
            self._gaussian_blur,
            self._sharpen,
            self._erode,
            self._dilate,
            self._morph_open,
            self._morph_close,
        ]

        h, w = image.shape[:2]
        samples: list[np.ndarray] = []
        for transform in transforms:
            if self._rng.random() >= self.sample_rate:
                continue
            transformed = transform(image)
            if transformed.shape[:2] != (h, w):
                transformed = cv2.resize(transformed, (w, h), interpolation=cv2.INTER_LINEAR)
            if transformed.shape == image.shape:
                samples.append(transformed)
        return samples

    @staticmethod
    def _is_valid_image(image: np.ndarray) -> bool:
        return isinstance(image, np.ndarray) and image.ndim in (2, 3) and image.size > 0

    @staticmethod
    def _horizontal_flip(image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)

    @staticmethod
    def _rotate_90_cw(image: np.ndarray) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    @staticmethod
    def _rotate_90_ccw(image: np.ndarray) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @staticmethod
    def _rotate_180(image: np.ndarray) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_180)

    @staticmethod
    def _center_crop_resize(image: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
        h, w = image.shape[:2]
        crop_h = max(1, int(h * crop_ratio))
        crop_w = max(1, int(w * crop_ratio))
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        cropped = image[start_y : start_y + crop_h, start_x : start_x + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _scale_down_resize(image: np.ndarray, scale: float = 0.75) -> np.ndarray:
        h, w = image.shape[:2]
        down_w = max(1, int(w * scale))
        down_h = max(1, int(h * scale))
        down = cv2.resize(image, (down_w, down_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _scale_up_resize(image: np.ndarray, scale: float = 1.25) -> np.ndarray:
        h, w = image.shape[:2]
        up_w = max(w + 1, int(w * scale))
        up_h = max(h + 1, int(h * scale))
        up = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
        start_y = (up_h - h) // 2
        start_x = (up_w - w) // 2
        return up[start_y : start_y + h, start_x : start_x + w]

    @staticmethod
    def _blur_3x3(image: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.float32) / 9.0
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def _blur_7x7(image: np.ndarray) -> np.ndarray:
        kernel = np.ones((7, 7), np.float32) / 49.0
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def _gaussian_blur(image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)

    @staticmethod
    def _sharpen(image: np.ndarray) -> np.ndarray:
        kernel = np.array(
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
            dtype=np.float32,
        )
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def _erode(image: np.ndarray) -> np.ndarray:
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    @staticmethod
    def _dilate(image: np.ndarray) -> np.ndarray:
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    @staticmethod
    def _morph_open(image: np.ndarray) -> np.ndarray:
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def _morph_close(image: np.ndarray) -> np.ndarray:
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)