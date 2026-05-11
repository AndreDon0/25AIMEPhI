# powered by cursor

"""DCGAN-style Generator and Discriminator.

Why DCGAN and not a pure-MLP "vanilla" GAN?

A vanilla GAN (the one in 39_GAN.ipynb) is a stack of nn.Linear layers
that maps a latent vector z ∈ R^latent_dim to a flat vector of pixels.
For 28x28 grayscale MNIST that is fine: the output dim is 784 and the
spatial structure is so weak that an MLP can memorize it.

For natural RGB images (here 3xHxW) two things break down:

1) Capacity blows up.  A single nn.Linear(1024, 3*224*224) has
   ~154M parameters - one matrix in one layer.  Training that with
   ~140 images and a BCE adversarial signal is hopeless.
2) An MLP has no spatial inductive bias.  Pixel (0,0) and pixel (0,1)
   are no "closer" to it than pixel (0,0) and pixel (223,223).  Real
   images are dominated by local structure (edges, textures), which is
   exactly what convolutions exploit.

A DCGAN [Radford et al., 2015, https://arxiv.org/abs/1511.06434]
fixes both issues:

* Generator: project z to a small spatial map (e.g. 4x4) and upsample
  with strided ConvTranspose2d layers, doubling the spatial size each
  step until the target resolution.
* Discriminator: a strided-Conv2d classifier that halves the spatial
  size each step until 1x1.
* BatchNorm in both nets except on the generator's output and the
  discriminator's input/output layers.
* LeakyReLU(0.2) in the discriminator, ReLU in the generator.
* Output activation is Tanh, so images live in [-1, 1] (matched by the
  Normalize(0.5, 0.5) in data_handling.py).
* Weights initialized from N(0, 0.02), the canonical DCGAN init.

The architecture below is parameterized by `img_size`, which must be a
power of two >= 4.  For each doubling of `img_size` we add one extra
up/downsampling block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _num_upsample_blocks(img_size: int) -> int:
    """Number of x2 up/down blocks needed to reach ``img_size`` from 4x4."""
    if img_size < 4 or (img_size & (img_size - 1)) != 0:
        raise ValueError(
            f"img_size must be a power of two and >= 4, got {img_size}"
        )
    return int(math.log2(img_size)) - 2


class Generator(nn.Module):
    """DCGAN generator: latent z -> (channels, img_size, img_size)."""

    def __init__(
        self,
        img_shape: tuple[int, int, int],
        latent_dim: int = 100,
        base_channels: int = 64,
    ):
        super().__init__()
        channels, height, width = img_shape
        if height != width:
            raise ValueError(f"DCGAN expects square images, got {img_shape}")

        self.img_shape = img_shape
        self.latent_dim = latent_dim

        n_blocks = _num_upsample_blocks(height)
        # Channel widths shrink as resolution grows: e.g. for 64x64 we get
        # [512, 256, 128, 64] starting from a 4x4 feature map.
        widths = [base_channels * (2 ** i) for i in reversed(range(n_blocks))]

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                latent_dim, widths[0], kernel_size=4, stride=1, padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        ]

        for in_c, out_c in zip(widths[:-1], widths[1:]):
            layers += [
                nn.ConvTranspose2d(
                    in_c, out_c, kernel_size=4, stride=2, padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]

        layers += [
            nn.ConvTranspose2d(
                widths[-1], channels, kernel_size=4, stride=2, padding=1,
                bias=False,
            ),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

        self.apply(_dcgan_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # The conv stack expects (B, C, H, W); promote 2D z -> 4D feature map.
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)


class Discriminator(nn.Module):
    """DCGAN discriminator: image -> single logit per sample (no Sigmoid).

    The Sigmoid is intentionally omitted so that we can train with
    nn.BCEWithLogitsLoss, which is numerically stabler than the
    Sigmoid + BCELoss combination used in the original vanilla GAN.
    """

    def __init__(
        self,
        img_shape: tuple[int, int, int],
        base_channels: int = 64,
    ):
        super().__init__()
        channels, height, width = img_shape
        if height != width:
            raise ValueError(f"DCGAN expects square images, got {img_shape}")

        self.img_shape = img_shape

        n_blocks = _num_upsample_blocks(height)
        widths = [base_channels * (2 ** i) for i in range(n_blocks)]

        layers: list[nn.Module] = [
            nn.Conv2d(
                channels, widths[0], kernel_size=4, stride=2, padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for in_c, out_c in zip(widths[:-1], widths[1:]):
            layers += [
                nn.Conv2d(
                    in_c, out_c, kernel_size=4, stride=2, padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Final 4x4 -> 1x1 conv collapses spatial dims to a per-image logit.
        layers += [
            nn.Conv2d(widths[-1], 1, kernel_size=4, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

        self.apply(_dcgan_weights_init)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        logits = self.model(img)
        return logits.view(img.size(0), 1)


def _dcgan_weights_init(m: nn.Module) -> None:
    """Canonical DCGAN initialization: N(0, 0.02) for conv weights and BN."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
