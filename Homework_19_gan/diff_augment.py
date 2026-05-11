# powered by cursor

"""DiffAugment for GAN training in low-data regimes.

Reference:
    Zhao, Liu, Lin, Zhu, Han.
    "Differentiable Augmentation for Data-Efficient GAN Training."
    NeurIPS 2020.  https://arxiv.org/abs/2006.10738
    Original repo: https://github.com/mit-han-lab/data-efficient-gans

Why this matters here
---------------------
With only ~140 training images, the discriminator can simply *memorize*
the training set: it then provides no useful gradient to the generator,
and G collapses or produces garbage.  This is the classic low-data GAN
failure mode.

DiffAugment fixes this by applying *differentiable* augmentations T(.)
to BOTH the real and the fake batch right before the discriminator:

    L_D = E[ log D(T(x_real)) ] + E[ log(1 - D(T(G(z)))) ]
    L_G = E[ log D(T(G(z))) ]

Two crucial properties:

1. The same augmentation distribution is applied to real and fake.
   That means D cannot distinguish images by their augmentation marks;
   it has to compare *content*, exactly what we want.
2. The augmentations are differentiable, so gradients flow through
   T into G.  G implicitly learns to produce images that look real
   *under random color/translation/cutout perturbations*, which is a
   much stronger requirement than "look real" and forces it to
   generalize beyond memorizing single pixels.

Critically, this is NOT the same as augmenting the dataset on disk:
- Disk-level augmentation only changes what the dataset distribution
  looks like.  D can still memorize that augmented distribution.
- DiffAugment applies the *same* augmentation pipeline to G's output,
  so D sees real and fake under matched conditions, and G is
  responsible for being robust to the perturbation.

The augmentation operators below assume images in [-1, 1] (which is
what our Tanh + Normalize(0.5, 0.5) pipeline produces).  This is the
range the DiffAugment paper was tuned for.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diff_augment(x: torch.Tensor, policy: str = "") -> torch.Tensor:
    """Apply a comma-separated list of DiffAugment policies to ``x``.

    Args:
        x: NCHW image tensor in approximately [-1, 1].
        policy: e.g. "color,translation,cutout".  Empty string is a no-op.
    """
    if not policy:
        return x
    for name in policy.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in AUGMENT_FNS:
            raise ValueError(
                f"Unknown DiffAugment policy '{name}'. "
                f"Valid options: {sorted(AUGMENT_FNS)}"
            )
        for fn in AUGMENT_FNS[name]:
            x = fn(x)
    return x.contiguous()


# ---------------------------------------------------------------------------
# Color: per-image additive brightness, multiplicative saturation, and
# multiplicative contrast around the image mean.  All three are smooth
# w.r.t. the input tensor, so gradients flow back to G.
# ---------------------------------------------------------------------------

def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    shift = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    return x + shift


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=1, keepdim=True)  # mean over channels
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2.0
    return (x - x_mean) * factor + x_mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)  # mean over CHW
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - x_mean) * factor + x_mean


# ---------------------------------------------------------------------------
# Translation: independent per-image integer pixel shifts in [-r*H, r*H]
# and [-r*W, r*W], implemented as zero-padded gather to stay differentiable.
# ---------------------------------------------------------------------------

def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    n, c, h, w = x.shape
    shift_h, shift_w = int(h * ratio + 0.5), int(w * ratio + 0.5)

    translation_h = torch.randint(
        -shift_h, shift_h + 1, size=(n, 1, 1), device=x.device
    )
    translation_w = torch.randint(
        -shift_w, shift_w + 1, size=(n, 1, 1), device=x.device
    )

    grid_n, grid_h, grid_w = torch.meshgrid(
        torch.arange(n, dtype=torch.long, device=x.device),
        torch.arange(h, dtype=torch.long, device=x.device),
        torch.arange(w, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    # +1 because we pad x by 1 on each side so out-of-bounds shifts read zero.
    grid_h = torch.clamp(grid_h + translation_h + 1, 0, h + 1)
    grid_w = torch.clamp(grid_w + translation_w + 1, 0, w + 1)

    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    out = (
        x_pad.permute(0, 2, 3, 1).contiguous()[grid_n, grid_h, grid_w]
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return out


# ---------------------------------------------------------------------------
# Cutout: zero out a random rectangular region per image.
# ---------------------------------------------------------------------------

def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    n, c, h, w = x.shape
    cutout_h, cutout_w = int(h * ratio + 0.5), int(w * ratio + 0.5)

    offset_h = torch.randint(
        0, h + (1 - cutout_h % 2), size=(n, 1, 1), device=x.device
    )
    offset_w = torch.randint(
        0, w + (1 - cutout_w % 2), size=(n, 1, 1), device=x.device
    )

    grid_n, grid_h, grid_w = torch.meshgrid(
        torch.arange(n, dtype=torch.long, device=x.device),
        torch.arange(cutout_h, dtype=torch.long, device=x.device),
        torch.arange(cutout_w, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_h = torch.clamp(grid_h + offset_h - cutout_h // 2, 0, h - 1)
    grid_w = torch.clamp(grid_w + offset_w - cutout_w // 2, 0, w - 1)

    mask = torch.ones(n, h, w, dtype=x.dtype, device=x.device)
    mask[grid_n, grid_h, grid_w] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
