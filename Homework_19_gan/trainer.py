"""GAN trainer with a few standard stability tricks.

Compared to the textbook vanilla loop, this trainer adds:

* BCEWithLogitsLoss (the discriminator now returns logits, no Sigmoid)
  - sigmoid_cross_entropy_with_logits is numerically stabler than
    Sigmoid -> BCELoss when the logit becomes large/small.
* One-sided label smoothing: real labels are 0.9 instead of 1.0.
  This caps the discriminator's confidence on real data and prevents
  the "D collapses to 1.0 everywhere on reals" failure mode that is
  common with small datasets.  See Salimans et al. 2016, "Improved
  Techniques for Training GANs".
* A fixed evaluation noise tensor.  Tracking the same z over training
  shows whether the generator is actually moving or stuck/colla psed.
* Optional callback hook (`on_epoch_end`) so the notebook can render
  sample grids during training.
* Independent forward passes for G and D to keep the BatchNorm running
  stats sane (avoids reusing a fake batch generated under one set of G
  parameters when computing D's loss).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from diff_augment import diff_augment


class GANTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        adversarial_loss: nn.Module,
        device: torch.device | str,
        epochs: int = 10,
        latent_dim: Optional[int] = None,
        real_label_smoothing: float = 0.9,
        fixed_noise_size: int = 16,
        on_epoch_end: Optional[Callable[["GANTrainer", int], None]] = None,
        diff_augment_policy: str = "",
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.adversarial_loss = adversarial_loss
        self.device = device
        self.epochs = epochs
        self.latent_dim = latent_dim or self._infer_latent_dim()
        self.real_label_smoothing = real_label_smoothing
        self.on_epoch_end = on_epoch_end
        self.diff_augment_policy = diff_augment_policy

        self.generator_loss_history: list[float] = []
        self.discriminator_loss_history: list[float] = []

        # A frozen z that we re-feed every epoch so visual progress is
        # comparable across epochs (same "view" of latent space).
        self.fixed_noise = torch.randn(
            fixed_noise_size, self.latent_dim, device=self.device
        )

    def _infer_latent_dim(self) -> int:
        """Look up generator.latent_dim if available, else error out cleanly."""
        if hasattr(self.generator, "latent_dim"):
            return int(self.generator.latent_dim)
        raise ValueError(
            "latent_dim is not provided and could not be inferred. "
            "Pass latent_dim=<int> to GANTrainer."
        )

    @staticmethod
    def _extract_images_from_batch(batch):
        # Datasets sometimes return (image, label); we only need the image.
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn((batch_size, self.latent_dim), device=self.device)

    def _make_targets(self, batch_size: int):
        real_targets = torch.full(
            (batch_size, 1), self.real_label_smoothing, device=self.device
        )
        fake_targets = torch.zeros((batch_size, 1), device=self.device)
        return real_targets, fake_targets

    def _augment_for_d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DiffAugment to anything that is about to be fed into D.

        The same policy is applied to real and fake inputs - this is the
        critical invariant of DiffAugment.  Each call samples fresh
        random augmentations, so D never sees the same augmented view
        twice.
        """
        return diff_augment(x, self.diff_augment_policy)

    def _discriminator_step(
        self,
        real_images: torch.Tensor,
        real_targets: torch.Tensor,
        fake_targets: torch.Tensor,
    ) -> torch.Tensor:
        self.discriminator_optimizer.zero_grad(set_to_none=True)

        real_preds = self.discriminator(self._augment_for_d(real_images))
        real_loss = self.adversarial_loss(real_preds, real_targets)

        with torch.no_grad():
            noise = self._sample_noise(real_images.size(0))
            generated = self.generator(noise)

        fake_preds = self.discriminator(self._augment_for_d(generated))
        fake_loss = self.adversarial_loss(fake_preds, fake_targets)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.discriminator_optimizer.step()
        return d_loss

    def _generator_step(self, batch_size: int) -> torch.Tensor:
        self.generator_optimizer.zero_grad(set_to_none=True)

        noise = self._sample_noise(batch_size)
        generated = self.generator(noise)
        # G wants D to think generated images are real, so target is 1.0.
        # We deliberately do NOT label-smooth here (smoothing is only on
        # real targets - one-sided smoothing).
        misleading_targets = torch.ones((batch_size, 1), device=self.device)
        # IMPORTANT: DiffAugment must be inside the gradient path so its
        # derivatives flow back through G.  That is the whole point of
        # the augmentation being differentiable - it teaches G to be
        # robust to color/translation/cutout perturbations.
        preds = self.discriminator(self._augment_for_d(generated))
        g_loss = self.adversarial_loss(preds, misleading_targets)
        g_loss.backward()
        self.generator_optimizer.step()
        return g_loss

    def _train_epoch(self, loader) -> tuple[float, float]:
        g_sum, d_sum, n = 0.0, 0.0, 0
        for batch in loader:
            images = self._extract_images_from_batch(batch)
            real_images = images.to(self.device, dtype=torch.float32)
            batch_size = real_images.size(0)
            real_targets, fake_targets = self._make_targets(batch_size)

            d_loss = self._discriminator_step(
                real_images=real_images,
                real_targets=real_targets,
                fake_targets=fake_targets,
            )
            g_loss = self._generator_step(batch_size=batch_size)

            g_sum += float(g_loss.detach().item())
            d_sum += float(d_loss.detach().item())
            n += 1

        n = max(n, 1)
        return g_sum / n, d_sum / n

    def fit(self, loader) -> None:
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.generator.train()
        self.discriminator.train()

        epoch_bar = tqdm(range(self.epochs), desc="Epochs")
        for epoch in epoch_bar:
            g_loss_epoch, d_loss_epoch = self._train_epoch(loader)

            self.generator_loss_history.append(g_loss_epoch)
            self.discriminator_loss_history.append(d_loss_epoch)
            epoch_bar.set_postfix(
                g_loss=f"{g_loss_epoch:.4f}",
                d_loss=f"{d_loss_epoch:.4f}",
            )

            if self.on_epoch_end is not None:
                self.on_epoch_end(self, epoch)

    @torch.no_grad()
    def predict(self, num_samples: int) -> torch.Tensor:
        self.generator.eval()
        self.generator.to(self.device)
        noise = self._sample_noise(num_samples)
        pred = self.generator(noise)
        self.generator.train()
        return pred.detach().cpu()

    @torch.no_grad()
    def predict_fixed(self) -> torch.Tensor:
        """Generate samples from the trainer's frozen evaluation noise."""
        self.generator.eval()
        pred = self.generator(self.fixed_noise)
        self.generator.train()
        return pred.detach().cpu()
