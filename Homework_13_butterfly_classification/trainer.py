import datetime as dt
import inspect
from typing import Callable

import torch


class Trainer:
    """
    Parameters:
        net: модель (torch.nn.Module)
        criterion: функция потерь
        optimizer: уже созданный оптимизатор (например Adam(net.parameters(), lr=...))
        device: устройство для вычислений
        epoch_amount: число эпох
        max_batches_per_epoch: ограничение числа батчей за эпоху (train/val) или None
        early_stopping: остановка, если val loss не улучшается столько эпох
        scheduler: фабрика расписания шага, scheduler(optimizer) или None

    Attributes:
        start_model: исходная модель
        best_model: ссылка на модель при лучшем val loss (та же сеть, что и start_model)
        train_loss: средний loss по эпохам на train
        val_loss: средний loss по эпохам на val

    Methods:
        fit(train_loader, val_loader=None): обучение (с валидацией или без)
        predict(test_loader): предсказания для батчей без меток, tensor на CPU
    """

    def __init__(
        self,
        net,
        criterion,
        optimizer,
        device,
        *,
        epoch_amount=1000,
        max_batches_per_epoch=None,
        early_stopping=10,
        scheduler=None,
        epoch_end_callback: Callable[[int, float, float | None], None] | None = None,
        verbose: bool = True,
    ):
        self.start_model = net
        self.best_model = net
        self.loss_f = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch_amount = epoch_amount
        self.max_batches_per_epoch = max_batches_per_epoch
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.epoch_end_callback = epoch_end_callback
        self.verbose = verbose

        self.train_loss = []
        self.val_loss = []

    def fit(self, train_loader, val_loader=None):
        if self.loss_f is None or self.optimizer is None:
            raise RuntimeError(
                "Cannot call fit() without criterion and optimizer. "
                "Use load_best_model()/from_checkpoint() only for inference."
            )
        Net = self.start_model
        Net.to(self.device)
        Net.train()

        sched = None
        if self.scheduler is not None:
            sched = self.scheduler(self.optimizer)

        best_val_loss = float("inf")
        best_ep = 0
        best_state_dict = None

        for epoch in range(self.epoch_amount):
            start = dt.datetime.now()
            if self.verbose:
                print(f"Epoch: {epoch}", end=" ")
            Net.train()
            mean_loss = 0.0
            batch_n = 0

            for batch_X, target in train_loader:
                if self.max_batches_per_epoch is not None and batch_n >= self.max_batches_per_epoch:
                    break

                self.optimizer.zero_grad()
                # Keep a stable dtype across NumPy/DataLoader conversions.
                batch_X = batch_X.to(self.device, dtype=torch.float32, non_blocking=True)
                target = target.to(self.device, non_blocking=True).long()

                predicted_values = Net(batch_X)
                num_classes = predicted_values.shape[1]
                min_target = int(target.min().item())
                max_target = int(target.max().item())
                if min_target < 0 or max_target >= num_classes:
                    raise ValueError(
                        f"Target labels must be in [0, {num_classes - 1}], "
                        f"got [{min_target}, {max_target}]"
                    )
                loss = self.loss_f(predicted_values, target)
                # Guard against criterion(reduction="none") returning per-sample losses.
                if loss.ndim > 0:
                    loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                mean_loss += loss.item()
                batch_n += 1

            mean_loss /= max(batch_n, 1)
            self.train_loss.append(mean_loss)
            if self.verbose:
                print(f"Loss_train: {mean_loss}, {dt.datetime.now() - start} sec")

            if val_loader is None and self.epoch_end_callback is not None:
                self.epoch_end_callback(epoch, self.train_loss[-1], None)

            metric_for_scheduler = self.train_loss[-1]
            if val_loader is not None:
                Net.eval()
                mean_loss = 0.0
                batch_n = 0

                with torch.no_grad():
                    for batch_X, target in val_loader:
                        if self.max_batches_per_epoch is not None and batch_n >= self.max_batches_per_epoch:
                            break

                        batch_X = batch_X.to(self.device, dtype=torch.float32, non_blocking=True)
                        target = target.to(self.device, non_blocking=True).long()
                        predicted_values = Net(batch_X)
                        num_classes = predicted_values.shape[1]
                        min_target = int(target.min().item())
                        max_target = int(target.max().item())
                        if min_target < 0 or max_target >= num_classes:
                            raise ValueError(
                                f"Target labels must be in [0, {num_classes - 1}], "
                                f"got [{min_target}, {max_target}]"
                            )
                        loss = self.loss_f(predicted_values, target)
                        if loss.ndim > 0:
                            loss = loss.mean()

                        mean_loss += loss.item()
                        batch_n += 1

                mean_loss /= max(batch_n, 1)
                self.val_loss.append(mean_loss)
                metric_for_scheduler = mean_loss
                if self.verbose:
                    print(f"Loss_val: {mean_loss}")

                if self.epoch_end_callback is not None:
                    # Called exactly once per epoch after both train+val losses are computed.
                    self.epoch_end_callback(epoch, self.train_loss[-1], self.val_loss[-1])

                if mean_loss < best_val_loss:
                    best_val_loss = mean_loss
                    best_ep = epoch
                    # Freeze best weights for later predict().
                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in Net.state_dict().items()
                    }
                elif epoch - best_ep > self.early_stopping:
                    if self.verbose:
                        print(
                            f"No improvement for {self.early_stopping} epochs. Stopping training..."
                        )
                    break
            if sched is not None:
                # PyTorch 2.x ReduceLROnPlateau.step(metrics=...) — avoid isinstance
                # (reload/Jupyter quirks) and plain .step() which omits required metrics.
                sig = inspect.signature(sched.step)
                if "metrics" in sig.parameters:
                    sched.step(metrics=metric_for_scheduler)
                else:
                    sched.step()
            if self.verbose:
                print()

        # Load best weights once at the end of training (or early stopping).
        if best_state_dict is not None:
            self.best_model.load_state_dict(best_state_dict)

    def save_best_model(self, path: str, *, include_optimizer: bool = False, extra: dict | None = None):
        checkpoint = {
            "model_state_dict": self.best_model.state_dict(),
        }
        if include_optimizer and self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if extra is not None:
            checkpoint["extra"] = extra
        torch.save(checkpoint, path)

    def load_best_model(
        self,
        path: str,
        *,
        map_location=None,
        strict: bool = True,
        load_optimizer: bool = False,
    ):
        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            # Support plain state_dict checkpoints.
            model_state_dict = checkpoint

        self.best_model.load_state_dict(model_state_dict, strict=strict)
        self.start_model.load_state_dict(model_state_dict, strict=strict)

        if (
            load_optimizer
            and self.optimizer is not None
            and isinstance(checkpoint, dict)
            and "optimizer_state_dict" in checkpoint
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    @classmethod
    def from_checkpoint(
        cls,
        net,
        device,
        checkpoint_path: str,
        *,
        strict: bool = True,
        verbose: bool = False,
    ):
        trainer = cls(
            net,
            criterion=None,
            optimizer=None,
            device=device,
            epoch_amount=0,
            verbose=verbose,
        )
        trainer.load_best_model(checkpoint_path, strict=strict)
        return trainer

    def predict(self, test_loader):
        self.best_model.eval()
        self.best_model.to(self.device)
        out = []
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    batch_X = batch[0]
                else:
                    batch_X = batch
                batch_X = batch_X.to(self.device, dtype=torch.float32, non_blocking=True)
                pred = self.best_model(batch_X)
                out.append(pred.detach().cpu())

        return torch.cat(out, dim=0)


def fit_with_live_plot(
    net,
    criterion,
    optimizer,
    device,
    train_loader,
    valid_loader=None,
    *,
    epoch_amount=1000,
    max_batches_per_epoch=None,
    early_stopping=10,
    scheduler=None,
    figsize=(7, 4),
    verbose=False,
):
    """
    Train model with an in-notebook live loss plot.

    Returns:
        trainer: trained Trainer instance
        history: dict with epoch/train_loss/val_loss lists
    """
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display

    fig, ax = plt.subplots(figsize=figsize)
    train_line, = ax.plot([], [], label="train")
    val_line, = ax.plot([], [], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Some notebook frontends do not support display_id updates reliably.
    # Keep a fallback that redraws the figure each epoch.
    handle = display(fig, display_id=True)
    can_update_handle = hasattr(handle, "update")

    epoch_numbers: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    def on_epoch_end(epoch: int, train_loss: float, val_loss: float | None) -> None:
        epoch_numbers.append(epoch + 1)
        train_losses.append(train_loss)
        if val_loss is not None:
            val_losses.append(val_loss)

        train_line.set_data(epoch_numbers, train_losses)
        if val_losses:
            val_line.set_data(epoch_numbers[: len(val_losses)], val_losses)

        ax.relim()
        ax.autoscale_view()
        if can_update_handle:
            try:
                handle.update(fig)
            except Exception:
                clear_output(wait=True)
                display(fig)
        else:
            clear_output(wait=True)
            display(fig)
        plt.pause(0.001)

    trainer = Trainer(
        net,
        criterion,
        optimizer,
        device,
        epoch_amount=epoch_amount,
        max_batches_per_epoch=max_batches_per_epoch,
        early_stopping=early_stopping,
        scheduler=scheduler,
        epoch_end_callback=on_epoch_end,
        verbose=verbose,
    )
    trainer.fit(train_loader, valid_loader)

    history = {
        "epoch": epoch_numbers,
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
    return trainer, history
