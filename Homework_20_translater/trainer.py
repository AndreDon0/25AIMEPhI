import datetime as dt
import inspect

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _to_logits(output):
    """HF classification models return a ModelOutput with .logits; RNN modules may return (logits, state)."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


def _cat_batches_maybe_pad_seq(pieces):
    """
    Stack batch outputs on dim=0. For sequence logits (B, L, C), each batch can have a
    different L because collate pads only within the batch; pad L to the global max before cat.
    """
    if not pieces:
        return torch.empty(0)
    if pieces[0].dim() != 3:
        return torch.cat(pieces, dim=0)
    max_len = max(p.size(1) for p in pieces)
    padded = []
    for p in pieces:
        if p.size(1) < max_len:
            d = max_len - p.size(1)
            p = F.pad(p, (0, 0, 0, d, 0, 0))
        padded.append(p)
    return torch.cat(padded, dim=0)


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
        save(path): сохраняет checkpoint с лучшей моделью и историей обучения
        load(path): загружает checkpoint из .pt (модель и история loss, см. save)
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

        self.train_loss = []
        self.val_loss = []

    def fit(self, train_loader, val_loader=None):
        Net = self.start_model
        Net.to(self.device)
        Net.train()

        device_type = getattr(self.device, "type", None)
        if device_type is None:
            # Fallback for string devices like "cpu"
            device_type = str(self.device)
        use_amp = device_type == "cuda"

        # New AMP API (PyTorch 2.4+): torch.amp.* instead of torch.cuda.amp.*
        scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp))

        sched = None
        if self.scheduler is not None:
            sched = self.scheduler(self.optimizer)

        best_val_loss = float("inf")
        best_ep = 0
        best_state_dict = None

        def _loader_total(loader):
            try:
                return len(loader)
            except TypeError:
                return None

        for epoch in range(self.epoch_amount):
            start = dt.datetime.now()
            Net.train()
            mean_loss = 0.0
            batch_n = 0

            train_total = _loader_total(train_loader)
            if train_total is not None and self.max_batches_per_epoch is not None:
                train_total = min(train_total, self.max_batches_per_epoch)

            train_pbar = tqdm(
                train_loader,
                total=train_total,
                desc=f"Epoch {epoch + 1}/{self.epoch_amount} train",
                leave=False,
            )
            for batch_X, target in train_pbar:
                if self.max_batches_per_epoch is not None and batch_n >= self.max_batches_per_epoch:
                    break

                self.optimizer.zero_grad()
                batch_X = batch_X.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True).long()

                with torch.amp.autocast("cuda", enabled=bool(use_amp)):
                    predicted_values = _to_logits(Net(batch_X))
                    if predicted_values.dim() == 3:
                        predicted_values = predicted_values.reshape(
                            -1, predicted_values.size(-1)
                        )
                        target = target.reshape(-1)
                    loss = self.loss_f(predicted_values, target)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                mean_loss += loss.item()
                batch_n += 1
                train_pbar.set_postfix(loss=f"{mean_loss / batch_n:.4f}")

            mean_loss /= max(batch_n, 1)
            self.train_loss.append(mean_loss)
            tqdm.write(
                f"Эпоха {epoch + 1}: Loss_train: {mean_loss}, {dt.datetime.now() - start} сек"
            )

            metric_for_scheduler = self.train_loss[-1]
            if val_loader is not None:
                Net.eval()
                mean_loss = 0.0
                batch_n = 0

                val_total = _loader_total(val_loader)
                if val_total is not None and self.max_batches_per_epoch is not None:
                    val_total = min(val_total, self.max_batches_per_epoch)

                val_pbar = tqdm(
                    val_loader,
                    total=val_total,
                    desc=f"Epoch {epoch + 1}/{self.epoch_amount} val",
                    leave=False,
                )

                with torch.no_grad():
                    for batch_X, target in val_pbar:
                        if self.max_batches_per_epoch is not None and batch_n >= self.max_batches_per_epoch:
                            break

                        batch_X = batch_X.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True).long()
                        with torch.amp.autocast("cuda", enabled=bool(use_amp)):
                            predicted_values = _to_logits(Net(batch_X))
                            if predicted_values.dim() == 3:
                                predicted_values = predicted_values.reshape(
                                    -1, predicted_values.size(-1)
                                )
                                target = target.reshape(-1)
                            loss = self.loss_f(predicted_values, target)

                        mean_loss += loss.item()
                        batch_n += 1
                        val_pbar.set_postfix(loss=f"{mean_loss / batch_n:.4f}")

                mean_loss /= max(batch_n, 1)
                self.val_loss.append(mean_loss)
                metric_for_scheduler = mean_loss
                tqdm.write(f"Эпоха {epoch + 1}: Loss_val: {mean_loss}")

                if mean_loss < best_val_loss:
                    best_val_loss = mean_loss
                    best_ep = epoch
                    # Freeze best weights for later predict().
                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in Net.state_dict().items()
                    }
                elif epoch - best_ep > self.early_stopping:
                    tqdm.write(
                        f"{self.early_stopping} без улучшений. Прекращаем обучение..."
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

        # Load best weights once at the end of training (or early stopping).
        if best_state_dict is not None:
            self.best_model.load_state_dict(best_state_dict)

    def predict(self, test_loader):
        self.best_model.eval()
        self.best_model.to(self.device)
        out = []
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader), desc="Predicting"):
                if isinstance(batch, (list, tuple)):
                    batch_X = batch[0]
                else:
                    batch_X = batch
                batch_X = batch_X.to(self.device)
                pred = _to_logits(self.best_model(batch_X))
                out.append(pred.detach().cpu())

        return _cat_batches_maybe_pad_seq(out)

    def save(self, path):
        checkpoint = {
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in self.best_model.state_dict().items()
            },
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }
        torch.save(checkpoint, path)

    def load(self, path, map_location=None):
        """
        Load weights and optional training history from a .pt file written by save().

        If the file is a raw state_dict (only tensors), only the model weights are restored.
        """
        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.best_model.load_state_dict(checkpoint["model_state_dict"])
            if "train_loss" in checkpoint and checkpoint["train_loss"] is not None:
                self.train_loss = list(checkpoint["train_loss"])
            if "val_loss" in checkpoint and checkpoint["val_loss"] is not None:
                self.val_loss = list(checkpoint["val_loss"])
        else:
            self.best_model.load_state_dict(checkpoint)
