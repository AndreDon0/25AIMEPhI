import datetime as dt
import inspect

import torch
from tqdm.auto import tqdm


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

    def _batch_bar_total(self, loader):
        if self.max_batches_per_epoch is None:
            return len(loader) if hasattr(loader, "__len__") else None
        if hasattr(loader, "__len__"):
            return min(len(loader), self.max_batches_per_epoch)
        return self.max_batches_per_epoch

    @staticmethod
    def _logits(outputs):
        """HF models return ModelOutput with `.logits`; plain modules return logits tensor."""
        return outputs.logits if hasattr(outputs, "logits") else outputs

    @staticmethod
    def _prepare_target(target: torch.Tensor, device) -> torch.Tensor:
        """Multi-label (B, C) with C>1 → float for BCEWithLogitsLoss; else class indices long."""
        target = target.to(device)
        if target.dim() == 2 and target.shape[1] > 1:
            return target.float()
        if target.dim() == 2 and target.shape[1] == 1:
            return target.squeeze(1).long()
        return target.long()

    def fit(self, train_loader, val_loader=None):
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
            print(f"Эпоха: {epoch}", end=" ")
            Net.train()
            mean_loss = 0.0
            batch_n = 0

            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch} train",
                total=self._batch_bar_total(train_loader),
                leave=False,
            )
            for batch_X, target in train_pbar:
                if self.max_batches_per_epoch is not None and batch_n >= self.max_batches_per_epoch:
                    break

                self.optimizer.zero_grad()
                batch_X = batch_X.to(self.device)
                target = self._prepare_target(target, self.device)

                logits = self._logits(Net(batch_X))
                loss = self.loss_f(logits, target)
                loss.backward()
                self.optimizer.step()

                mean_loss += loss.item()
                batch_n += 1
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            mean_loss /= max(batch_n, 1)
            self.train_loss.append(mean_loss)
            print(f"Loss_train: {mean_loss}, {dt.datetime.now() - start} сек")

            metric_for_scheduler = self.train_loss[-1]
            if val_loader is not None:
                Net.eval()
                mean_loss = 0.0
                batch_n = 0

                with torch.no_grad():
                    val_pbar = tqdm(
                        val_loader,
                        desc=f"Epoch {epoch} val",
                        total=self._batch_bar_total(val_loader),
                        leave=False,
                    )
                    for batch_X, target in val_pbar:
                        if self.max_batches_per_epoch is not None and batch_n >= self.max_batches_per_epoch:
                            break

                        batch_X = batch_X.to(self.device)
                        target = self._prepare_target(target, self.device)
                        logits = self._logits(Net(batch_X))
                        loss = self.loss_f(logits, target)

                        mean_loss += loss.item()
                        batch_n += 1
                        val_pbar.set_postfix(loss=f"{loss.item():.4f}")

                mean_loss /= max(batch_n, 1)
                self.val_loss.append(mean_loss)
                metric_for_scheduler = mean_loss
                print(f"Loss_val: {mean_loss}")

                if mean_loss < best_val_loss:
                    best_val_loss = mean_loss
                    best_ep = epoch
                    # Freeze best weights for later predict().
                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in Net.state_dict().items()
                    }
                elif epoch - best_ep > self.early_stopping:
                    print(
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
            print()

        # Load best weights once at the end of training (or early stopping).
        if best_state_dict is not None:
            self.best_model.load_state_dict(best_state_dict)

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
                batch_X = batch_X.to(self.device)
                pred = self._logits(self.best_model(batch_X))
                out.append(pred.detach().cpu())

        return torch.cat(out, dim=0)

    def save(self, path):
        checkpoint = {
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in self.best_model.state_dict().items()
            },
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }
        torch.save(checkpoint, path)
