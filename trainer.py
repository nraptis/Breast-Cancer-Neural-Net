# trainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from collections import Counter
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from filesystem.file_utils import FileUtils
from filesystem.file_io import FileIO

from breast_cancer_classifier import BreastCancerClassifier


@dataclass(frozen=True)
class TrainStats:
    loss: float
    acc: float


class Trainer:
    """
    Minimal Torch trainer for binary classification (benign vs malignant).

    Notes:
      - Trainer does NOT define the model architecture.
      - Trainer consumes BreastCancerClassifier as the single source of truth.
      - Validation NEVER updates weights.
    """

    # Training hyperparameters
    LR = 0.0001
    EPOCHS = 5

    # Mirror model constants (for sanity checks / prints only)
    IMG_W = BreastCancerClassifier.IMG_W
    IMG_H = BreastCancerClassifier.IMG_H
    IMG_C = BreastCancerClassifier.IMG_C
    NUM_CLASSES = BreastCancerClassifier.NUM_CLASSES

    @classmethod
    def _select_device(cls) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @classmethod
    def _loader_info(cls, name: str, loader: DataLoader) -> None:
        ds = loader.dataset
        n = len(ds)

        ds_name = type(ds).__name__

        xb, yb = next(iter(loader))

        xb_shape = tuple(xb.shape)
        yb_shape = tuple(yb.shape)

        y_list = yb.detach().cpu().tolist()
        ctr = Counter(y_list)

        y_min = min(y_list) if y_list else None
        y_max = max(y_list) if y_list else None

        print(f"[Data] {name}: dataset={ds_name} n={n} batch_size={loader.batch_size}")
        print(f"[Data] {name}: xb shape={xb_shape} dtype={xb.dtype}")
        print(f"[Data] {name}: yb shape={yb_shape} dtype={yb.dtype}")
        print(f"[Data] {name}: label counts={dict(ctr)} range=[{y_min},{y_max}]")
        print(
            f"[Data] {name}: expected xb=[N,{cls.IMG_C},{cls.IMG_H},{cls.IMG_W}] "
            f"y in [0,{cls.NUM_CLASSES-1}]"
        )

    @classmethod
    def train(
        cls,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ):
        if epochs is None:
            epochs = cls.EPOCHS

        cls._loader_info("train", train_loader)
        if val_loader is not None:
            cls._loader_info("val", val_loader)
        else:
            print("[Data] val: None (no validation loader provided)")

        device = cls._select_device()

        # ---- build model from canonical classifier ----
        swan_model = BreastCancerClassifier.bake_swan()
        model = swan_model.to_torch_sequential().to(device)

        # Save architecture snapshot
        FileUtils.save_local_json(
            obj=swan_model.to_json(),
            subdirectory="training_run",
            name="latest_configuration",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=cls.LR)
        criterion = nn.CrossEntropyLoss()

        print(swan_model.to_pretty_print())
        print(f"[Trainer] device={device} optimizer=Adam lr={cls.LR} epochs={epochs}")

        for epoch in range(1, epochs + 1):
            tr = cls._run_one_epoch(
                device=device,
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                train=True,
            )

            if val_loader is not None:
                va = cls._run_one_epoch(
                    device=device,
                    model=model,
                    loader=val_loader,
                    optimizer=optimizer,   # unused
                    criterion=criterion,
                    train=False,
                )
                print(
                    f"epoch {epoch:02d} | "
                    f"train loss={tr.loss:.4f} acc={tr.acc:.4f} | "
                    f"val loss={va.loss:.4f} acc={va.acc:.4f}"
                )
            else:
                print(
                    f"epoch {epoch:02d} | "
                    f"train loss={tr.loss:.4f} acc={tr.acc:.4f}"
                )

            if (epoch % 10) == 0 or epoch == epochs:
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)

                FileIO.save_local(
                    data=buffer.read(),
                    subdirectory="training_run",
                    name=f"latest_{epoch:04d}",
                    extension="pt",
                )

        return swan_model

    @classmethod
    def _run_one_epoch(
        cls,
        device: torch.device,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train: bool,
    ) -> TrainStats:
        model.train() if train else model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                logits = model(xb)
                loss = criterion(logits, yb)

                if train:
                    loss.backward()
                    optimizer.step()

            batch_size = int(yb.numel())
            total_loss += float(loss.item()) * batch_size
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += batch_size

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return TrainStats(loss=avg_loss, acc=acc)
