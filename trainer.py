# trainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from filesystem.file_utils import FileUtils
from filesystem.file_io import FileIO

from swan.swan_factory import SwanFactory
from swan.swan_model import SwanModel
from swan.swan_layer import (
    SwanConv2d,
    SwanReLU,
    SwanMaxPool2d,
    SwanFlatten,
    SwanLinear,
    SwanDropout,
)


@dataclass(frozen=True)
class TrainStats:
    loss: float
    acc: float


class Trainer:
    """
    Minimal Torch trainer for binary classification (benign vs malignant).

    Fixed for now:
      - optimizer: Adam
      - lr: 0.001
      - loss: CrossEntropyLoss (expects logits, NOT softmax)
    """

    IMG_W = 224
    IMG_H = 224
    IMG_C = 3

    NUM_CLASSES = 2

    LR = 0.0001
    EPOCHS = 5

    GRAD_CLIP_NORM = 1.0  # set to 0.0 to disable

    @classmethod
    def _select_device(cls) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @classmethod
    def bake(cls) -> SwanModel:
        layers = [
            SwanConv2d(out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            SwanReLU(),
            SwanMaxPool2d(kernel_size=2, stride=2),

            SwanConv2d(out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            SwanReLU(),
            SwanMaxPool2d(kernel_size=2, stride=2),

            SwanConv2d(out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            SwanReLU(),
            SwanMaxPool2d(kernel_size=2, stride=2),

            SwanFlatten(),
            SwanLinear(out_features=256, bias=True),
            SwanReLU(),
            SwanDropout(p=0.5),
            SwanLinear(out_features=cls.NUM_CLASSES, bias=True),
        ]

        return SwanFactory.build(
            input_image_width=cls.IMG_W,
            input_image_height=cls.IMG_H,
            input_image_channels=cls.IMG_C,
            layers=layers,
        )

    @classmethod
    def train(
        cls,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> SwanModel:
        if epochs is None:
            epochs = cls.EPOCHS

        swan_model = cls.bake()
        device = cls._select_device()

        # Use Swan model (recommended). If you want the baseline linear model,
        # swap the next line to the commented block below.
        model = swan_model.to_torch_sequential().to(device)

        # Baseline linear model:
        # model = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(cls.IMG_W * cls.IMG_H * cls.IMG_C, cls.NUM_CLASSES),
        # ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cls.LR)
        criterion = nn.CrossEntropyLoss()

        print(swan_model.to_pretty_print())
        print(f"[Trainer] device={device} optimizer=Adam lr={optimizer.param_groups[0]['lr']} epochs={epochs}")

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
                    optimizer=optimizer,   # unused in eval, but fine
                    criterion=criterion,
                    train=False,
                )
                print(
                    f"epoch {epoch:02d} | "
                    f"train loss={tr.loss:.4f} acc={tr.acc:.4f} | "
                    f"val loss={va.loss:.4f} acc={va.acc:.4f}"
                )
            else:
                print(f"epoch {epoch:02d} | train loss={tr.loss:.4f} acc={tr.acc:.4f}")

        # ------------------------------------------------------------
        # Save latest model + configuration
        # ------------------------------------------------------------
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        FileIO.save_local(
            data=buffer.read(),
            subdirectory="training_run",
            name="latest",
            extension="pt",
        )

        FileUtils.save_local_json(
            obj=swan_model.to_json(),
            subdirectory="training_run",
            name="latest_configuration",
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
                logits = model(xb)          # [N, 2]
                loss = criterion(logits, yb)

                if train:
                    loss.backward()
                    if cls.GRAD_CLIP_NORM and cls.GRAD_CLIP_NORM > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cls.GRAD_CLIP_NORM)
                    optimizer.step()

            # Stats from the same forward pass (no logits2, no loss2)
            batch_size = int(yb.numel())
            total_loss += float(loss.item()) * batch_size
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += batch_size
        
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return TrainStats(loss=avg_loss, acc=acc)
