# trainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    NUM_CATEGORIES = 2

    LR = 0.001

    EPOCHS = 5

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

            SwanFlatten(),

            # Keras: Dense(NUM_CATEGORIES * 2)
            SwanLinear(out_features=cls.NUM_CATEGORIES * 2, bias=True),
            SwanDropout(p=0.5),

            # Keras: Dense(NUM_CATEGORIES, activation="softmax")
            # Torch: output logits; do NOT add softmax here
            SwanLinear(out_features=cls.NUM_CATEGORIES, bias=True),
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
        """
        Trains the baked model. Returns the SwanModel (which contains torch Sequential).
        """
        if epochs is None:
            epochs = cls.EPOCHS

        swan_model = cls.bake()
        device = cls._select_device()
        model = swan_model.to_torch_sequential().to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=cls.LR)
        criterion = nn.CrossEntropyLoss()

        print(swan_model.to_pretty_print())
        print(f"[Trainer] device={device} lr={cls.LR} epochs={epochs}")

        for epoch in range(1, epochs + 1):
            tr = cls._run_one_epoch(device, model, train_loader, optimizer, criterion, train=True)
            if val_loader is not None:
                va = cls._run_one_epoch(device, model, val_loader, optimizer, criterion, train=False)
                print(f"epoch {epoch:02d} | train loss={tr.loss:.4f} acc={tr.acc:.4f} | val loss={va.loss:.4f} acc={va.acc:.4f}")
            else:
                print(f"epoch {epoch:02d} | train loss={tr.loss:.4f} acc={tr.acc:.4f}")

        return swan_model

    # -------------------------
    # internal
    # -------------------------

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
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                logits = model(xb)              # shape: [N, 2]
                loss = criterion(logits, yb)     # yb shape: [N] of {0,1}

                if train:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.item()) * int(xb.shape[0])
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(xb.shape[0])

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return TrainStats(loss=avg_loss, acc=acc)
