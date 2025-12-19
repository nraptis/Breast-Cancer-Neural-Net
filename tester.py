# tester.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from filesystem.file_io import FileIO
from breast_cancer_classifier import BreastCancerClassifier


@dataclass(frozen=True)
class TestStats:
    loss: float
    acc: float
    total: int


class Tester:
    """
    Cookbook model tester.

    - Builds the canonical model architecture via BreastCancerClassifier
    - Loads a saved state_dict from FileIO (bytes)
    - Evaluates on a provided DataLoader (no gradients, no weight updates)
    """

    @classmethod
    def _select_device(cls) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @classmethod
    def test(
        cls,
        loader: DataLoader,
        model_subdir: str = "training_run",
        model_file_name: str = "latest_0040",
        model_extension: str = "pt",
        device: Optional[torch.device] = None,
    ) -> TestStats:
        if device is None:
            device = cls._select_device()

        # 1) Build model (must match training)
        model = BreastCancerClassifier.bake_torch().to(device)

        # 2) Load weights (state_dict)
        weight_bytes = FileIO.load_local(
            subdirectory=model_subdir,
            name=model_file_name,
            extension=model_extension,
        )
        buf = io.BytesIO(weight_bytes)
        state_dict = torch.load(buf, map_location=device)
        model.load_state_dict(state_dict)

        # 3) Eval loop
        criterion = nn.CrossEntropyLoss()
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                batch = int(yb.numel())
                total_loss += float(loss.item()) * batch
                preds = logits.argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += batch

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)

        print(f"[Test] device={device} model={model_subdir}/{model_file_name}.{model_extension} "
              f"loss={avg_loss:.4f} acc={acc:.4f} n={total}")

        return TestStats(loss=avg_loss, acc=acc, total=total)
