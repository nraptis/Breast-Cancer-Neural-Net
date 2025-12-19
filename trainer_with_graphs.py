# trainer_with_graphs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import io
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from filesystem.file_io import FileIO
from filesystem.file_utils import FileUtils

from breast_cancer_classifier import BreastCancerClassifier


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    split: str  # "train" or "val"
    loss: float
    acc: float


class TrainerWithGraphs:
    """
    Trainer that records epoch stats and writes high-impact seaborn graphs.

    Output folder:
      training_run/graphs/
        - training_curves.png
        - train_vs_val_gap.png
        - metrics_table.csv
        - latest_configuration.json (same as before)
        - latest_####.pt checkpoints (same as before)
    """

    LR = 0.0001
    EPOCHS = 40

    # Mirror model constants for sanity prints only
    IMG_W = BreastCancerClassifier.IMG_W
    IMG_H = BreastCancerClassifier.IMG_H
    IMG_C = BreastCancerClassifier.IMG_C
    NUM_CLASSES = BreastCancerClassifier.NUM_CLASSES

    GRAPH_SUBDIR = "training_run/graphs"
    RUN_SUBDIR = "training_run"

    @classmethod
    def _select_device(cls) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @classmethod
    def _save_fig_local_png(cls, fig: plt.Figure, name: str) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
        buf.seek(0)
        FileIO.save_local(buf.read(), subdirectory=cls.GRAPH_SUBDIR, name=name, extension="png")
        plt.close(fig)

    @classmethod
    def _run_one_epoch(
        cls,
        device: torch.device,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train: bool,
    ) -> Dict[str, float]:
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

            bs = int(yb.numel())
            total_loss += float(loss.item()) * bs
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += bs

        return {
            "loss": total_loss / max(1, total),
            "acc": correct / max(1, total),
        }

    @classmethod
    def _make_graphs(cls, df: pd.DataFrame) -> None:
        sns.set_theme(style="whitegrid", context="talk")

        # 1) Training curves (loss + acc)
        fig = plt.figure(figsize=(14, 8))
        ax1 = plt.subplot(2, 1, 1)
        sns.lineplot(data=df, x="epoch", y="loss", hue="split", marker="o", ax=ax1)
        ax1.set_title("ALARM BELL: Loss Curve (Train vs Validation)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy Loss")

        ax2 = plt.subplot(2, 1, 2)
        sns.lineplot(data=df, x="epoch", y="acc", hue="split", marker="o", ax=ax2)
        ax2.set_title("ALARM BELL: Accuracy Curve (Train vs Validation)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")

        plt.tight_layout()
        cls._save_fig_local_png(fig, "training_curves")

        # 2) Gap plot (train - val): overfit siren
        piv = df.pivot(index="epoch", columns="split", values=["loss", "acc"])
        # Some runs might not have val
        if ("loss", "val") in piv.columns and ("loss", "train") in piv.columns:
            gap = pd.DataFrame({
                "epoch": piv.index.values,
                "loss_gap_train_minus_val": (piv[("loss", "train")] - piv[("loss", "val")]).values,
                "acc_gap_train_minus_val": (piv[("acc", "train")] - piv[("acc", "val")]).values,
            })

            fig2 = plt.figure(figsize=(14, 6))
            ax = plt.gca()
            sns.lineplot(data=gap, x="epoch", y="loss_gap_train_minus_val", marker="o", ax=ax, label="loss gap (train - val)")
            sns.lineplot(data=gap, x="epoch", y="acc_gap_train_minus_val", marker="o", ax=ax, label="acc gap (train - val)")
            ax.axhline(0.0, linestyle="--", linewidth=2)
            ax.set_title("OVERFITTING SIREN: Train–Val Gap (closer to 0 is better)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Gap")
            plt.tight_layout()
            cls._save_fig_local_png(fig2, "train_vs_val_gap")

    @classmethod
    def train(
        cls,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        checkpoint_every: int = 10,
    ):
        if epochs is None:
            epochs = cls.EPOCHS

        device = cls._select_device()

        swan_model = BreastCancerClassifier.bake_swan()
        model = swan_model.to_torch_sequential().to(device)

        FileUtils.save_local_json(
            obj=swan_model.to_json(),
            subdirectory=cls.RUN_SUBDIR,
            name="latest_configuration",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=cls.LR)
        criterion = nn.CrossEntropyLoss()

        print(swan_model.to_pretty_print())
        print(f"[TrainerWithGraphs] device={device} lr={cls.LR} epochs={epochs}")

        rows: List[EpochStats] = []
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            tr = cls._run_one_epoch(device, model, train_loader, optimizer, criterion, train=True)
            rows.append(EpochStats(epoch=epoch, split="train", loss=tr["loss"], acc=tr["acc"]))

            if val_loader is not None:
                va = cls._run_one_epoch(device, model, val_loader, optimizer, criterion, train=False)
                rows.append(EpochStats(epoch=epoch, split="val", loss=va["loss"], acc=va["acc"]))
                print(f"epoch {epoch:02d} | train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | val loss={va['loss']:.4f} acc={va['acc']:.4f}")
            else:
                print(f"epoch {epoch:02d} | train loss={tr['loss']:.4f} acc={tr['acc']:.4f}")

            if (epoch % checkpoint_every) == 0 or epoch == epochs:
                buf = io.BytesIO()
                torch.save(model.state_dict(), buf)
                buf.seek(0)
                FileIO.save_local(
                    data=buf.read(),
                    subdirectory=cls.RUN_SUBDIR,
                    name=f"latest_{epoch:04d}",
                    extension="pt",
                )

        dt = time.time() - t0
        print(f"[TrainerWithGraphs] finished in {dt:.1f}s")

        df = pd.DataFrame([r.__dict__ for r in rows])
        # Save table
        FileUtils.save_local_json(df.to_dict(orient="list"), subdirectory=cls.GRAPH_SUBDIR, name="metrics_table", extension="json")

        # Graphs
        cls._make_graphs(df)

        return swan_model
