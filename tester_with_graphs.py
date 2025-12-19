# tester_with_graphs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from filesystem.file_io import FileIO
from filesystem.file_utils import FileUtils

from breast_cancer_classifier import BreastCancerClassifier


@dataclass(frozen=True)
class TestStats:
    loss: float
    acc: float
    total: int


class TesterWithGraphs:
    """
    Cookbook evaluator with "alarm bell" graphs.

    Output folder:
      training_run/test_graphs/
        - confusion_matrix.png
        - roc_curve.png
        - pr_curve.png
        - calibration_curve.png
        - probability_histograms.png
        - metrics_summary.json
    """

    RUN_SUBDIR = "training_run"
    GRAPH_SUBDIR = "training_run/test_graphs"

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
        fig.savefig(buf, format="png", dpi=240, bbox_inches="tight")
        buf.seek(0)
        FileIO.save_local(buf.read(), subdirectory=cls.GRAPH_SUBDIR, name=name, extension="png")
        plt.close(fig)

    @classmethod
    def test(
        cls,
        loader: DataLoader,
        model_subdir: str,
        model_file_name: str,
        model_extension: str = "pt",
        device: Optional[torch.device] = None,
    ) -> TestStats:
        sns.set_theme(style="whitegrid", context="talk")

        if device is None:
            device = cls._select_device()

        model = BreastCancerClassifier.bake_torch().to(device)

        weight_bytes = FileIO.load_local(subdirectory=model_subdir, name=model_file_name, extension=model_extension)
        buf = io.BytesIO(weight_bytes)
        state_dict = torch.load(buf, map_location=device)
        model.load_state_dict(state_dict)

        criterion = nn.CrossEntropyLoss()
        model.eval()

        y_true: List[int] = []
        y_pred: List[int] = []
        y_prob1: List[float] = []  # probability of class 1 (malignant), assuming label 1 = malignant
        losses: List[float] = []

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)

                bs = int(yb.numel())
                total_loss += float(loss.item()) * bs
                correct += int((pred == yb).sum().item())
                total += bs

                y_true.extend(yb.detach().cpu().tolist())
                y_pred.extend(pred.detach().cpu().tolist())
                y_prob1.extend(probs[:, 1].detach().cpu().tolist())
                # per-sample loss (approx): store batch loss repeated for bs (fine for viz)
                losses.extend([float(loss.item())] * bs)

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)

        print(f"[TesterWithGraphs] device={device} model={model_subdir}/{model_file_name}.{model_extension} loss={avg_loss:.4f} acc={acc:.4f} n={total}")

        # Build dataframe for plotting
        df = pd.DataFrame({
            "y_true": np.array(y_true, dtype=int),
            "y_pred": np.array(y_pred, dtype=int),
            "p_malignant": np.array(y_prob1, dtype=float),
            "loss": np.array(losses, dtype=float),
        })

        # Confusion matrix
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(df["y_true"].values, df["y_pred"].values):
            cm[int(t), int(p)] += 1

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, square=True, ax=ax)
        ax.set_title("CONFUSION MATRIX (ALARM BELL VIEW)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticklabels(["benign(0)", "malignant(1)"])
        ax.set_yticklabels(["benign(0)", "malignant(1)"], rotation=0)
        cls._save_fig_local_png(fig, "confusion_matrix")

        # ROC + PR + Calibration (requires sklearn, but we fail gracefully)
        summary: dict = {
            "loss": float(avg_loss),
            "acc": float(acc),
            "n": int(total),
        }

        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
            from sklearn.calibration import calibration_curve

            y = df["y_true"].values
            p = df["p_malignant"].values

            # ROC
            fpr, tpr, _ = roc_curve(y, p)
            roc_auc = auc(fpr, tpr)
            summary["roc_auc"] = float(roc_auc)

            fig2 = plt.figure(figsize=(9, 7))
            ax2 = plt.gca()
            ax2.plot(fpr, tpr, linewidth=3)
            ax2.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
            ax2.set_title(f"ROC CURVE (AUC={roc_auc:.4f})")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            cls._save_fig_local_png(fig2, "roc_curve")

            # PR
            prec, rec, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            summary["average_precision"] = float(ap)

            fig3 = plt.figure(figsize=(9, 7))
            ax3 = plt.gca()
            ax3.plot(rec, prec, linewidth=3)
            ax3.set_title(f"PRECISION-RECALL CURVE (AP={ap:.4f})")
            ax3.set_xlabel("Recall")
            ax3.set_ylabel("Precision")
            cls._save_fig_local_png(fig3, "pr_curve")

            # Calibration curve
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
            fig4 = plt.figure(figsize=(9, 7))
            ax4 = plt.gca()
            ax4.plot(mean_pred, frac_pos, marker="o", linewidth=3)
            ax4.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
            ax4.set_title("CALIBRATION CURVE (Are probabilities honest?)")
            ax4.set_xlabel("Mean predicted probability")
            ax4.set_ylabel("Fraction of positives")
            cls._save_fig_local_png(fig4, "calibration_curve")

        except Exception as e:
            summary["sklearn_graphs_error"] = str(e)

        # Probability histograms (most visceral)
        fig5 = plt.figure(figsize=(14, 7))
        ax5 = plt.gca()
        sns.histplot(data=df, x="p_malignant", hue="y_true", bins=30, stat="density", common_norm=False, element="step", ax=ax5)
        ax5.set_title("PROBABILITY SEPARATION (ALARM BELL): p(malignant) by true class")
        ax5.set_xlabel("Predicted probability of malignant (class 1)")
        ax5.set_ylabel("Density")
        cls._save_fig_local_png(fig5, "probability_histograms")

        # Save summary + raw table
        FileUtils.save_local_json(summary, subdirectory=cls.GRAPH_SUBDIR, name="metrics_summary")
        FileUtils.save_local_json(df.to_dict(orient="list"), subdirectory=cls.GRAPH_SUBDIR, name="predictions_table", extension="json")

        return TestStats(loss=float(avg_loss), acc=float(acc), total=int(total))
