# sequence_generator.py

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PIL import Image
import matplotlib.pyplot as plt

from filesystem.file_io import FileIO
from breast_cancer_classifier import BreastCancerClassifier


@dataclass(frozen=True)
class SequenceFrame:
    title: str
    image: Image.Image


class SequenceGenerator:
    """
    Produces a 1280x768 "sequence diagram" image showing the model transforming an input.

    - Uses BreastCancerClassifier to build model
    - Loads weights from FileIO like Tester
    - Captures intermediate tensors from the torch Sequential
    - Renders a 2x4 gallery where each tile is 256x256 (dumb upscaled)
    """

    OUT_SUBDIR = "training_run/sequence"
    OUT_W = 1280
    OUT_H = 768

    TILE = 256
    GRID_COLS = 4
    GRID_ROWS = 2

    DPI = 100  # 1280/100 = 12.8 inches, 768/100 = 7.68 inches

    @classmethod
    def _select_device(cls) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ----------------------------
    # Tensor -> image helpers
    # ----------------------------

    @classmethod
    def _to_uint8_gray(cls, x2d: np.ndarray) -> np.ndarray:
        x = x2d.astype(np.float32)
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x, dtype=np.uint8)
        x = (x - mn) / (mx - mn)
        return (x * 255.0).clip(0, 255).astype(np.uint8)

    @classmethod
    def _tensor_4d_to_gray_tile(cls, t: torch.Tensor, size: int = 256) -> Image.Image:
        """
        t: [1, C, H, W] or [C, H, W] or [H, W]
        We convert to a single 2D map by mean over channels, then min/max normalize, then upscale.
        """
        with torch.no_grad():
            tt = t.detach().float().cpu()
            if tt.ndim == 4:
                tt = tt[0]  # [C,H,W]
            if tt.ndim == 3:
                tt = tt.mean(dim=0)  # [H,W]
            if tt.ndim != 2:
                raise ValueError(f"Expected 2D/3D/4D tensor for gray tile, got shape {tuple(tt.shape)}")

            arr = tt.numpy()
            u8 = cls._to_uint8_gray(arr)
            im = Image.fromarray(u8, mode="L")
            im = im.resize((size, size), resample=Image.NEAREST)  # dumb upscale
            # convert to RGB for consistent plotting
            return im.convert("RGB")

    @classmethod
    def _tensor_input_to_rgb_tile(cls, xb: torch.Tensor, size: int = 256) -> Image.Image:
        """
        xb: [1,3,H,W] in float. We clamp to [0,1] if it looks normalized.
        """
        with torch.no_grad():
            x = xb.detach().float().cpu()
            if x.ndim != 4 or x.shape[1] != 3:
                # fallback: show as gray
                return cls._tensor_4d_to_gray_tile(x, size=size)

            x = x[0]  # [3,H,W]
            # Heuristic: if values look like [0..1], keep; if [0..255], normalize
            mn = float(x.min())
            mx = float(x.max())
            if mx > 1.5:
                x = x / 255.0
            x = x.clamp(0.0, 1.0)

            arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # [H,W,3]
            im = Image.fromarray(arr, mode="RGB")
            im = im.resize((size, size), resample=Image.NEAREST)
            return im

    @classmethod
    def _linear_1xN_to_tile(cls, t: torch.Tensor, size: int = 256, side: int = 16) -> Image.Image:
        """
        t: [1, N] where N == side*side (default 256 -> 16x16)
        Render as a heatmap tile.
        """
        with torch.no_grad():
            tt = t.detach().float().cpu()
            if tt.ndim != 2 or tt.shape[0] != 1:
                return cls._tensor_4d_to_gray_tile(tt, size=size)

            n = int(tt.shape[1])
            if n != side * side:
                # fallback: squash to nearest square-ish by truncating
                s = int(np.floor(np.sqrt(n)))
                n2 = s * s
                vals = tt[0, :n2].reshape(s, s).numpy()
            else:
                vals = tt[0].reshape(side, side).numpy()

            u8 = cls._to_uint8_gray(vals)
            im = Image.fromarray(u8, mode="L").resize((size, size), resample=Image.NEAREST)
            return im.convert("RGB")

    @classmethod
    def _logits_to_color_strip(cls, logits: torch.Tensor, size: int = 256) -> Image.Image:
        """
        logits: [1,2]. Render as a simple 2-cell strip, then upscale.
        """
        with torch.no_grad():
            tt = logits.detach().float().cpu()
            if tt.ndim != 2 or tt.shape[1] < 2:
                return cls._tensor_4d_to_gray_tile(tt, size=size)

            vals = tt[0, :2].numpy().astype(np.float32)
            # normalize logits to 0..1 for visualization
            mn, mx = float(vals.min()), float(vals.max())
            if mx - mn < 1e-12:
                norm = np.zeros_like(vals)
            else:
                norm = (vals - mn) / (mx - mn)

            # create 1x2 image (two “bars”), then upscale
            bar = (norm.reshape(1, 2) * 255.0).astype(np.uint8)
            im = Image.fromarray(bar, mode="L").resize((size, size), resample=Image.NEAREST)
            return im.convert("RGB")

    # ----------------------------
    # Model stepping and capture
    # ----------------------------

    @classmethod
    def _load_model(cls, model_subdir: str, model_file_name: str, model_extension: str, device: torch.device) -> nn.Module:
        model = BreastCancerClassifier.bake_torch().to(device)

        weight_bytes = FileIO.load_local(
            subdirectory=model_subdir,
            name=model_file_name,
            extension=model_extension,
        )
        buf = io.BytesIO(weight_bytes)
        state_dict = torch.load(buf, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @classmethod
    def _forward_capture(cls, model: nn.Module, xb1: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Assumes model is nn.Sequential.
        Returns:
          - list of activations at each layer output
          - final logits
        """
        if not isinstance(model, nn.Sequential):
            raise TypeError("SequenceGenerator expects BreastCancerClassifier.bake_torch() to return nn.Sequential")

        acts: List[torch.Tensor] = []
        x = xb1
        with torch.no_grad():
            for layer in model:
                x = layer(x)
                acts.append(x)
        logits = acts[-1]
        return acts, logits

    # ----------------------------
    # Public API
    # ----------------------------

    @classmethod
    def generate(
        cls,
        loader: DataLoader,
        model_subdir: str = "training_run",
        model_file_name: str = "latest_0050",
        model_extension: str = "pt",
        out_subdir: Optional[str] = None,
        out_name: str = "sequence_diagram",
        sample_index_in_batch: int = 0,
    ) -> None:
        """
        Creates a 1280x768 PNG with 2 rows of feature transformations.

        sample_index_in_batch:
          - picks which item inside the first fetched batch to visualize
        """
        if out_subdir is None:
            out_subdir = cls.OUT_SUBDIR

        device = cls._select_device()
        model = cls._load_model(model_subdir, model_file_name, model_extension, device=device)

        xb, yb = next(iter(loader))
        xb1 = xb[sample_index_in_batch:sample_index_in_batch + 1].to(device)

        acts, logits = cls._forward_capture(model, xb1)

        # ---- Pick indices for “conv/relu/pool” chain ----
        # Your sequential from Swan likely looks like:
        # 0 Conv, 1 ReLU, 2 Pool, 3 Conv, 4 ReLU, 5 Pool, 6 Conv, 7 ReLU, 8 Pool, 9 Flatten, 10 Linear(256), 11 ReLU, 12 Dropout, 13 Linear(2)
        #
        # If this differs, you can tweak these numbers quickly.
        idx_conv1 = 0
        idx_relu1 = 1
        idx_pool1 = 2
        idx_conv2 = 3
        idx_relu2 = 4
        idx_pool2 = 5
        idx_conv3 = 6
        idx_pool3 = 8  # after relu3 at 7

        idx_linear1 = 10
        idx_logits = len(acts) - 1

        frames: List[SequenceFrame] = []

        # Row 1
        frames.append(SequenceFrame("INPUT (upscaled)", cls._tensor_input_to_rgb_tile(xb1, size=cls.TILE)))
        frames.append(SequenceFrame("CONV1 → feature map", cls._tensor_4d_to_gray_tile(acts[idx_conv1], size=cls.TILE)))
        frames.append(SequenceFrame("RELU1 → feature map", cls._tensor_4d_to_gray_tile(acts[idx_relu1], size=cls.TILE)))
        frames.append(SequenceFrame("POOL1 → feature map", cls._tensor_4d_to_gray_tile(acts[idx_pool1], size=cls.TILE)))

        # Row 2
        frames.append(SequenceFrame("CONV2 → feature map", cls._tensor_4d_to_gray_tile(acts[idx_conv2], size=cls.TILE)))
        frames.append(SequenceFrame("RELU2 → feature map", cls._tensor_4d_to_gray_tile(acts[idx_relu2], size=cls.TILE)))
        frames.append(SequenceFrame("POOL2 → feature map", cls._tensor_4d_to_gray_tile(acts[idx_pool2], size=cls.TILE)))

        # “Linear represented with color”
        # linear1 is [1,256] in your net, render 16x16 heat tile
        lin = acts[idx_linear1]
        lin_tile = cls._linear_1xN_to_tile(lin, size=cls.TILE, side=16)

        # also blend logits as a strip in the corner? keep simple: just the linear heat
        frames.append(SequenceFrame("LINEAR(256) heat (16×16)", lin_tile))

        # ---- Render 2x4 into a single 1280x768 canvas ----
        fig = plt.figure(figsize=(cls.OUT_W / cls.DPI, cls.OUT_H / cls.DPI), dpi=cls.DPI)
        fig.suptitle(
            f"Model Signal Flow: {model_file_name}.{model_extension}  (device={device})",
            fontsize=18,
            fontweight="bold",
        )

        for i, frame in enumerate(frames):
            ax = plt.subplot(cls.GRID_ROWS, cls.GRID_COLS, i + 1)
            ax.imshow(frame.image)
            ax.set_title(frame.title, fontsize=12)
            ax.axis("off")

        # Add a small logits stamp at bottom-right of the whole figure
        logits_tile = cls._logits_to_color_strip(acts[idx_logits], size=128)
        # place as an inset-like image
        ax_inset = fig.add_axes([0.86, 0.04, 0.10, 0.10])  # [left,bottom,width,height] in figure coords
        ax_inset.imshow(logits_tile)
        ax_inset.set_title("LOGITS", fontsize=10)
        ax_inset.axis("off")

        plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.92])

        out_buf = io.BytesIO()
        fig.savefig(out_buf, format="png", dpi=cls.DPI)
        plt.close(fig)
        out_buf.seek(0)

        FileIO.save_local(
            data=out_buf.read(),
            subdirectory=out_subdir,
            name=out_name,
            extension="png",
        )

        print(f"[SequenceGenerator] wrote {out_subdir}/{out_name}.png")
