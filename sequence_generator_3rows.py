# sequence_generator_3rows.py

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw, ImageFont

from filesystem.file_io import FileIO
from breast_cancer_classifier import BreastCancerClassifier


@dataclass(frozen=True)
class Tile:
    title: str
    image: Image.Image


class SequenceGenerator3Rows:
    """
    3-row sequence visualization with:
      - 3 rows x 4 cols of 256x256 tiles
      - final prediction strip: 256 tall x 64 wide
      - final linear weights visualized as red/blue "filter" tile
      - labels drawn UNDER images (never on top)
      - top header includes file name + label
      - under prediction: english label + probability scores

    Padding:
      top=256, bottom=256, left=64, right=64
    Uses exact space needed.
    """

    OUT_SUBDIR = "training_run/sequence"

    TILE = 256
    COLS = 4
    ROWS = 3

    STRIP_W = 64
    STRIP_H = 256

    PAD_TOP = 256
    PAD_BOTTOM = 256
    PAD_LEFT = 64
    PAD_RIGHT = 64

    GAP_X = 24
    GAP_Y = 36
    LABEL_H = 40
    STRIP_GAP = 24

    # Label mapping (assumes 0=benign, 1=malignant)
    LABELS = {0: "benign", 1: "malignant"}

    @classmethod
    def _select_device(cls) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ----------------------------
    # Fonts + text
    # ----------------------------

    @classmethod
    def _load_font(cls, size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("Arial.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    @classmethod
    def _draw_centered_text(cls, draw: ImageDraw.ImageDraw, text: str, x0: int, y0: int, w: int, h: int, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x0 + (w - tw) // 2
        ty = y0 + (h - th) // 2
        draw.text((tx, ty), text, fill=(20, 20, 20), font=font)

    # ----------------------------
    # Model load + capture
    # ----------------------------

    @classmethod
    def _load_model(
        cls,
        model_subdir: str,
        model_file_name: str,
        model_extension: str,
        device: torch.device
    ) -> nn.Module:
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
    def _forward_capture(cls, model: nn.Module, xb1: torch.Tensor) -> List[torch.Tensor]:
        if not isinstance(model, nn.Sequential):
            raise TypeError("Expected nn.Sequential from BreastCancerClassifier.bake_torch()")

        acts: List[torch.Tensor] = []
        x = xb1
        with torch.no_grad():
            for layer in model:
                x = layer(x)
                acts.append(x)
        return acts

    # ----------------------------
    # Tensor -> image helpers
    # ----------------------------

    @staticmethod
    def _norm_to_u8(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mn = float(x.min())
        mx = float(x.max())
        if mx - mn < 1e-12:
            return np.zeros_like(x, dtype=np.uint8)
        x = (x - mn) / (mx - mn)
        return (x * 255.0).clip(0, 255).astype(np.uint8)

    @classmethod
    def _input_to_rgb_tile(cls, xb1: torch.Tensor) -> Image.Image:
        x = xb1.detach().float().cpu()
        if x.ndim != 4 or x.shape[1] != 3:
            return cls._feature_to_gray_tile(x)

        x = x[0]  # [3,H,W]
        mx = float(x.max())
        if mx > 1.5:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)

        arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        im = Image.fromarray(arr, mode="RGB")
        return im.resize((cls.TILE, cls.TILE), resample=Image.NEAREST)

    @classmethod
    def _feature_to_gray_tile(cls, t: torch.Tensor) -> Image.Image:
        tt = t.detach().float().cpu()
        if tt.ndim == 4:
            tt = tt[0]  # [C,H,W]
        if tt.ndim == 3:
            tt = tt.mean(dim=0)  # [H,W]
        if tt.ndim != 2:
            raise ValueError(f"Expected feature map 2D/3D/4D; got {tuple(tt.shape)}")

        u8 = cls._norm_to_u8(tt.numpy())
        im = Image.fromarray(u8, mode="L").resize((cls.TILE, cls.TILE), resample=Image.NEAREST)
        return im.convert("RGB")

    @classmethod
    def _linear_activations_1xN_to_tile(cls, t: torch.Tensor, side: int = 16) -> Image.Image:
        """
        Expects [1, N]. For N=256, render 16x16 and upscale to 256x256.
        """
        tt = t.detach().float().cpu()
        if tt.ndim != 2 or tt.shape[0] != 1:
            return cls._feature_to_gray_tile(tt)

        n = int(tt.shape[1])
        s = side
        if n < s * s:
            # pad with zeros if smaller
            vals = torch.zeros((s * s,), dtype=torch.float32)
            vals[:n] = tt[0]
            vals = vals.reshape(s, s).numpy()
        else:
            vals = tt[0, : s * s].reshape(s, s).numpy()

        u8 = cls._norm_to_u8(vals)
        im = Image.fromarray(u8, mode="L").resize((cls.TILE, cls.TILE), resample=Image.NEAREST)
        return im.convert("RGB")

    @classmethod
    def _weights_red_blue_tile(cls, w: torch.Tensor) -> Image.Image:
        """
        Visualize weights as:
          - max weight => solid red
          - min weight => solid blue
          - interpolate linearly

        Accepts nn.Linear.weight shape [out, in] (here [2,256] or [256,*]).
        We render it as a small matrix then upscale to 256x256.
        """
        ww = w.detach().float().cpu().numpy().astype(np.float32)

        # Normalize by max abs for symmetric red/blue
        maxabs = float(np.max(np.abs(ww))) if ww.size else 1.0
        if maxabs < 1e-12:
            maxabs = 1.0
        norm = ww / maxabs  # [-1..1]

        # Map: negative -> blue, positive -> red
        # red = max(0,norm), blue = max(0,-norm)
        red = np.clip(norm, 0.0, 1.0)
        blue = np.clip(-norm, 0.0, 1.0)

        rgb = np.zeros((norm.shape[0], norm.shape[1], 3), dtype=np.uint8)
        rgb[..., 0] = (red * 255.0).astype(np.uint8)
        rgb[..., 2] = (blue * 255.0).astype(np.uint8)

        im = Image.fromarray(rgb, mode="RGB")
        im = im.resize((cls.TILE, cls.TILE), resample=Image.NEAREST)
        return im

    @classmethod
    def _truth_strip_256x64(cls, probs_2: np.ndarray) -> Image.Image:
        """
        probs_2: [benign_prob, malignant_prob]
        top half = benign (green brightness)
        bottom half = malignant (red brightness)
        """
        p0 = float(probs_2[0])
        p1 = float(probs_2[1])

        b0 = int(np.clip(p0 * 255.0, 0, 255))
        b1 = int(np.clip(p1 * 255.0, 0, 255))

        im = Image.new("RGB", (cls.STRIP_W, cls.STRIP_H), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, cls.STRIP_W - 1, (cls.STRIP_H // 2) - 1], fill=(0, b0, 0))
        draw.rectangle([0, cls.STRIP_H // 2, cls.STRIP_W - 1, cls.STRIP_H - 1], fill=(b1, 0, 0))
        return im

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
        out_name: str = "sequence_3rows",
        label: str = "Model Sequence (3 rows)",
        sample_index_in_batch: int = 0,
    ) -> None:
        if out_subdir is None:
            out_subdir = cls.OUT_SUBDIR

        device = cls._select_device()
        model = cls._load_model(model_subdir, model_file_name, model_extension, device=device)

        xb, yb = next(iter(loader))
        xb1 = xb[sample_index_in_batch:sample_index_in_batch + 1].to(device)

        acts = cls._forward_capture(model, xb1)

        # Expected Sequential indices:
        # 0 Conv1, 1 ReLU1, 2 Pool1,
        # 3 Conv2, 4 ReLU2, 5 Pool2,
        # 6 Conv3, 7 ReLU3, 8 Pool3,
        # 9 Flatten,
        # 10 Linear(256), 11 ReLU, 12 Dropout, 13 Linear(2)
        idx_conv1 = 0
        idx_relu1 = 1
        idx_pool1 = 2

        idx_conv2 = 3
        idx_relu2 = 4
        idx_pool2 = 5

        idx_conv3 = 6
        idx_relu3 = 7
        idx_pool3 = 8

        idx_linear1 = 10
        idx_logits = len(acts) - 1

        # Compute prediction + probabilities
        logits = acts[idx_logits].detach().float().cpu()
        if logits.ndim == 2:
            logits2 = logits[0, :2]
        else:
            logits2 = logits[:2]
        probs = torch.softmax(logits2, dim=0).numpy()  # [2]
        pred_idx = int(np.argmax(probs))
        pred_label = cls.LABELS.get(pred_idx, str(pred_idx))

        # Grab final linear weights for the red/blue filter view
        # last layer in Sequential should be nn.Linear(out=2)
        last_layer = None
        for layer in reversed(model):
            if isinstance(layer, nn.Linear):
                last_layer = layer
                break
        if last_layer is None:
            raise RuntimeError("Could not find nn.Linear layer to visualize weights.")
        w_last = last_layer.weight  # [2, 256] expected

        # Build 3 rows x 4 cols tiles
        tiles: List[Tile] = [
            # Row 1
            Tile("INPUT", cls._input_to_rgb_tile(xb1)),
            Tile("CONV1", cls._feature_to_gray_tile(acts[idx_conv1])),
            Tile("RELU1", cls._feature_to_gray_tile(acts[idx_relu1])),
            Tile("POOL1", cls._feature_to_gray_tile(acts[idx_pool1])),

            # Row 2
            Tile("CONV2", cls._feature_to_gray_tile(acts[idx_conv2])),
            Tile("RELU2", cls._feature_to_gray_tile(acts[idx_relu2])),
            Tile("POOL2", cls._feature_to_gray_tile(acts[idx_pool2])),
            Tile("CONV3", cls._feature_to_gray_tile(acts[idx_conv3])),

            # Row 3
            Tile("RELU3", cls._feature_to_gray_tile(acts[idx_relu3])),
            Tile("POOL3", cls._feature_to_gray_tile(acts[idx_pool3])),
            Tile("LINEAR(256) activations", cls._linear_activations_1xN_to_tile(acts[idx_linear1], side=16)),
            Tile("FINAL LINEAR weights (red/blue)", cls._weights_red_blue_tile(w_last)),
        ]

        # Truth strip (benign/malignant)
        strip = cls._truth_strip_256x64(probs)

        # --- Layout math (exact size) ---
        grid_w = cls.COLS * cls.TILE + (cls.COLS - 1) * cls.GAP_X
        grid_h = cls.ROWS * (cls.TILE + cls.LABEL_H) + (cls.ROWS - 1) * cls.GAP_Y

        content_w = grid_w + cls.STRIP_GAP + cls.STRIP_W
        content_h = grid_h

        canvas_w = cls.PAD_LEFT + content_w + cls.PAD_RIGHT
        canvas_h = cls.PAD_TOP + content_h + cls.PAD_BOTTOM

        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        font_header = cls._load_font(26)
        font_label = cls._load_font(18)
        font_pred = cls._load_font(20)

        # Header: file name + label
        header_text = f"{model_file_name}.{model_extension}   |   {label}"
        cls._draw_centered_text(draw, header_text, 0, 0, canvas_w, cls.PAD_TOP, font_header)

        start_x = cls.PAD_LEFT
        start_y = cls.PAD_TOP

        # Paste tiles + labels under them
        for r in range(cls.ROWS):
            for c in range(cls.COLS):
                i = r * cls.COLS + c
                tile = tiles[i]

                x = start_x + c * (cls.TILE + cls.GAP_X)
                y = start_y + r * (cls.TILE + cls.LABEL_H + cls.GAP_Y)

                canvas.paste(tile.image, (x, y))

                # label box below image
                ly = y + cls.TILE
                cls._draw_centered_text(draw, tile.title, x, ly, cls.TILE, cls.LABEL_H, font_label)

        # Place strip aligned with the FIRST row image top
        strip_x = start_x + grid_w + cls.STRIP_GAP
        strip_y = start_y
        canvas.paste(strip, (strip_x, strip_y))

        # Under strip: prediction + probabilities
        # We use the space under the strip within the same column area.
        pred_box_y = strip_y + cls.STRIP_H
        cls._draw_centered_text(draw, "BENIGN / MALIGNANT", strip_x, pred_box_y, cls.STRIP_W, cls.LABEL_H, font_label)

        # Put the english prediction and percentages in the bottom padding area,
        # centered under the strip column, so it reads like a final verdict.
        verdict_y0 = cls.PAD_TOP + content_h + 10
        verdict_h = cls.PAD_BOTTOM - 20

        benign_pct = probs[0] * 100.0
        malignant_pct = probs[1] * 100.0

        verdict_lines = [
            f"Prediction: {pred_label}",
            f"malignant: {malignant_pct:.1f}%",
            f"benign: {benign_pct:.1f}%",
        ]

        # Draw these lines centered under the strip column area
        # Create a temporary draw to measure line spacing simply
        line_h = 28
        total_text_h = line_h * len(verdict_lines)
        base_y = verdict_y0 + (verdict_h - total_text_h) // 2

        for k, line in enumerate(verdict_lines):
            y = base_y + k * line_h
            cls._draw_centered_text(draw, line, strip_x - 96, y, cls.STRIP_W + 192, line_h, font_pred)

        # Save
        bio = io.BytesIO()
        canvas.save(bio, format="PNG")
        bio.seek(0)

        FileIO.save_local(
            data=bio.read(),
            subdirectory=out_subdir,
            name=out_name,
            extension="png",
        )

        print(f"[SequenceGenerator3Rows] wrote {out_subdir}/{out_name}.png  size={canvas_w}x{canvas_h}")
