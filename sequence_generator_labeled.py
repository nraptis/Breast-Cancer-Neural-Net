# sequence_generator_labeled.py

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, List, Tuple

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


class SequenceGeneratorLabeled:
    """
    Generates a clean, labeled sequence visualization.

    Layout:
      - 2 rows x 4 columns of 256x256 tiles
      - One final True/False strip: 256 tall x 64 wide (softmax probs)
      - Text is drawn BELOW each tile (never on top)
      - Padding:
          top=256, bottom=256, left=64, right=64
      - Uses exact space required.
    """

    OUT_SUBDIR = "training_run/sequence"

    TILE = 256
    COLS = 4
    ROWS = 2

    STRIP_W = 64
    STRIP_H = 256

    PAD_TOP = 256
    PAD_BOTTOM = 256
    PAD_LEFT = 64
    PAD_RIGHT = 64

    GAP_X = 24      # spacing between tiles
    GAP_Y = 36      # spacing between rows
    LABEL_H = 40    # vertical space reserved under each tile for its label
    STRIP_GAP = 24  # gap between 4th column and the strip

    @classmethod
    def _select_device(cls) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cuda")
        # return torch.device("cpu")

    # ----------------------------
    # Loading + forward capture
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
        mn, mx = float(x.min()), float(x.max())
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
    def _truth_strip_256x64(cls, logits_1x2: torch.Tensor) -> Image.Image:
        """
        Render softmax([benign, malignant]) into a 256x64 vertical strip.

        Top half = benign probability brightness
        Bottom half = malignant probability brightness

        (This reads as a simple True/False meter.)
        """
        with torch.no_grad():
            lg = logits_1x2.detach().float().cpu()
            if lg.ndim == 2:
                lg = lg[0]
            if lg.numel() < 2:
                # fallback: all black
                return Image.new("RGB", (cls.STRIP_W, cls.STRIP_H), (0, 0, 0))

            probs = torch.softmax(lg[:2], dim=0).numpy().astype(np.float32)
            p0 = float(probs[0])  # benign
            p1 = float(probs[1])  # malignant

        # brightness 0..255
        b0 = int(np.clip(p0 * 255.0, 0, 255))
        b1 = int(np.clip(p1 * 255.0, 0, 255))

        im = Image.new("RGB", (cls.STRIP_W, cls.STRIP_H), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        # Make it colorful and loud:
        # benign = green channel, malignant = red channel
        draw.rectangle([0, 0, cls.STRIP_W - 1, (cls.STRIP_H // 2) - 1], fill=(0, b0, 0))
        draw.rectangle([0, cls.STRIP_H // 2, cls.STRIP_W - 1, cls.STRIP_H - 1], fill=(b1, 0, 0))

        return im

    # ----------------------------
    # Rendering helpers
    # ----------------------------

    @classmethod
    def _load_font(cls, size: int = 18) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("Arial.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    @classmethod
    def _draw_centered_text(cls, draw: ImageDraw.ImageDraw, text: str, x0: int, y0: int, w: int, h: int, font):
        # center text inside a box (x0,y0,w,h)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x0 + (w - tw) // 2
        ty = y0 + (h - th) // 2
        draw.text((tx, ty), text, fill=(20, 20, 20), font=font)

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
        out_name: str = "sequence_labeled",
        sample_index_in_batch: int = 0,
    ) -> None:
        if out_subdir is None:
            out_subdir = cls.OUT_SUBDIR

        device = cls._select_device()
        model = cls._load_model(model_subdir, model_file_name, model_extension, device=device)

        xb, yb = next(iter(loader))
        xb1 = xb[sample_index_in_batch:sample_index_in_batch + 1].to(device)

        acts = cls._forward_capture(model, xb1)

        # Expected Sequential indices (same assumption as before)
        # 0 Conv1, 1 ReLU1, 2 Pool1, 3 Conv2, 4 ReLU2, 5 Pool2, 6 Conv3, 7 ReLU3, 8 Pool3, 9 Flatten, 10 Linear256, 11 ReLU, 12 Dropout, 13 Linear2
        idx_conv1 = 0
        idx_relu1 = 1
        idx_pool1 = 2
        idx_conv2 = 3

        idx_relu2 = 4
        idx_pool2 = 5
        idx_conv3 = 6
        idx_pool3 = 8

        idx_logits = len(acts) - 1

        # 2 rows x 4 cols
        tiles: List[Tile] = [
            Tile("INPUT", cls._input_to_rgb_tile(xb1)),
            Tile("CONV1", cls._feature_to_gray_tile(acts[idx_conv1])),
            Tile("RELU1", cls._feature_to_gray_tile(acts[idx_relu1])),
            Tile("POOL1", cls._feature_to_gray_tile(acts[idx_pool1])),

            Tile("CONV2", cls._feature_to_gray_tile(acts[idx_conv2])),
            Tile("RELU2", cls._feature_to_gray_tile(acts[idx_relu2])),
            Tile("POOL2", cls._feature_to_gray_tile(acts[idx_pool2])),
            Tile("CONV3/POOL3", cls._feature_to_gray_tile(acts[idx_pool3])),
        ]

        strip = cls._truth_strip_256x64(acts[idx_logits])

        # Compute exact canvas size
        grid_w = cls.COLS * cls.TILE + (cls.COLS - 1) * cls.GAP_X
        grid_h = cls.ROWS * (cls.TILE + cls.LABEL_H) + (cls.ROWS - 1) * cls.GAP_Y

        content_w = grid_w + cls.STRIP_GAP + cls.STRIP_W
        content_h = grid_h

        canvas_w = cls.PAD_LEFT + content_w + cls.PAD_RIGHT
        canvas_h = cls.PAD_TOP + content_h + cls.PAD_BOTTOM

        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = cls._load_font(size=18)
        font_big = cls._load_font(size=22)

        # Optional header text in the top padding area
        header = f"Model Signal Flow: {model_file_name}.{model_extension}   (device={device})"
        cls._draw_centered_text(draw, header, 0, 0, canvas_w, cls.PAD_TOP, font_big)

        # Paste tiles + labels
        start_x = cls.PAD_LEFT
        start_y = cls.PAD_TOP

        for r in range(cls.ROWS):
            for c in range(cls.COLS):
                i = r * cls.COLS + c
                tile = tiles[i]

                x = start_x + c * (cls.TILE + cls.GAP_X)
                y = start_y + r * (cls.TILE + cls.LABEL_H + cls.GAP_Y)

                canvas.paste(tile.image, (x, y))

                label_y = y + cls.TILE
                cls._draw_centered_text(draw, tile.title, x, label_y, cls.TILE, cls.LABEL_H, font)

        # Paste strip (centered vertically over the two rows’ image area)
        strip_x = start_x + grid_w + cls.STRIP_GAP
        # place strip aligned with top row image (not including label)
        strip_y = start_y  # top row image top
        canvas.paste(strip, (strip_x, strip_y))

        # Label for strip under it
        cls._draw_centered_text(draw, "BENIGN / MALIGNANT", strip_x, strip_y + cls.STRIP_H, cls.STRIP_W, cls.LABEL_H, font)

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

        print(f"[SequenceGeneratorLabeled] wrote {out_subdir}/{out_name}.png  size={canvas_w}x{canvas_h}")
