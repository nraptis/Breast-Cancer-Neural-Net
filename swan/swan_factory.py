# swan/swan_factory.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch.nn as nn

from swan.swan_layer import (
    SwanLayer,
    SwanConv2d,
    SwanReLU,
    SwanMaxPool2d,
    SwanDropout,
    SwanLinear,
    SwanFlatten,
)
from swan.swan_model import SwanModel


@dataclass(frozen=True)
class _ShapeCHW:
    c: int
    h: int
    w: int


@dataclass(frozen=True)
class _ShapeFlat:
    n: int


class SwanFactory:
    """
    Builds a SwanModel:
      - wires Conv2d(in_channels=...) and Linear(in_features=...)
      - tracks tensor shape through Conv/Pool/Flatten/Linear
      - does NOT insert layers automatically
      - raises if a layer sequence is invalid (e.g., Linear before Flatten)
    """

    @classmethod
    def build(
        cls,
        input_image_width: int,
        input_image_height: int,
        input_image_channels: int,
        layers: List[SwanLayer],
    ) -> SwanModel:
        shape: _ShapeCHW | _ShapeFlat = _ShapeCHW(
            c=input_image_channels,
            h=input_image_height,
            w=input_image_width,
        )

        modules: List[nn.Module] = []

        for idx, layer in enumerate(layers):
            # --------------------
            # Conv2d
            # --------------------
            if isinstance(layer, SwanConv2d):
                if not isinstance(shape, _ShapeCHW):
                    raise ValueError(
                        f"Layer[{idx}] Conv2d expects CHW input, but current shape is FLAT(n={shape.n}). "
                        f"Did you put Conv2d after Flatten/Linear?"
                    )

                conv = nn.Conv2d(
                    in_channels=shape.c,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=layer.bias,
                )
                modules.append(conv)
                shape = cls._shape_after_conv2d(shape, conv)

            # --------------------
            # ReLU
            # --------------------
            elif isinstance(layer, SwanReLU):
                modules.append(layer.to_torch_module())
                # shape unchanged

            # --------------------
            # MaxPool2d
            # --------------------
            elif isinstance(layer, SwanMaxPool2d):
                if not isinstance(shape, _ShapeCHW):
                    raise ValueError(
                        f"Layer[{idx}] MaxPool2d expects CHW input, but current shape is FLAT(n={shape.n}). "
                        f"Did you put pooling after Flatten/Linear?"
                    )

                pool = layer.to_torch_module()
                modules.append(pool)
                shape = cls._shape_after_maxpool2d(shape, pool)

            # --------------------
            # Dropout
            # --------------------
            elif isinstance(layer, SwanDropout):
                modules.append(layer.to_torch_module())
                # shape unchanged (works for both CHW and FLAT)

            # --------------------
            # Flatten
            # --------------------
            elif isinstance(layer, SwanFlatten):
                if isinstance(shape, _ShapeFlat):
                    raise ValueError(
                        f"Layer[{idx}] Flatten is redundant: current shape is already FLAT(n={shape.n})."
                    )

                modules.append(layer.to_torch_module())
                shape = _ShapeFlat(n=shape.c * shape.h * shape.w)

            # --------------------
            # Linear
            # --------------------
            elif isinstance(layer, SwanLinear):
                if not isinstance(shape, _ShapeFlat):
                    raise ValueError(
                        f"Layer[{idx}] Linear expects FLAT input, but current shape is CHW(c={shape.c}, h={shape.h}, w={shape.w}). "
                        f"Add SwanFlatten() before SwanLinear()."
                    )

                lin = nn.Linear(in_features=shape.n, out_features=layer.out_features, bias=layer.bias)
                modules.append(lin)
                shape = _ShapeFlat(n=layer.out_features)

            else:
                raise TypeError(f"Unsupported SwanLayer type: {type(layer).__name__}")

        torch_seq = nn.Sequential(*modules)

        return SwanModel(
            input_image_width=input_image_width,
            input_image_height=input_image_height,
            input_image_channels=input_image_channels,
            layers=list(layers),
            _torch=torch_seq,
        )

    # ================================================================
    # Shape math
    # ================================================================

    @staticmethod
    def _as_hw(x: int | Tuple[int, int]) -> Tuple[int, int]:
        """
        Normalize a torch int-or-pair property to (h, w).
        We do this explicitly instead of a generic _pair helper.
        """
        if isinstance(x, tuple):
            if len(x) != 2:
                raise ValueError(f"Expected 2-tuple, got: {x}")
            return int(x[0]), int(x[1])
        return int(x), int(x)

    @classmethod
    def _shape_after_conv2d(cls, shape: _ShapeCHW, conv: nn.Conv2d) -> _ShapeCHW:
        """
        PyTorch conv output size formula per dimension:
          out = floor((in + 2p - d*(k-1) - 1)/s + 1)
        """
        k_h, k_w = cls._as_hw(conv.kernel_size)
        s_h, s_w = cls._as_hw(conv.stride)
        p_h, p_w = cls._as_hw(conv.padding)
        d_h, d_w = cls._as_hw(conv.dilation)

        out_h = (shape.h + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        out_w = (shape.w + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"Conv2d produced non-positive shape: H={out_h}, W={out_w}")

        return _ShapeCHW(c=int(conv.out_channels), h=int(out_h), w=int(out_w))

    @classmethod
    def _shape_after_maxpool2d(cls, shape: _ShapeCHW, pool: nn.MaxPool2d) -> _ShapeCHW:
        k_h, k_w = cls._as_hw(pool.kernel_size)

        # stride defaults to kernel_size if not provided
        if pool.stride is None:
            s_h, s_w = k_h, k_w
        else:
            s_h, s_w = cls._as_hw(pool.stride)

        p_h, p_w = cls._as_hw(pool.padding)
        d_h, d_w = cls._as_hw(pool.dilation)

        out_h = (shape.h + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        out_w = (shape.w + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"MaxPool2d produced non-positive shape: H={out_h}, W={out_w}")

        return _ShapeCHW(c=int(shape.c), h=int(out_h), w=int(out_w))
