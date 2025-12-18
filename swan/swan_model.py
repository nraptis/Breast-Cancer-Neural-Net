# swan/swan_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch.nn as nn

from swan.swan_layer import SwanLayer

@dataclass(frozen=True)
class SwanModel:
    input_image_width: int
    input_image_height: int
    input_image_channels: int
    layers: List[SwanLayer]
    _torch: nn.Sequential

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "swan_model",
            "input": {
                "width": self.input_image_width,
                "height": self.input_image_height,
                "channels": self.input_image_channels,
            },
            "layers": [layer.to_json() for layer in self.layers],
        }

    def to_torch_sequential(self) -> nn.Sequential:
        return self._torch

    def to_pretty_print(self) -> str:
        lines: List[str] = []
        lines.append("SwanModel")
        lines.append(f"  input:  W={self.input_image_width} H={self.input_image_height} C={self.input_image_channels}")
        lines.append("  layers:")
        for i, layer in enumerate(self.layers):
            lines.append(f"    {i:02d}: {layer.to_pretty_print()}")
        lines.append("  torch:")
        lines.append(str(self._torch))
        return "\n".join(lines)
