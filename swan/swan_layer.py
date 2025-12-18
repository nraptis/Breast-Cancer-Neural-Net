# swan_layer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, runtime_checkable

@runtime_checkable
class SwanLayer(Protocol):
    """
    A tiny interface for Swan layer specs.

    Goals:
      - serialize to a JSON-friendly dict (for run logging)
      - convert to an actual torch.nn.Module (for model construction)
      - pretty-print (for console summaries)

    Supported layer types (for now):
      - nn.Conv2d
      - nn.ReLU
      - nn.MaxPool2d
      - nn.Dropout
      - nn.Linear
    """

    def to_json(self) -> Dict[str, Any]:
        ...

    def to_torch_module(self) -> "Any":
        ...

    def to_pretty_print(self) -> str:
        ...


# -------------------------------------------------------------------
# Concrete Swan layers
# -------------------------------------------------------------------

@dataclass(frozen=True)
class SwanConv2d:
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    dilation: int = 1
    groups: int = 1
    bias: bool = True

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "conv2d",
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
        }

    def to_torch_module(self) -> Any:
        # Input channels are unknown until factory wiring time.
        # SwanFactory will patch/instantiate Conv2d with the correct in_channels.
        raise NotImplementedError(
            "SwanConv2d.to_torch_module() requires in_channels. "
            "Use SwanFactory to build the model."
        )

    def to_pretty_print(self) -> str:
        return (
            f"SwanConv2d(out={self.out_channels}, k={self.kernel_size}, "
            f"s={self.stride}, p={self.padding}, d={self.dilation}, "
            f"g={self.groups}, bias={self.bias})"
        )


@dataclass(frozen=True)
class SwanReLU:
    inplace: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {"type": "relu", "inplace": self.inplace}

    def to_torch_module(self) -> Any:
        import torch.nn as nn
        return nn.ReLU(inplace=self.inplace)

    def to_pretty_print(self) -> str:
        return f"SwanReLU(inplace={self.inplace})"


@dataclass(frozen=True)
class SwanMaxPool2d:
    kernel_size: int = 2
    stride: int | None = None
    padding: int = 0
    dilation: int = 1
    ceil_mode: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "max_pool2d",
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "ceil_mode": self.ceil_mode,
        }

    def to_torch_module(self) -> Any:
        import torch.nn as nn
        return nn.MaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
        )

    def to_pretty_print(self) -> str:
        return (
            f"SwanMaxPool2d(k={self.kernel_size}, "
            f"s={self.stride if self.stride is not None else 'auto'}, "
            f"p={self.padding}, d={self.dilation}, ceil={self.ceil_mode})"
        )


@dataclass(frozen=True)
class SwanDropout:
    p: float = 0.5
    inplace: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"Dropout p must be in [0,1], got {self.p}")

    def to_json(self) -> Dict[str, Any]:
        return {"type": "dropout", "p": self.p, "inplace": self.inplace}

    def to_torch_module(self) -> Any:
        import torch.nn as nn
        return nn.Dropout(p=self.p, inplace=self.inplace)

    def to_pretty_print(self) -> str:
        return f"SwanDropout(p={self.p}, inplace={self.inplace})"


@dataclass(frozen=True)
class SwanLinear:
    out_features: int
    bias: bool = True

    def to_json(self) -> Dict[str, Any]:
        return {"type": "linear", "out_features": self.out_features, "bias": self.bias}

    def to_torch_module(self) -> Any:
        # in_features is unknown until factory wiring time.
        # SwanFactory will patch/instantiate Linear with the correct in_features.
        raise NotImplementedError(
            "SwanLinear.to_torch_module() requires in_features. "
            "Use SwanFactory to build the model."
        )

    def to_pretty_print(self) -> str:
        return f"SwanLinear(out={self.out_features}, bias={self.bias})"

@dataclass(frozen=True)
class SwanFlatten:
    """
    Flatten layer spec.

    Converts a tensor from (N, C, H, W) -> (N, C*H*W).

    Notes:
      - This layer has no parameters.
      - SwanFactory will insert one automatically before the first Linear
        if the user forgets, but including it explicitly is preferred
        for readability and reproducibility.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "flatten",
        }

    def to_torch_module(self) -> Any:
        import torch.nn as nn
        return nn.Flatten()

    def to_pretty_print(self) -> str:
        return "SwanFlatten()"