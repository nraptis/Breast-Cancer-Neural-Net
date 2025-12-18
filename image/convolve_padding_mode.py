# convolve_padding_mode.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Union


# --------------------------------------------------
# Convolution padding modes (Swift-style ADT)
# --------------------------------------------------

@dataclass(frozen=True)
class ConvolvePaddingSame:
    """Torch-style SAME padding (offset_x = offset_y = 0)."""
    pass


@dataclass(frozen=True)
class ConvolvePaddingValid:
    """VALID padding (no padding, kernel must fully fit)."""
    pass


@dataclass(frozen=True)
class ConvolvePaddingOffsetSame:
    """
    SAME padding, but expanded to safely accommodate offsets.

    max_offset_x / max_offset_y specify the maximum absolute
    offset that the padding must support without OOB access.
    """
    max_offset_x: int
    max_offset_y: int


@dataclass(frozen=True)
class ConvolvePaddingOffsetValid:
    """
    VALID padding semantics with offset allowance.

    Offsets are permitted up to max_offset_x / max_offset_y,
    but no implicit padding is introduced beyond what is
    required to honor the offset bounds.
    """
    max_offset_x: int
    max_offset_y: int


# Public union type
ConvolvePaddingMode = Union[
    ConvolvePaddingSame,
    ConvolvePaddingValid,
    ConvolvePaddingOffsetSame,
    ConvolvePaddingOffsetValid,
]
