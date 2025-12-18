# convolve_kernel_alignment.py

from __future__ import annotations
from enum import Enum, auto

class ConvolveKernelAlignment(Enum):
    """
    Defines which kernel element is treated as the anchor for coordinate mapping.

    CENTER uses integer center:
        cx = kw // 2
        cy = kh // 2

    Corner modes treat the specified corner tap as the anchor.
    """
    CENTER = auto()

    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()

    @staticmethod
    def anchor(alignment: "ConvolveKernelAlignment", kw: int, kh: int) -> tuple[int, int]:
        """
        Return (ax, ay) kernel tap indices (0..kw-1, 0..kh-1) used as the anchor.
        """
        kw = int(kw)
        kh = int(kh)

        if alignment == ConvolveKernelAlignment.TOP_LEFT:
            return (0, 0)
        if alignment == ConvolveKernelAlignment.TOP_RIGHT:
            return (max(0, kw - 1), 0)
        if alignment == ConvolveKernelAlignment.BOTTOM_LEFT:
            return (0, max(0, kh - 1))
        if alignment == ConvolveKernelAlignment.BOTTOM_RIGHT:
            return (max(0, kw - 1), max(0, kh - 1))
        
        return (kw // 2, kh // 2)
