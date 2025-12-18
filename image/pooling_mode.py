# image/pooling_mode.py
from __future__ import annotations
from enum import Enum, auto

class PoolingMode(Enum):
    
    # ----------------------------------
    # Per-channel pooling (classic CNN)
    # ----------------------------------
    MAX_PER_CHANNEL = auto()     # max over window for each channel independently
    MIN_PER_CHANNEL = auto()
    AVERAGE_PER_CHANNEL = auto()

    # ----------------------------------
    # Pixel-wise pooling (winner-take-all)
    # ----------------------------------
    MAX_PIXEL_BY_RGB_SUM = auto()  # argmax of (r + g + b), alpha ignored
    MIN_PIXEL_BY_RGB_SUM = auto()  # argmin of (r + g + b), alpha ignored
    