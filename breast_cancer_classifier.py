# breast_cancer_classifier.py

from __future__ import annotations

import torch.nn as nn

from swan.swan_factory import SwanFactory
from swan.swan_model import SwanModel
from swan.swan_layer import (
    SwanConv2d,
    SwanReLU,
    SwanMaxPool2d,
    SwanFlatten,
    SwanLinear,
    SwanDropout,
)

class BreastCancerClassifier:
    """
    Canonical model definition for breast cancer image classification.

    This class is the single source of truth for:
      - architecture
      - input dimensions
      - number of classes

    Trainer, Tester, and any future inference tools must build models from here.
    """

    IMG_W = 224
    IMG_H = 224
    IMG_C = 3
    NUM_CLASSES = 2

    @classmethod
    def bake_swan(cls) -> SwanModel:
        layers = [
            SwanConv2d(out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            SwanReLU(),
            SwanMaxPool2d(kernel_size=2, stride=2),

            SwanConv2d(out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            SwanReLU(),
            SwanMaxPool2d(kernel_size=2, stride=2),

            SwanConv2d(out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            SwanReLU(),
            SwanMaxPool2d(kernel_size=2, stride=2),

            SwanFlatten(),
            SwanLinear(out_features=256, bias=True),
            SwanReLU(),
            SwanDropout(p=0.5),
            SwanLinear(out_features=cls.NUM_CLASSES, bias=True),
        ]

        return SwanFactory.build(
            input_image_width=cls.IMG_W,
            input_image_height=cls.IMG_H,
            input_image_channels=cls.IMG_C,
            layers=layers,
        )

    @classmethod
    def bake_torch(cls) -> nn.Module:
        """
        Convenience wrapper for inference / training code that only needs
        a torch.nn.Module.
        """
        swan = cls.bake_swan()
        return swan.to_torch_sequential()
