#!/usr/bin/env python3
"""Entrena DeepLabV3Plus con ResNet101."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_individual import train_single_model

if __name__ == "__main__":
    train_single_model(
        arch="deeplabv3plus",
        encoder="resnet101",
        name="DeepLabV3Plus_ResNet101",
        epochs=20
    )
