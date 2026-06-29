#!/usr/bin/env python3
"""Entrena UNet con SegFormer (mit_b2)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_individual import train_single_model

if __name__ == "__main__":
    train_single_model(
        arch="unet",
        encoder="mit_b2",
        name="UNet_SegFormer",
        epochs=20
    )
