#!/usr/bin/env python3
"""Entrena U-Net con encoder ResNet + pesos ImageNet para segmentacion de heridas.

Proyecto: Deteccion y clasificacion de los tipos de heridas mediante
           tecnicas de vision computacional.

Estrategia:
  - TverskyLoss (a=0.5, b=0.5): balance precision/sensibilidad
  - CosineAnnealingWarmRestarts: reinicios cada 30 epocas
  - Augmentacion pesada: ElasticTransform, GridDistortion, HSV, GaussNoise

Uso:
    python scripts/training/6_train_unet.py                        # ResNet50 (384x384, batch 16)
    python scripts/training/6_train_unet.py --encoder resnet18     # ResNet18 (512x512, batch 24)
    python scripts/training/6_train_unet.py --epochs 100 --lr 1e-4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.datasets.wound_dataset import WoundDataset, get_default_transforms, get_training_transforms
from src.losses.tversky_loss import TverskyLoss
from src.models.factory import create_model
from src.training.trainer import Trainer


ENCODER_PRESETS = {
    "resnet50": {
        "image_size": (384, 384),
        "batch_size": 16,
        "save_name": "unet_final_pretrained.pth",
        "description": "ResNet50 — 32x downsampling, optimo para heridas grandes/irregulares",
    },
    "resnet18": {
        "image_size": (512, 512),
        "batch_size": 24,
        "save_name": "unet_r18_pretrained.pth",
        "description": "ResNet18 — 16x downsampling, preserva estructuras finas/lineales",
    },
}


class SchedulerWrapper:
    """Adapta CosineAnnealingWarmRestarts a la interfaz step(val_dice) del Trainer."""

    def __init__(self, scheduler):
        self._scheduler = scheduler
        self._epoch = 0

    def step(self, _metric: float | None = None) -> None:
        self._scheduler.step(self._epoch)
        self._epoch += 1

    def state_dict(self):
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self._scheduler.load_state_dict(state_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena U-Net con encoder ResNet + ImageNet")
    parser.add_argument("--encoder", choices=["resnet50", "resnet18"], default="resnet50",
                        help="Encoder backbone (default: resnet50)")
    parser.add_argument("--epochs", type=int, default=150, help="Numero de epocas (default: 150)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--tversky-alpha", type=float, default=0.5, dest="tversky_alpha")
    parser.add_argument("--tversky-beta", type=float, default=0.5, dest="tversky_beta")
    parser.add_argument("--T0", type=int, default=30, help="Reinicio del scheduler cada N epocas")
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV de dataset (default: data/processed/dataset_final.csv)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Ruta de salida del modelo (default: models/<encoder>.pth)")
    return parser.parse_args()


def setup_logging(save_name: str) -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{Path(save_name).stem}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def crear_dataloaders(csv_path: Path, image_size: tuple, batch_size: int, val_split: float = 0.2) -> tuple:
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"]
    imagenes = [Path(p) for p in train_df["image_path"]]
    mascaras = [Path(p) for p in train_df["mask_path"]]

    idx = list(range(len(imagenes)))
    train_idx, val_idx = train_test_split(idx, test_size=val_split, random_state=42)

    train_ds = WoundDataset(
        [imagenes[i] for i in train_idx], [mascaras[i] for i in train_idx],
        transform=get_training_transforms(image_size),
    )
    val_ds = WoundDataset(
        [imagenes[i] for i in val_idx], [mascaras[i] for i in val_idx],
        transform=get_default_transforms(image_size),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4),
    )


def main() -> None:
    args = parse_args()
    preset = ENCODER_PRESETS[args.encoder]
    csv_path = args.csv or PROJECT_ROOT / "data" / "processed" / "dataset_final.csv"
    save_path = args.output or PROJECT_ROOT / "models" / preset["save_name"]

    logger = setup_logging(preset["save_name"])
    logger.info("=" * 60)
    logger.info("  ENTRENAMIENTO U-Net — %s PRETRAINED", args.encoder.upper())
    logger.info("  %s", preset["description"])
    logger.info("=" * 60)
    logger.info("  Imagen: %s | Epocas: %d | Batch: %d | LR: %.0e",
                preset["image_size"], args.epochs, preset["batch_size"], args.lr)
    logger.info("  Loss: TverskyLoss (a=%.1f FP, b=%.1f FN) — BALANCEADO",
                args.tversky_alpha, args.tversky_beta)
    logger.info("  Scheduler: CosineAnnealingWarmRestarts (T_0=%d)", args.T0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s (%.1f GB)", gpu, vram)

    train_loader, val_loader = crear_dataloaders(csv_path, preset["image_size"], preset["batch_size"])
    logger.info("Train: %d | Val: %d", len(train_loader.dataset), len(val_loader.dataset))

    model = create_model("unet", encoder_name=args.encoder, pretrained=True)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Parametros: %.1f M", n_params / 1e6)

    criterion = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer_factory=lambda p: AdamW(p, lr=args.lr),
        scheduler_factory=lambda o: SchedulerWrapper(
            CosineAnnealingWarmRestarts(o, T_0=args.T0, T_mult=1, eta_min=1e-6),
        ),
        device=device,
    )

    t_start = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
    t_total = (time.time() - t_start) / 60

    if device == "cuda":
        torch.cuda.empty_cache()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": {
            "encoder": args.encoder,
            "image_size": preset["image_size"],
            "batch_size": preset["batch_size"],
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "pretrained": True,
            "loss": "TverskyLoss",
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
            "scheduler": "CosineAnnealingWarmRestarts",
            "T_0": args.T0,
        },
    }, save_path)

    best_dice = max((h.get("val_dice", 0) for h in history), default=0)
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO COMPLETO en %.1f min", t_total)
    logger.info("Mejor Dice (val): %.4f", best_dice)
    logger.info("Modelo guardado en: %s", save_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
