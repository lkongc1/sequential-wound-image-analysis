#!/usr/bin/env python3
"""Entrena U-Net ResNet50 PRETRAINED — modelo final de segmentacion de heridas.

Proyecto: Deteccion y clasificacion de los tipos de heridas mediante
           tecnicas de vision computacional.

Fase actual: DETECCION / SEGMENTACION (modelo ganador del benchmark).

Este script entrena SOLO U-Net con encoder ResNet50 + pesos ImageNet.
Es el modelo que obtuvo el mejor F2-Score (0.9897) y Dice (0.8399) en
la evaluacion comparativa de 4 arquitecturas.

Uso:
    python scripts/6_train_unet_final.py

Salida:
    models/unet_final_pretrained.pth  — pesos del modelo entrenado
    logs/unet_final_YYYYMMDD_HHMMSS.log — bitacora de entrenamiento
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.datasets.wound_dataset import WoundDataset, get_default_transforms
from src.losses.dice_loss import BCEDiceLoss
from src.models.factory import create_model
from src.training.trainer import Trainer

# ------------------------------------------------------------------ #
# LOGGING
# ------------------------------------------------------------------ #

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"unet_final_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ================================================================== #
# CONFIG — ajustada para maximizar sensibilidad y Dice
# ================================================================== #

IMAGE_SIZE = (256, 256)    # Resolucion del benchmark
BATCH_SIZE = 16            # ~2 GB VRAM con AMP — 3× más rápido que batch=4
EPOCHS = 50                # Suficiente para converger con pretrained
LEARNING_RATE = 1e-3
NUM_WORKERS = 2
VAL_SPLIT = 0.2
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final.csv"
SAVE_PATH = PROJECT_ROOT / "models" / "unet_final_pretrained.pth"

# ================================================================== #
# DATOS
# ================================================================== #

def crear_dataloaders() -> tuple[DataLoader, DataLoader]:
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"]
    imagenes = [Path(p) for p in train_df["image_path"]]
    mascaras = [Path(p) for p in train_df["mask_path"]]

    idx = list(range(len(imagenes)))
    train_idx, val_idx = train_test_split(idx, test_size=VAL_SPLIT, random_state=42)

    transform = get_default_transforms(IMAGE_SIZE)
    train_ds = WoundDataset(
        [imagenes[i] for i in train_idx],
        [mascaras[i] for i in train_idx],
        transform=transform,
    )
    val_ds = WoundDataset(
        [imagenes[i] for i in val_idx],
        [mascaras[i] for i in val_idx],
        transform=transform,
    )

    logger.info("Train: %d | Val: %d | Batch: %d", len(train_ds), len(val_ds), BATCH_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader


# ================================================================== #
# ENTRENAMIENTO
# ================================================================== #

def entrenar_unet(train_loader: DataLoader, val_loader: DataLoader) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s (%.1f GB) | Device: %s", gpu, vram, device)

    # Modelo U-Net con encoder ResNet50 + pesos ImageNet
    logger.info("Creando U-Net ResNet50 PRETRAINED (ImageNet)...")
    model = create_model("unet", pretrained=True)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Parametros: %s (%.1f M)", n_params, n_params / 1e6)

    criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer_factory=lambda p: AdamW(p, lr=LEARNING_RATE),
        scheduler_factory=lambda o: ReduceLROnPlateau(o, mode="max", patience=5),
        device=device,
    )

    t_start = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)
    t_total = (time.time() - t_start) / 60

    if device == "cuda":
        torch.cuda.empty_cache()

    # Guardar modelo final
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "config": {
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "pretrained": True,
                "encoder": "resnet50",
            },
        },
        SAVE_PATH,
    )

    best_dice = max((h.get("val_dice", 0) for h in history), default=0)
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO COMPLETO en %.1f min", t_total)
    logger.info("Mejor Dice (val): %.4f", best_dice)
    logger.info("Modelo guardado en: %s", SAVE_PATH)
    logger.info("Log guardado en: %s", log_path)
    logger.info("=" * 60)

    return SAVE_PATH


# ================================================================== #
# MAIN
# ================================================================== #

def main() -> None:
    logger.info("=" * 60)
    logger.info("  ENTRENAMIENTO FINAL — U-Net ResNet50 PRETRAINED")
    logger.info("  Proyecto: Deteccion y clasificacion de heridas")
    logger.info("  Fase: SEGMENTACION (modelo ganador del benchmark)")
    logger.info("=" * 60)
    logger.info("  Imagen: %s | Epocas: %d | Batch: %d", IMAGE_SIZE, EPOCHS, BATCH_SIZE)
    logger.info("  Encoder: ResNet50 + pesos ImageNet | Loss: BCEDiceLoss")
    logger.info("  Guardado: %s", SAVE_PATH)
    logger.info("=" * 60)

    train_loader, val_loader = crear_dataloaders()
    entrenar_unet(train_loader, val_loader)

    logger.info("Siguiente paso: python scripts/5_evaluate.py")


if __name__ == "__main__":
    main()
