#!/usr/bin/env python3
"""Entrena U-Net ResNet50 PRETRAINED — modelo final de segmentacion de heridas.

Proyecto: Deteccion y clasificacion de los tipos de heridas mediante
           tecnicas de vision computacional.

Fase actual: DETECCION / SEGMENTACION (modelo ganador del benchmark).

Este script entrena SOLO U-Net con encoder ResNet50 + pesos ImageNet.
Es el modelo que obtuvo el mejor F2-Score y Dice (0.8927) en
la evaluacion comparativa de 4 arquitecturas.

Estrategia de mejora (v4 — mascaras limpias):
  - TverskyLoss (a=0.5, b=0.5): balance precision/sensibilidad — evita falsos positivos
  - CosineAnnealingWarmRestarts: reinicios cada 30 epocas
  - Augmentacion pesada: ElasticTransform, GridDistortion, HSV, GaussNoise
    -> desensibiliza contra manchas de sangre, costras, suturas, tonos rojizos de piel
  - 384x384, batch=16, 150 epocas

Objetivo: Mascaras limpias sin falsos positivos (manchas rojas, dedos, ruido de textura).

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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.datasets.wound_dataset import WoundDataset, get_default_transforms, get_training_transforms
from src.losses.tversky_loss import TverskyLoss
from src.models.factory import create_model
from src.training.trainer import Trainer


class SchedulerWrapper:
    """Wrapper que adapta CosineAnnealingWarmRestarts a la interfaz del Trainer.

    El Trainer llama scheduler.step(val_dice) — compatible con ReduceLROnPlateau.
    CosineAnnealingWarmRestarts espera step(epoch: int), no una métrica float.
    Este wrapper ignora la métrica y auto-incrementa el contador de épocas.
    """
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
# CONFIG — TverskyLoss + CosineAnnealing + augmentation para Sens >= 0.90
# ================================================================== #

IMAGE_SIZE = (384, 384)    # Mejor detalle espacial que 256x256
BATCH_SIZE = 16            # Batch probado: mejor generalización que 32
EPOCHS = 150               # 150 épocas con warm restarts cada 30
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
VAL_SPLIT = 0.2
T_0 = 30                   # Reinicio del scheduler cada 30 épocas
ETA_MIN = 1e-6             # LR mínimo tras cada ciclo

# Tversky: α=0.5 (FP), β=0.5 (FN) → balance precision/sensibilidad
TVERSKY_ALPHA = 0.5
TVERSKY_BETA = 0.5

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

    # Train: augmentation + resize + normalize
    train_transform = get_training_transforms(IMAGE_SIZE)
    # Val: solo resize + normalize (sin augmentation)
    val_transform = get_default_transforms(IMAGE_SIZE)

    train_ds = WoundDataset(
        [imagenes[i] for i in train_idx],
        [mascaras[i] for i in train_idx],
        transform=train_transform,
    )
    val_ds = WoundDataset(
        [imagenes[i] for i in val_idx],
        [mascaras[i] for i in val_idx],
        transform=val_transform,
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

    # TverskyLoss: α=0.5, β=0.5 → balance precision/sensibilidad
    # Diseñado para mascaras limpias sin falsos positivos
    criterion = TverskyLoss(alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA)
    logger.info("Loss: TverskyLoss (α=%.1f FP, β=%.1f FN) — BALANCEADO", TVERSKY_ALPHA, TVERSKY_BETA)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer_factory=lambda p: AdamW(p, lr=LEARNING_RATE),
        scheduler_factory=lambda o: SchedulerWrapper(
            CosineAnnealingWarmRestarts(o, T_0=T_0, T_mult=1, eta_min=ETA_MIN),
        ),
        device=device,
    )
    logger.info("Scheduler: CosineAnnealingWarmRestarts (T_0=%d, eta_min=%.0e)", T_0, ETA_MIN)

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
                "loss": "TverskyLoss",
                "tversky_alpha": TVERSKY_ALPHA,
                "tversky_beta": TVERSKY_BETA,
                "scheduler": "CosineAnnealingWarmRestarts",
                "T_0": T_0,
                "augmentation": True,
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
    logger.info("  ENTRENAMIENTO FINAL — U-Net ResNet50 PRETRAINED v4 (balanceado)")
    logger.info("  Proyecto: Deteccion y clasificacion de heridas")
    logger.info("  Fase: SEGMENTACION (mascaras limpias, sin falsos positivos)")
    logger.info("=" * 60)
    logger.info("  Imagen: %s | Epocas: %d | Batch: %d", IMAGE_SIZE, EPOCHS, BATCH_SIZE)
    logger.info("  Encoder: ResNet50 + pesos ImageNet")
    logger.info("  Loss: TverskyLoss (α=%.1f FP, β=%.1f FN) — BALANCEADO", TVERSKY_ALPHA, TVERSKY_BETA)
    logger.info("  Scheduler: CosineAnnealingWarmRestarts (T_0=%d)", T_0)
    logger.info("  Augmentation: ElasticTransform, GridDistortion, HSV, GaussNoise, Flips")
    logger.info("  Guardado: %s", SAVE_PATH)
    logger.info("=" * 60)

    train_loader, val_loader = crear_dataloaders()
    entrenar_unet(train_loader, val_loader)

    logger.info("Siguiente paso: python scripts/5_evaluate.py")


if __name__ == "__main__":
    main()
