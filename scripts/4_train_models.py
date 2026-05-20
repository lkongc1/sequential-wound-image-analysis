#!/usr/bin/env python3
"""Entrena 4 modelos de segmentacion de heridas secuencialmente en GPU.

Guarda el modelo final entrenado en models/{nombre}_final.pth.
Sin checkpoints intermedios, sin subcarpetas. Simple y directo.

Uso:
    python scripts/4_train_models.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #
IMAGE_SIZE = (256, 256)
EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_WORKERS = 2
VAL_SPLIT = 0.2
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# (nombre, batch_size, vram_estimado)
MODELOS: list[tuple[str, int]] = [
    ("unet",            16),  # ~2.2 GB
    ("attention_unet",  16),  # ~0.2 GB
    ("deeplabv3",       12),  # ~2.0 GB
    ("nested_unet",     8),   # ~0.7 GB
]


# ------------------------------------------------------------------ #
# DATOS
# ------------------------------------------------------------------ #

def crear_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
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

    logger.info("Train: %d | Val: %d | Batch: %d", len(train_ds), len(val_ds), batch_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader


# ------------------------------------------------------------------ #
# ENTRENAMIENTO
# ------------------------------------------------------------------ #

def entrenar_modelo(nombre: str, train_loader: DataLoader, val_loader: DataLoader) -> Path:
    logger.info("=" * 55)
    logger.info("  ENTRENANDO: %s", nombre)
    logger.info("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("  GPU: %s (%.1f GB)", gpu, vram)

    # Modelo
    try:
        model = create_model(nombre, pretrained=False)
    except TypeError:
        model = create_model(nombre)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  Parametros: %s (%.1f M)", n_params, n_params / 1e6)

    # Trainer sin CheckpointManager ni EarlyStopping — entrena todo y guarda al final
    criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer_factory=lambda p: AdamW(p, lr=LEARNING_RATE),
        scheduler_factory=lambda o: ReduceLROnPlateau(o, mode="max", patience=3),
        device=device,
    )

    history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    if device == "cuda":
        torch.cuda.empty_cache()

    # Guardar modelo final
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"{nombre}_final.pth"
    torch.save({"model_state_dict": model.state_dict(), "history": history}, save_path)

    best_dice = max((h.get("val_dice", 0) for h in history), default=0)
    logger.info("  GUARDADO: %s  (Dice=%.4f)", save_path.name, best_dice)
    return save_path


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

def main() -> None:
    t0 = time.time()

    modelos_str = ", ".join(f"{n}(bs={b})" for n, b in MODELOS)
    logger.info("=" * 55)
    logger.info("  ENTRENAMIENTO MULTI-MODELO — Segmentation")
    logger.info("  Modelos: %s", modelos_str)
    logger.info("  Imagen: %s | Epocas: %d | VRAM max: ~6 GB", IMAGE_SIZE, EPOCHS)
    logger.info("  Guardado: models/<nombre>_final.pth")
    logger.info("=" * 55)

    for i, (nombre, batch_size) in enumerate(MODELOS, 1):
        t1 = time.time()
        train_loader, val_loader = crear_dataloaders(batch_size)
        try:
            path = entrenar_modelo(nombre, train_loader, val_loader)
            logger.info("  [%d/%d] %s -> listo en %.1f min", i, len(MODELOS), nombre, (time.time() - t1) / 60)
        except Exception as e:
            logger.error("  [%d/%d] %s FALLO: %s", i, len(MODELOS), nombre, e)

    logger.info("=" * 55)
    logger.info("  ENTRENAMIENTO COMPLETO — %.1f min", (time.time() - t0) / 60)
    logger.info("  Modelos guardados en: %s", MODELS_DIR.resolve())
    logger.info("  Siguiente paso: python scripts/5_evaluate.py")
    logger.info("=" * 55)

    # Limpiar checkpoints temporales que el Trainer genera por defecto
    import shutil
    tmp_ckpt = Path("models/checkpoints")
    if tmp_ckpt.exists():
        shutil.rmtree(tmp_ckpt)


if __name__ == "__main__":
    main()
