#!/usr/bin/env python3
"""Evaluacion completa del modelo final U-Net ResNet50 PRETRAINED.

Metricas: Matriz de confusion, Accuracy, Precision, Recall, Specificity,
          F1-Score, F2-Score, Dice, IoU, NPV, ROC-AUC.

Uso:
    python scripts/7_evaluate_pretrained.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ComparativeConfig
from src.datasets.wound_dataset import create_dataset_from_csv, get_default_transforms
from src.inference.predictor import Predictor
from src.metrics.confusion import ConfusionMatrix
from src.models.factory import create_model

SEP = "=" * 78
SEP2 = "-" * 78

# ================================================================== #
# METRICAS
# ================================================================== #

def compute_metrics(cm: ConfusionMatrix) -> dict:
    """Calcula todas las metricas desde la matriz de confusion."""
    tp, tn, fp, fn = cm.tp, cm.tn, cm.fp, cm.fn
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "npv": npv, "f1": f1, "f2": f2,
        "dice": dice, "iou": iou,
    }


def compute_roc_auc(model, dataset, device: str, max_samples: int = 5000) -> float:
    """Calcula ROC-AUC muestreando pixeles del test set.
    
    Usa muestreo estratificado para no cargar 35M de pixeles en RAM.
    """
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(42)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    all_probs = []
    all_labels = []

    samples_per_image = max(1, max_samples // len(dataset))
    rng = np.random.RandomState(42)

    predictor = Predictor(model, device=device)

    with torch.inference_mode():
        for batch in loader:
            img = (batch["image"] if isinstance(batch, dict) else batch[0]).to(device)
            msk = (batch["mask"] if isinstance(batch, dict) else batch[1])
            pred = predictor.predict(img)  # (1, H, W) probabilidades
            pred_np = pred.cpu().numpy().flatten()
            msk_np = msk.cpu().numpy().flatten().astype(np.int32)

            # Muestrear pixeles de esta imagen
            n_pixels = len(pred_np)
            n_sample = min(samples_per_image, n_pixels)
            idx = rng.choice(n_pixels, size=n_sample, replace=False)
            all_probs.append(pred_np[idx])
            all_labels.append(msk_np[idx])

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return float(roc_auc_score(labels, probs))


# ================================================================== #
# RENDERERS
# ================================================================== #

def print_header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_confusion_matrix(m: dict) -> None:
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    total = m["total"]

    print_header("MATRIZ DE CONFUSION")
    print(f"  Total de pixeles evaluados: {total:,}")
    print()
    print(f"                 PRED NEG    PRED POS")
    print(f"  REAL NEG      {tn:>10,}   {fp:>10,}")
    print(f"  REAL POS      {fn:>10,}   {tp:>10,}")


def print_metrics_table(m: dict, roc_auc: float, elapsed: float) -> None:
    print_header("METRICAS DE RENDIMIENTO")

    metrics = [
        ("Accuracy",        m["accuracy"]),
        ("Precision",       m["precision"]),
        ("Recall",          m["recall"]),
        ("Specificity",     m["specificity"]),
        ("F1-Score",        m["f1"]),
        ("ROC-AUC",         roc_auc),
    ]

    for name, val in metrics:
        print(f"  {name:<16} {val:.4f}")

    print(f"\n  Tiempo de evaluacion: {elapsed:.1f} s")


# ================================================================== #
# MAIN
# ================================================================== #

def main() -> None:
    config = ComparativeConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = PROJECT_ROOT / "models" / "unet_final_pretrained.pth"

    t0 = time.time()

    # --- Cabecera ---
    print()
    print(SEP)
    print("  EVALUACION COMPLETA — U-Net ResNet50 PRETRAINED (ImageNet)")
    print("  Proyecto: Deteccion y clasificacion de heridas")
    print(SEP)
    print(f"  Modelo:   {model_path.name}")
    print(f"  Device:   {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"  Dataset:  dataset_final.csv")

    # --- Cargar modelo ---
    model = create_model("unet", pretrained=True)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()

    # --- Datos ---
    transform = get_default_transforms(config.image_size)
    dataset = create_dataset_from_csv(
        csv_path=config.clean_csv, split="test",
        image_size=config.image_size, transform=transform,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"  Test set: {len(dataset)} muestras  |  {config.image_size[0]}x{config.image_size[1]} px")
    print(f"  Total pixeles: {len(dataset) * config.image_size[0] * config.image_size[1]:,}")

    # --- Evaluar ---
    predictor = Predictor(model, device=device)
    cm = ConfusionMatrix()

    with torch.inference_mode():
        for batch in loader:
            imgs = batch["image"] if isinstance(batch, dict) else batch[0]
            msks = batch["mask"] if isinstance(batch, dict) else batch[1]
            preds = predictor.predict_batch(imgs.to(device))
            cm.accumulate(preds.cpu(), msks.cpu(), 0.5)

    m = compute_metrics(cm)

    # --- ROC-AUC ---
    print(f"\n  Calculando ROC-AUC...")
    roc_auc = compute_roc_auc(model, dataset, device)

    elapsed = time.time() - t0

    # --- Resultados ---
    print_confusion_matrix(m)
    print_metrics_table(m, roc_auc, elapsed)

    print()


if __name__ == "__main__":
    main()
