#!/usr/bin/env python3
"""Evaluacion comparativa de modelos de segmentacion de heridas — SOLO CONSOLA.

Evalua U-Net, Attention U-Net, Nested U-Net y DeepLabV3+ sobre el test set.
Imprime matrices de confusion (TP/TN/FP/FN), tabla de metricas, ranking clinico
y justificacion regulatoria. NO genera archivos .md ni .png.

Uso:
    python scripts/eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import ComparativeConfig
from src.datasets.wound_dataset import create_dataset_from_csv, get_default_transforms
from src.metrics.confusion import ConfusionMatrix


# ================================================================== #
# RENDERERS — pura consola, cero archivos
# ================================================================== #

def bar(metric: float, max_val: float = 1.0, width: int = 20) -> str:
    """Barra ASCII proporcional."""
    filled = int(metric / max_val * width) if max_val > 0 else 0
    return "|" + "#" * filled + "-" * (width - filled) + "|"


def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def print_confusion_matrix(cm: ConfusionMatrix, model_name: str, total_pixels: int) -> dict:
    """Imprime matriz de confusion en ASCII. Retorna metricas."""
    m = cm.derive_metrics()

    print()
    print(f"  >>> {model_name}")
    print(f"  +{'-'*30}+{'-'*22}+{'-'*22}+")
    print(f"  | {'':28} | {'PRED: NEGATIVO':>20} | {'PRED: POSITIVO':>20} |")
    print(f"  +{'-'*30}+{'-'*22}+{'-'*22}+")

    tn_pct = cm.tn / total_pixels * 100 if total_pixels else 0
    fp_pct = cm.fp / total_pixels * 100 if total_pixels else 0
    print(f"  | {'REAL: NEGATIVO (sano)':<28} | TN={cm.tn:>12,} ({tn_pct:5.1f}%) | FP={cm.fp:>12,} ({fp_pct:5.1f}%) |")
    print(f"  +{'-'*30}+{'-'*22}+{'-'*22}+")

    fn_pct = cm.fn / total_pixels * 100 if total_pixels else 0
    tp_pct = cm.tp / total_pixels * 100 if total_pixels else 0
    print(f"  | {'REAL: POSITIVO (herida)':<28} | FN={cm.fn:>12,} ({fn_pct:5.1f}%) | TP={cm.tp:>12,} ({tp_pct:5.1f}%) |")
    print(f"  +{'-'*30}+{'-'*22}+{'-'*22}+")

    print(f"  Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  "
          f"Prec={m['precision']:.4f}  NPV={m['npv']:.4f}")
    print(f"  Dice={m['dice']:.4f}  IoU={m['iou']:.4f}  "
          f"F2={m['f2']:.4f}  Acc={m['accuracy']:.4f}")
    return m


def print_comparison_table(results: dict[str, dict]) -> None:
    """Imprime tabla comparativa con barras ASCII."""
    print_header("TABLA COMPARATIVA DE METRICAS")
    print(f"  {'Modelo':<20} {'Dice':>8} {'Sens':>8} {'Spec':>8} {'F2':>8} {'NPV':>8}  Barras (Dice)")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*22}")

    # Ordenar por F2-score descendente
    sorted_models = sorted(results.items(), key=lambda x: x[1]["metrics"]["f2"], reverse=True)
    for name, data in sorted_models:
        m = data["metrics"]
        print(f"  {name:<20} {m['dice']:8.4f} {m['sensitivity']:8.4f} "
              f"{m['specificity']:8.4f} {m['f2']:8.4f} {m['npv']:8.4f}  "
              f"{bar(m['dice'])}")


def print_clinical_ranking(results: dict[str, dict]) -> None:
    """Imprime ranking clinico con justificacion regulatoria."""
    print_header("RANKING CLINICO (F2-Score — prioriza minimizar Falsos Negativos)")

    sorted_models = sorted(results.items(), key=lambda x: x[1]["metrics"]["f2"], reverse=True)

    medals = ["1ro (GANADOR)", "2do", "3ro", "4to"]
    for rank, (name, data) in enumerate(sorted_models):
        m = data["metrics"]
        medal = medals[rank] if rank < len(medals) else f"{rank+1}ro"
        cm = data["confusion"]

        print(f"\n  #{rank+1} {medal}: {name}")
        print(f"    F2-Score = {m['f2']:.4f}  |  Dice = {m['dice']:.4f}")
        print(f"    Sensibilidad = {m['sensitivity']:.4f}  (detecta el {m['sensitivity']*100:.1f}% de herida)")
        print(f"    Especificidad = {m['specificity']:.4f}  (solo {100-m['specificity']*100:.1f}% falsos positivos)")
        print(f"    Falsos Negativos = {cm.fn:,} px  |  Falsos Positivos = {cm.fp:,} px")

    # Justificacion
    best_name, best_data = sorted_models[0]
    best_m = best_data["metrics"]
    best_cm = best_data["confusion"]

    print_header("JUSTIFICACION CLINICA (FDA 510(k))")
    print(f"""
    Modelo recomendado: {best_name}

    FUNDAMENTO: El F2-score (beta=2) penaliza 4 veces mas los Falsos
    Negativos (FN = no detectar herida) que los Falsos Positivos
    (FP = marcar piel sana como herida). En contexto clinico, pasar
    por alto tejido de herida es mucho mas peligroso que una falsa
    alarma, porque puede retrasar el tratamiento.

    {best_name} obtuvo el F2-score mas alto ({best_m['f2']:.4f}) gracias a:
      - Alta sensibilidad ({best_m['sensitivity']:.4f}): detecta el
        {best_m['sensitivity']*100:.1f}% de los pixeles de herida real.
      - Buena especificidad ({best_m['specificity']:.4f}): solo
        {100-best_m['specificity']*100:.1f}% de falsos positivos.
      - Solo {best_cm.fn:,} pixeles de herida NO detectados (FN).
      - Solo {best_cm.fp:,} pixeles de piel sana marcados como herida (FP).

    COMPARACION CON OTROS MODELOS:""")

    for name, data in sorted_models[1:]:
        m2 = data["metrics"]
        cm2 = data["confusion"]
        print(f"      - {name}: F2={m2['f2']:.4f} | "
              f"FN={cm2.fn:,} px perdidos | FP={cm2.fp:,} px falsos")

    print(f"""
    THRESHOLDS FDA 510(k):
      Sensibilidad >= 0.90  -->  {best_m['sensitivity']:.4f}  {'PASA' if best_m['sensitivity']>=0.90 else 'NO PASA (necesita mas entrenamiento)'}
      Especificidad >= 0.95 -->  {best_m['specificity']:.4f}  {'PASA' if best_m['specificity']>=0.95 else 'NO PASA'}
      Dice >= 0.85          -->  {best_m['dice']:.4f}  {'PASA' if best_m['dice']>=0.85 else 'NO PASA'}
      NPV >= 0.95           -->  {best_m['npv']:.4f}  {'PASA' if best_m['npv']>=0.95 else 'NO PASA'}

    NOTA: 10 epocas a 128x128 es un entrenamiento rapido. Con 50-100
    epocas a 256x256 los resultados mejoraran significativamente.
    """)


# ================================================================== #
# EVALUADOR — sin generar archivos
# ================================================================== #

def modelo_path(model_name: str) -> Path:
    """Devuelve la ruta al modelo entrenado. Simple y directo."""
    return PROJECT_ROOT / "models" / f"{model_name}_final.pth"


def load_checkpoint(model, checkpoint_path: Path, device: str = "cpu") -> None:
    """Carga pesos desde un archivo .pth con key 'model_state_dict'."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)


def create_model_from_registry(name: str):
    """Crea modelo desde la factory, manejando diferencias de API."""
    from src.models.factory import create_model
    try:
        return create_model(name, pretrained=False)
    except TypeError:
        return create_model(name)


def evaluate_one_model(model_name: str, checkpoint_path: Path,
                       data_loader: DataLoader, device: str,
                       threshold: float) -> ConfusionMatrix | None:
    """Evalua un modelo y devuelve su matriz de confusion."""
    from src.inference.predictor import Predictor

    try:
        model = create_model_from_registry(model_name)
    except Exception as e:
        print(f"  ERROR creando {model_name}: {e}")
        return None

    load_checkpoint(model, checkpoint_path, device)
    predictor = Predictor(model, device=device)
    cm = ConfusionMatrix()

    with torch.inference_mode():
        for batch in data_loader:
            if isinstance(batch, dict):
                images, masks = batch["image"], batch["mask"]
            else:
                images, masks = batch
            images = images.to(device)
            preds = predictor.predict_batch(images)
            cm.accumulate(preds.cpu(), masks.cpu(), threshold)

    return cm


# ================================================================== #
# MAIN
# ================================================================== #

def main() -> None:
    config = ComparativeConfig()

    # --- Datos ---
    df = pd.read_csv(config.clean_csv)
    split = config.split
    for fallback in ["test", "val", "train"]:
        if fallback in df["split"].values:
            split = fallback
            break

    n_samples = len(df[df["split"] == split])
    total_pixels = n_samples * config.image_size[0] * config.image_size[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print_header("EVALUACION COMPARATIVA DE MODELOS")
    print(f"  Dataset: {config.clean_csv.name}  |  Split: {split} ({n_samples} muestras)")
    print(f"  Imagen: {config.image_size[0]}x{config.image_size[1]}  |  Total pixeles: {total_pixels:,}")
    print(f"  Device: {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # --- DataLoader ---
    transform = get_default_transforms(config.image_size)
    dataset = create_dataset_from_csv(
        csv_path=config.clean_csv, split=split,
        image_size=config.image_size, transform=transform,
    )
    data_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )

    # --- Evaluar modelos ---
    print_header("MATRICES DE CONFUSION (a nivel de pixel)")
    print(f"  Cada celda muestra: ETIQUETA = conteo (porcentaje del total)")
    print(f"  [TN, TP] = ACIERTOS    [FP, FN] = ERRORES")

    results: dict[str, dict] = {}

    for model_name in config.model_names:
        ckpt = modelo_path(model_name)
        if not ckpt.exists():
            print(f"\n  >>> {model_name}: SIN MODELO ({ckpt} no existe) — salteado")
            continue

        print(f"\n  Modelo: {ckpt}")
        cm = evaluate_one_model(model_name, ckpt, data_loader, device, config.binarize_threshold)
        if cm is None:
            continue

        metrics = print_confusion_matrix(cm, model_name, total_pixels)
        results[model_name] = {"confusion": cm, "metrics": metrics, "checkpoint": str(ckpt)}

    if not results:
        print("\n  Ningun modelo evaluado. Ejecuta primero: python scripts/4_train_models.py")
        return

    # --- Tabla y ranking ---
    print_comparison_table(results)
    print_clinical_ranking(results)

    print_header("LISTO")
    print(f"  Modelos evaluados: {len(results)}/{len(config.model_names)}")
    print(f"  Para re-entrenar:  python scripts/4_train_models.py")
    print(f"  Para repetir eval: python scripts/5_evaluate.py")
    print()


if __name__ == "__main__":
    main()
