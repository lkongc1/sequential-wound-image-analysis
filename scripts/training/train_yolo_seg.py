#!/usr/bin/env python3
"""Entrena un modelo YOLO11n-seg para segmentacion de instancias de heridas.

Requiere ultralytics instalado y datos preparados con:
    python scripts/pipeline/8_prepare_yolo.py --segment

El modelo nano (yolo11n-seg.pt) se entrena con tamaño 640 y batch=16,
diseñado para caber en ~8 GB de VRAM.

Uso:
    python scripts/training/train_yolo_seg.py
    python scripts/training/train_yolo_seg.py --epochs 200 --batch 8
    python scripts/training/train_yolo_seg.py --resume  # reanudar desde ultimo checkpoint
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "screening"
SEG_DIR = PROJECT_ROOT / "data" / "yolo" / "segment"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenar YOLO11n-seg para segmentacion de instancias de heridas",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-seg.pt",
        help="Modelo pre-entrenado o ruta a checkpoint (default: yolo11n-seg.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(SEG_DIR / "data.yaml"),
        help="Ruta al archivo data.yaml (default: data/yolo/segment/data.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximo de epocas (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Tamano del batch (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamano de imagen de entrada (default: 640)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Numero de workers de dataloader (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Dispositivo (default: 0). Usar 'cpu' si no hay GPU.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "segment"),
        help="Directorio raiz de proyectos de entrenamiento",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="wound_seg",
        help="Nombre del experimento (default: wound_seg)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar entrenamiento desde el ultimo checkpoint en project/name",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        default=True,
        help="Sobrescribir directorio existente (default: True)",
    )
    args = parser.parse_args()

    # Ensure output model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Validate data.yaml exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] data.yaml no encontrado: {data_path}", file=sys.stderr)
        print("        Ejecuta primero: python scripts/pipeline/8_prepare_yolo.py --segment", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("YOLO11n-seg Training — Instance Segmentation")
    print("=" * 60)
    print(f"Modelo:    {args.model}")
    print(f"Data:      {args.data}")
    print(f"Epocas:    {args.epochs}")
    print(f"Patience:  {args.patience}")
    print(f"Batch:     {args.batch}")
    print(f"Imgsz:     {args.imgsz}")
    print(f"Workers:   {args.workers}")
    print(f"Device:    {args.device}")
    print(f"Project:   {args.project}")
    print(f"Name:      {args.name}")
    print(f"Seed:      {args.seed}")
    print(f"Resume:    {args.resume}")
    print()

    # ── Cargar modelo ──────────────────────────────────────────────
    print("Cargando modelo...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    # ── Entrenar ────────────────────────────────────────────────────
    print("Iniciando entrenamiento...")
    print("(Ctrl+C para detener gracefulmente — el mejor modelo hasta ahora se guardara)")
    print()

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            save=True,
            save_period=10,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            resume=args.resume,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUMPIDO] Ctrl+C detectado. Guardando mejor modelo hasta ahora...")
        _save_best(args.project, args.name)
        print("Entrenamiento interrumpido por el usuario.")
        sys.exit(0)

    # ── Copiar mejor modelo ────────────────────────────────────────
    _save_best(args.project, args.name)

    # ── Mostrar metricas finales ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Metricas finales:")
    print("=" * 60)

    if results and hasattr(results, "results_dict"):
        rd = results.results_dict
        mAP50 = rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", "N/A"))
        mAP5095 = rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50-95", "N/A"))
        print(f"  mAP@0.5:     {mAP50}")
        print(f"  mAP@0.5:0.95 {mAP5095}")
    else:
        print("  (metricas disponibles en tensorboard / runs/segment/wound_seg)")

    print(f"\nModelo guardado en: {MODEL_DIR / 'yolo11_seg_best.pt'}")
    print("Entrenamiento completado.")


def _save_best(project_dir: str, exp_name: str) -> None:
    """Copy best.pt from training run to models/screening/."""
    best_src = Path(project_dir) / exp_name / "weights" / "best.pt"
    if best_src.exists():
        dst = MODEL_DIR / "yolo11_seg_best.pt"
        import shutil
        shutil.copy2(str(best_src), str(dst))
        print(f"  → Mejor modelo copiado a: {dst}")
    else:
        print(f"  [WARNING] No se encontro best.pt en {best_src}")


if __name__ == "__main__":
    main()
