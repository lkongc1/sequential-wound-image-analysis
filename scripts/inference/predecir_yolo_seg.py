#!/usr/bin/env python3
"""Prediccion de segmentacion de instancias con YOLO11-seg.

Carga un modelo YOLO11-seg entrenado y genera:
  - Mascara de instancias (mapa uint16, 0=bg, 1..N=instancias)
  - Overlay por instancia con contornos coloreados + etiquetas + confianza
  - CSV con estadisticas por instancia (mismo formato que Phase 1 watershed)

Uso:
    python scripts/inference/predecir_yolo_seg.py imagen.jpg
    python scripts/inference/predecir_yolo_seg.py imagen.jpg --conf 0.5
    python scripts/inference/predecir_yolo_seg.py imagen.jpg --salida resultados/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.inference.postprocessing import compute_instance_stats

MODEL_PATH = PROJECT_ROOT / "models" / "screening" / "yolo11_seg_best.pt"
DEFAULT_CONF = 0.25
COLORES: list[tuple[int, int, int]] = [
    (255, 200, 0),     # verde azulado
    (0, 0, 255),       # rojo
    (255, 0, 0),       # azul
    (0, 255, 255),     # amarillo
    (0, 140, 255),     # naranja
    (255, 0, 255),     # magenta
    (255, 255, 0),     # amarillo brillante
    (0, 255, 255),     # cian
    (128, 0, 128),     # purpura
    (0, 255, 0),       # verde
]


def draw_instance_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    stats: list[dict],
    confidences: dict[int, float],
) -> np.ndarray:
    """Draw coloured contours + instance IDs + confidence on a copy of image.

    Args:
        image: BGR image (H, W, 3).
        instance_map: uint16 map (0=bg, 1..N=instances).
        stats: Per-instance stats list from compute_instance_stats().
        confidences: Mapping from instance_id -> confidence score.

    Returns:
        Overlay BGR image.
    """
    overlay = image.copy()
    for inst in stats:
        iid = inst["instance_id"]
        inst_mask = (instance_map == iid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = COLORES[(iid - 1) % len(COLORES)]
        # Draw filled contour with low alpha, then outline
        cv2.drawContours(overlay, contours, -1, color, -1)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

        # Label: "ID (conf%)" at centroid
        cx, cy = int(inst["centroid_x"]), int(inst["centroid_y"])
        conf = confidences.get(iid, 0.0)
        label = f"{iid} ({conf:.0%})"
        cv2.putText(
            overlay, label, (cx - 5, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, label, (cx - 5, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA,
        )

    # Blend with original
    overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predecir segmentacion de instancias con YOLO11-seg",
    )
    parser.add_argument("imagen", type=Path, help="Ruta a la imagen a analizar")
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Umbral de confianza (default: {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--modelo", type=Path, default=MODEL_PATH,
        help=f"Ruta al modelo YOLO11-seg (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--salida", type=Path, default=None,
        help="Directorio de salida (default: junto a la imagen)",
    )
    parser.add_argument(
        "--guardar-mapa", action="store_true",
        help="Guardar tambien el mapa de instancias raw (uint16 PNG)",
    )
    args = parser.parse_args()

    if not args.imagen.exists():
        print(f"[ERROR] No se encontro: {args.imagen}", file=sys.stderr)
        sys.exit(1)

    if not args.modelo.exists():
        print(f"[ERROR] Modelo no encontrado: {args.modelo}", file=sys.stderr)
        print("        Ejecuta primero: python scripts/train_yolo_seg.py", file=sys.stderr)
        sys.exit(1)

    salida = args.salida or args.imagen.parent
    salida.mkdir(parents=True, exist_ok=True)
    nombre = args.imagen.stem

    print(f"Imagen:  {args.imagen}")
    print(f"Modelo:  {args.modelo.name}")
    print(f"Conf:    {args.conf}")
    print(f"Salida:  {salida}/")
    print()

    # ── Cargar modelo ──────────────────────────────────────────────
    print("Cargando modelo YOLO11-seg...")
    model = YOLO(str(args.modelo))

    # ── Leer imagen ────────────────────────────────────────────────
    print("Leyendo imagen...")
    original = cv2.imread(str(args.imagen))
    if original is None:
        print(f"[ERROR] No se pudo leer la imagen: {args.imagen}", file=sys.stderr)
        sys.exit(1)

    orig_h, orig_w = original.shape[:2]
    print(f"Resolucion: {orig_w}x{orig_h}")

    # ── Inferencia ─────────────────────────────────────────────────
    print("Ejecutando inferencia...")
    results = model(args.imagen, conf=args.conf, verbose=False)

    if len(results) == 0:
        print("  Sin resultados de deteccion.")
        # Save empty outputs
        empty_map = np.zeros((orig_h, orig_w), dtype=np.uint16)
        csv_path = salida / f"{nombre}_instancias_yolo.csv"
        csv_path.write_text("instance_id,area_px,area_pct,bbox_x,bbox_y,bbox_w,bbox_h,centroid_x,centroid_y\n",
                            encoding="utf-8")
        # Blank overlay (original only)
        cv2.imwrite(str(salida / f"{nombre}_instancias_yolo.png"), original)
        print(f"\nResultados guardados en: {salida}/")
        print(f"  {nombre}_instancias_yolo.csv   (vacio — 0 instancias)")
        print(f"  {nombre}_instancias_yolo.png   (sin detecciones)")
        return

    result = results[0]

    # ── Extraer instancias ─────────────────────────────────────────
    instance_map = np.zeros((orig_h, orig_w), dtype=np.uint16)
    confidences: dict[int, float] = {}
    instance_id = 0

    if result.masks is not None and result.boxes is not None:
        boxes = result.boxes
        masks = result.masks

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < args.conf:
                continue

            # Get binary mask (ultralytics returns normalized mask, resize to original)
            mask_np = masks.data[i].cpu().numpy().astype(np.float32)
            mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_resized > 0.5).astype(np.uint8)

            # Skip empty masks
            if mask_bin.sum() < 10:
                continue

            instance_id += 1
            instance_map[mask_bin > 0] = instance_id
            confidences[instance_id] = conf

    num_instances = instance_id
    print(f"  Instancias detectadas: {num_instances}")

    # ── Estadisticas por instancia ─────────────────────────────────
    stats = compute_instance_stats(instance_map, image_shape=(orig_h, orig_w))

    # ── Guardar mapa de instancias raw ─────────────────────────────
    if args.guardar_mapa:
        map_path = salida / f"{nombre}_instance_map.png"
        cv2.imwrite(str(map_path), instance_map)
        print(f"  Mapa de instancias: {map_path.name}")

    # ── Guardar overlay por instancia ──────────────────────────────
    overlay = draw_instance_overlay(original, instance_map, stats, confidences)
    overlay_path = salida / f"{nombre}_instancias_yolo.png"
    cv2.imwrite(str(overlay_path), overlay)

    # ── Guardar CSV ────────────────────────────────────────────────
    csv_path = salida / f"{nombre}_instancias_yolo.csv"
    header = "instance_id,area_px,area_pct,bbox_x,bbox_y,bbox_w,bbox_h,centroid_x,centroid_y"
    rows: list[str] = []
    for inst in stats:
        rows.append(
            f"{inst['instance_id']},{inst['area_px']},{inst['area_pct']},"
            f"{inst['bbox_x']},{inst['bbox_y']},{inst['bbox_w']},{inst['bbox_h']},"
            f"{inst['centroid_x']},{inst['centroid_y']}"
        )
    csv_path.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")

    # ── Reporte ────────────────────────────────────────────────────
    print(f"\nResultados guardados en: {salida}/")
    print(f"  {nombre}_instancias_yolo.png    (overlay por instancia)")
    print(f"  {nombre}_instancias_yolo.csv    (estadisticas por instancia)")
    if args.guardar_mapa:
        print(f"  {nombre}_instance_map.png     (mapa de instancias raw)")
    for inst in stats:
        iid = inst["instance_id"]
        conf = confidences.get(iid, 0.0)
        print(f"    [{iid}] conf={conf:.2f}  area={inst['area_px']}px  "
              f"bbox=({inst['bbox_x']},{inst['bbox_y']},{inst['bbox_w']}x{inst['bbox_h']})  "
              f"centroid=({inst['centroid_x']},{inst['centroid_y']})")


if __name__ == "__main__":
    main()
