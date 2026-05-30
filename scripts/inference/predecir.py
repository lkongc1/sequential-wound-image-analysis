#!/usr/bin/env python3
"""Prueba local del modelo de segmentacion de heridas.

Predice la mascara para una imagen y guarda:
  - original.png
  - mascara.png
  - mascara_cruda.png (sin filtros, para comparar)
  - superpuesto.png (overlay)

Por defecto conserva TODOS los blobs de herida (para heridas multiples).
Usa --solo-mayor si la imagen tiene UNA sola herida grande.
Usa --cruda para ver la mascara sin ningun filtro.

Uso:
    python scripts/inference/predecir.py imagen.png
    python scripts/inference/predecir.py imagen.png --umbral 0.6 --limpiar 3
    python scripts/inference/predecir.py imagen.png --solo-mayor   (una sola herida)
    python scripts/inference/predecir.py imagen.png --cruda  (sin filtros)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
from PIL import Image

from src.datasets.wound_dataset import get_default_transforms
from src.inference.postprocessing import compute_instance_stats, watershed_instances
from src.models.factory import create_model

MODEL_PATH = PROJECT_ROOT / "models" / "screening" / "FPN_EfficientNetB3_best.pth"
IMAGE_SIZE = (384, 384)     # Resolucion de entrenamiento — inferencia confiable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORM = get_default_transforms(IMAGE_SIZE)

# ── Model registry: filename_pattern -> (factory_arch_name, encoder_name) ──
# Uses the factory's create_model (which wraps SMP) for consistent loading.
# Match the training config in scripts/11_screening_architectures.py
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "FPN_EfficientNetB3":      ("fpn", "efficientnet-b3"),
    "FPN_ResNet101":           ("fpn", "resnet101"),
    "FPN_ResNeXt50":           ("fpn", "resnext50_32x4d"),
    "FPN_SegFormer":           ("fpn", "mit_b2"),
    "UNet_EfficientNetB3":     ("unet", "efficientnet-b3"),
    "UNet_ResNet101":          ("unet", "resnet101"),
    "UNet_ResNeXt50":          ("unet", "resnext50_32x4d"),
    "UNet_SegFormer":          ("unet", "mit_b2"),
    "DeepLabV3Plus_ResNet101": ("deeplabv3plus", "resnet101"),
    "DeepLabV3Plus_ResNeXt50": ("deeplabv3plus", "resnext50_32x4d"),
    "DeepLabV3Plus_EfficientNetB3": ("deeplabv3plus", "efficientnet-b3"),
    "DeepLabV3_ResNeXt50":     ("deeplabv3", "resnext50_32x4d"),
}

# Fallback for old naming
_MODEL_FALLBACK: dict[str, str] = {
    "unet": "unet",
}


def _parse_model_name(path: Path) -> tuple[str, str]:
    """Parse model filename into (factory_arch_name, encoder_name).

    E.g. 'FPN_EfficientNetB3_best.pth' -> ('fpn', 'efficientnet-b3')
    """
    stem = path.stem
    # Strip known sufffixes
    import re
    for suffix in ["_best", "_final", "_v3", "_v4"]:
        stem = stem.replace(suffix, "")
    stem = re.sub(r"_v\d+$", "", stem)

    if stem in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[stem]
    if stem in _MODEL_FALLBACK:
        return (_MODEL_FALLBACK[stem], "resnet50")
    print(f"[WARN] Modelo '{stem}' no reconocido, usando fpn+efficientnet-b3 como fallback")
    return ("fpn", "efficientnet-b3")


def cargar_modelo() -> torch.nn.Module:
    arch_name, encoder_name = _parse_model_name(MODEL_PATH)
    print(f"  Arquitectura: {arch_name}  Encoder: {encoder_name}")
    modelo = create_model(arch_name, encoder_name=encoder_name, pretrained=False)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    modelo.load_state_dict(ckpt["model_state_dict"], strict=False)
    modelo.to(DEVICE).eval()
    return modelo


def predecir(modelo: torch.nn.Module, imagen: np.ndarray, umbral: float = 0.5, tta: bool = False) -> np.ndarray:
    """Predice mascara binaria CRUDA para una imagen BGR (H, W, 3).
    
    Args:
        modelo: modelo PyTorch
        imagen: imagen BGR
        umbral: threshold de binarizacion
        tta: Test-Time Augmentation (4 flips, promedio) — +0.003 Dice, 4x mas lento
    """
    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    transformado = TRANSFORM(image=img_rgb)
    tensor = transformado["image"].unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        if tta:
            # Test-Time Augmentation: original + 3 flips
            preds = []
            preds.append(torch.sigmoid(modelo(tensor)))                                    # original
            preds.append(torch.sigmoid(modelo(torch.flip(tensor, [3]))).flip([3]))        # H-flip
            preds.append(torch.sigmoid(modelo(torch.flip(tensor, [2]))).flip([2]))        # V-flip
            flipped = torch.flip(torch.flip(tensor, [2]), [3])
            preds.append(torch.sigmoid(modelo(flipped)).flip([2]).flip([3]))              # both
            pred = torch.stack(preds).mean(dim=0).squeeze().cpu().numpy()
        else:
            pred = torch.sigmoid(modelo(tensor)).squeeze().cpu().numpy()

    mascara = (pred > umbral).astype(np.uint8) * 255
    mascara = cv2.resize(mascara, (imagen.shape[1], imagen.shape[0]))
    return mascara


def limpiar_mascara(
    mascara: np.ndarray,
    suavizar: float = 0.0,
    kernel_apertura: int = 1,
    kernel_cierre: int = 7,
    area_min_pct: float = 0.005,
    solo_mayor: bool = False,
) -> np.ndarray:
    """Post-procesa la mascara para eliminar ruido y rellenar huecos.

    Args:
        mascara: mascara binaria cruda (0/255)
        suavizar: sigma del GaussianBlur (0 = sin suavizado)
        kernel_apertura: kernel de APERTURA (erosion->dilatacion). Elimina puntos aislados FUERA de la herida.
        kernel_cierre: kernel de CIERRE (dilatacion->erosion). Rellena huecos DENTRO de la herida
                       (manchas de sangre, costras, puntos de sutura).
        area_min_pct: % minimo del area de la imagen para conservar un blob
        solo_mayor: si True, conserva solo el componente conexo mas grande

    Returns:
        mascara limpia (0/255)
    """
    limpia = mascara.copy()

    # 1. Suavizado Gaussiano (reduce bordes dentados y ruido puntual)
    if suavizar > 0:
        ksize = int(2 * round(suavizar) + 1)
        limpia = cv2.GaussianBlur(limpia, (ksize, ksize), suavizar)

    # Re-binarizar post-suavizado (GaussianBlur genera valores intermedios)
    if suavizar > 0:
        _, limpia = cv2.threshold(limpia, 127, 255, cv2.THRESH_BINARY)

    # 2. CIERRE morfologico: dilatacion -> erosion
    #    Rellena huecos DENTRO de la herida (sangre seca, costras, suturas)
    if kernel_cierre >= 3:
        kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_cierre, kernel_cierre))
        limpia = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, kernel_c)

    # 3. APERTURA morfologica: erosion -> dilatacion
    #    Elimina ruido aislado FUERA de la herida
    if kernel_apertura >= 3:
        kernel_a = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_apertura, kernel_apertura))
        limpia = cv2.morphologyEx(limpia, cv2.MORPH_OPEN, kernel_a)

    # 4. Filtrar por area minima
    if area_min_pct > 0 or solo_mayor:
        _, binaria = cv2.threshold(limpia, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)

        if num_labels > 1:  # label 0 es el fondo
            altura, ancho = mascara.shape[:2]
            area_total = altura * ancho
            area_min_px = int(area_total * area_min_pct / 100)

            mascara_filtrada = np.zeros_like(limpia)

            if solo_mayor:
                # Conservar solo el componente mas grande (ignorando fondo)
                areas = stats[1:, cv2.CC_STAT_AREA]
                idx_mayor = int(np.argmax(areas)) + 1
                mascara_filtrada[labels == idx_mayor] = 255
            else:
                # Conservar componentes >= area minima
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= area_min_px:
                        mascara_filtrada[labels == i] = 255

            limpia = mascara_filtrada

    return limpia


def superponer(imagen: np.ndarray, mascara: np.ndarray, alpha: float = 0.4, color: tuple = (255, 200, 0)) -> np.ndarray:
    """Overlay sobre la region detectada.

    Args:
        imagen: imagen BGR original
        mascara: mascara binaria (0/255)
        alpha: transparencia del overlay (0=original, 1=solido)
        color: color BGR del overlay. Default: verde azulado (B=255, G=200, R=0)
    """
    overlay = imagen.copy()
    overlay[mascara > 0] = color
    return cv2.addWeighted(imagen, 1 - alpha, overlay, alpha, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probar segmentacion de heridas en una imagen local")
    parser.add_argument("imagen", type=Path, help="Ruta a la imagen a analizar")
    parser.add_argument("--umbral", type=float, default=0.5, help="Umbral de binarizacion (default: 0.5. Subir a 0.6-0.7 si detecta manchas rojas)")
    parser.add_argument("--suavizar", type=float, default=0.0, help="Sigma del GaussianBlur (0=sin filtro, default: 0.0)")
    parser.add_argument("--limpiar", type=int, default=1, help="Kernel de APERTURA (erosion->dilatacion). Elimina ruido externo. Usar 3-5.")
    parser.add_argument("--rellenar", type=int, default=7, help="Kernel de CIERRE (dilatacion->erosion). Rellena huecos de sangre/costras/suturas. Default: 7.")
    parser.add_argument("--area-min", type=float, default=0.005, help="%% minimo del area para conservar un blob (default: 0.005)")
    parser.add_argument("--solo-mayor", action="store_true", help="Conservar SOLO el componente conexo mas grande (para 1 herida)")
    parser.add_argument("--tta", action="store_true", help="Test-Time Augmentation (4 flips, +precision, mas lento)")
    parser.add_argument("--color", type=str, default="verde", choices=["verde", "rojo", "azul", "amarillo", "naranja"],
                        help="Color del overlay (default: verde)")
    parser.add_argument("--cruda", action="store_true", help="Mostrar mascara sin filtros (sin limpiar)")
    parser.add_argument("--instancias", action="store_true", help="Separar instancias de heridas via watershed (overlay + CSV)")
    parser.add_argument("--sam", action="store_true", help="Separar instancias via SAM2 (mas preciso, separa heridas pegadas)")
    parser.add_argument("--sam-model", type=str, default="sam2.1_hiera_l", choices=["sam2.1_hiera_l", "sam2.1_hiera_b+", "sam2.1_hiera_s"],
                        help="Modelo SAM2 (default: sam2.1_hiera_l = mas preciso)")
    parser.add_argument("--salida", type=Path, default=None, help="Directorio de salida (default: junto a la imagen)")
    args = parser.parse_args()

    if not args.imagen.exists():
        print(f"[ERROR] No se encontro: {args.imagen}", file=sys.stderr)
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Modelo no encontrado: {MODEL_PATH}", file=sys.stderr)
        print(f"        Ejecuta primero: python scripts/6_train_unet_final.py", file=sys.stderr)
        sys.exit(1)

    salida = args.salida or args.imagen.parent
    salida.mkdir(parents=True, exist_ok=True)

    nombre = args.imagen.stem

    print(f"Imagen:  {args.imagen}")
    print(f"Modelo:  {MODEL_PATH.name}")
    print(f"Device:  {DEVICE.upper()}")
    print(f"Umbral:  {args.umbral}")
    print(f"Color:   {args.color}")
    if not args.cruda:
        print(f"Filtros: umbral={args.umbral}  cierre={args.rellenar}  apertura={args.limpiar}  area-min={args.area_min}%  solo-mayor={args.solo_mayor}")
    else:
        print(f"Filtros: NINGUNO (--cruda)")
    print()

    print("Cargando modelo...")
    modelo = cargar_modelo()

    print("Leyendo imagen...")
    original = cv2.imread(str(args.imagen))
    if original is None:
        print(f"[ERROR] No se pudo leer la imagen: {args.imagen}", file=sys.stderr)
        sys.exit(1)

    print(f"Resolucion original: {original.shape[1]}x{original.shape[0]}")

    print("Prediciendo mascara...")
    mascara_cruda = predecir(modelo, original, args.umbral, tta=args.tta)

    if args.cruda:
        mascara_final = mascara_cruda
    else:
        print("Aplicando filtros de limpieza...")
        mascara_final = limpiar_mascara(
            mascara_cruda,
            suavizar=args.suavizar,
            kernel_apertura=args.limpiar,
            kernel_cierre=args.rellenar,
            area_min_pct=args.area_min,
            solo_mayor=args.solo_mayor,
        )

    herida_cruda_pct = (mascara_cruda > 0).sum() / mascara_cruda.size * 100
    herida_final_pct = (mascara_final > 0).sum() / mascara_final.size * 100
    print(f"Herida detectada: {herida_cruda_pct:.2f}% (cruda) -> {herida_final_pct:.2f}% (limpia)")

    # ── Instancia (watershed o SAM2) ────────────────────────────────
    instancias_stats = None
    instancias_map = None
    if args.instancias:
        print("[INSTANCIAS] Modo instancias activo — separacion por watershed")
        instancias_map = watershed_instances(
            mascara_final,
            dist_threshold_ratio=0.3,
            min_instance_area=50,
            kernel_close=3,
        )
        instancias_stats = compute_instance_stats(instancias_map, image_shape=original.shape[:2])
        num_wounds = len(instancias_stats)
        print(f"[INSTANCIAS] {num_wounds} herida(s) identificada(s)")

    if args.sam:
        print(f"[SAM2] Modo SAM2 activo — separacion con {args.sam_model}")
        from src.inference.sam2_seg import SAM2InstanceSegmenter

        # Convert BGR->RGB for SAM2
        image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        segmenter = SAM2InstanceSegmenter(
            model_type=args.sam_model,
            overlap_thresh=0.3,
            nms_thresh=0.5,
        )
        instancias_map = segmenter.separate_instances(image_rgb, mascara_final)
        instancias_stats = segmenter.compute_stats(instancias_map)
        num_wounds = len(instancias_stats)
        print(f"[SAM2] {num_wounds} herida(s) identificada(s)")

    if args.sam and args.instancias:
        print("[WARN] Se activaron --instancias y --sam. Se usa --sam (mas preciso).")

    print("Guardando resultados...")
    COLORES = {
        "verde":    (255, 200, 0),    # BGR: verde azulado
        "rojo":     (0, 0, 255),      # BGR: rojo
        "azul":     (255, 0, 0),      # BGR: azul
        "amarillo": (0, 255, 255),    # BGR: amarillo
        "naranja":  (0, 140, 255),    # BGR: naranja
    }
    color_bgr = COLORES.get(args.color, COLORES["verde"])

    cv2.imwrite(str(salida / f"{nombre}_original.png"), original)
    cv2.imwrite(str(salida / f"{nombre}_mascara_cruda.png"), mascara_cruda)
    cv2.imwrite(str(salida / f"{nombre}_mascara.png"), mascara_final)
    cv2.imwrite(str(salida / f"{nombre}_superpuesto.png"), superponer(original, mascara_final, color=color_bgr))

    # ── Instancia outputs ───────────────────────────────────────────
    if args.instancias and instancias_map is not None:
        # Build a colourised overlay with per-instance contours + labels
        overlay_inst = original.copy()  # BGR
        # Use COLORES as base palette, extend for many instances
        inst_palette = list(COLORES.values()) + [
            (255, 0, 255),    # magenta
            (255, 255, 0),    # amarillo brillante
            (0, 255, 255),    # cian
            (128, 0, 128),    # purpura
        ]

        for inst in (instancias_stats or []):
            iid = inst["instance_id"]
            # Contour mask for this instance
            inst_mask = (instancias_map == iid).astype(np.uint8) * 255
            contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = inst_palette[(iid - 1) % len(inst_palette)]
            cv2.drawContours(overlay_inst, contours, -1, color, 2)

            # Label at centroid
            cx, cy = int(inst["centroid_x"]), int(inst["centroid_y"])
            cv2.putText(
                overlay_inst, str(iid), (cx - 5, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )

        cv2.imwrite(str(salida / f"{nombre}_instancias.png"), overlay_inst)

        # CSV with per-instance stats
        csv_path = salida / f"{nombre}_instancias.csv"
        header = "instance_id,area_px,area_pct,bbox_x,bbox_y,bbox_w,bbox_h,centroid_x,centroid_y"
        rows = []
        for inst in (instancias_stats or []):
            rows.append(
                f"{inst['instance_id']},{inst['area_px']},{inst['area_pct']},"
                f"{inst['bbox_x']},{inst['bbox_y']},{inst['bbox_w']},{inst['bbox_h']},"
                f"{inst['centroid_x']},{inst['centroid_y']}"
            )
        csv_path.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")

    print(f"\nResultados guardados en: {salida}/")
    print(f"  {nombre}_original.png")
    print(f"  {nombre}_mascara_cruda.png   (sin filtros)")
    print(f"  {nombre}_mascara.png         (limpia)")
    print(f"  {nombre}_superpuesto.png     (overlay)")
    if args.instancias or args.sam:
        inst_type = "SAM2" if args.sam else "watershed"
        print(f"  {nombre}_instancias.png    (instancias coloreadas - {inst_type})")
        print(f"  {nombre}_instancias.csv    (estadisticas por instancia - {inst_type})")
        if instancias_stats:
            for inst in instancias_stats:
                print(f"    [{inst['instance_id']}] area={inst['area_px']}px  "
                      f"bbox=({inst['bbox_x']},{inst['bbox_y']},{inst['bbox_w']}x{inst['bbox_h']})  "
                      f"centroid=({inst['centroid_x']},{inst['centroid_y']})")

    if not args.cruda:
        reduccion = herida_cruda_pct - herida_final_pct
        print(f"\n  Los filtros eliminaron {reduccion:.2f}%% de ruido/falsos positivos.")


if __name__ == "__main__":
    main()
