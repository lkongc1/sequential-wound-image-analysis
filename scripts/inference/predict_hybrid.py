#!/usr/bin/env python3
"""Pipeline YOLO + U-Net: detecta region de herida, recorta, segmenta.

Proposito: Hard Attention — YOLO localiza, U-Net segmenta solo la region relevante.

Con --tipo CHECKPOINT: clasifica cada bbox detectado por YOLO usando
el clasificador de tipo de herida (4ch: RGB + mascara U-Net).

Uso:
    python scripts/inference/predict_hybrid.py imagen.png
    python scripts/inference/predict_hybrid.py imagen.png --tipo models/classifier/best.pth
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2, numpy as np, torch
from ultralytics import YOLO
from src.datasets.wound_dataset import get_default_transforms
from src.models.factory import create_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT = Path(__file__).resolve().parent.parent.parent

# ── Modelos (carga perezosa para no ralentizar --help) ──────────────
_yolo = None
_unet = None
_classifier_cache: dict[str, tuple] = {}


def _get_yolo():
    global _yolo
    if _yolo is None:
        _yolo = YOLO(str(PROJECT / "runs/detect/wound_detector/weights/best.pt"))
    return _yolo


def _get_unet():
    global _unet
    if _unet is None:
        _unet = create_model("unet", encoder_name="resnet50", pretrained=False)
        ckpt = torch.load(
            str(PROJECT / "models/sensibilidad/unet_resnet50_v3.pth"),
            map_location=DEVICE, weights_only=False,
        )
        _unet.load_state_dict(ckpt["model_state_dict"])
        _unet.to(DEVICE).eval()
    return _unet


def _load_classifier(checkpoint_path: str) -> tuple:
    """Load wound classifier from checkpoint. Cached."""
    if checkpoint_path in _classifier_cache:
        return _classifier_cache[checkpoint_path]

    from src.config import ClassificationConfig

    config = ClassificationConfig()

    # Auto-detect num_classes from checkpoint head shape
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
        head_key = "model.head.1.weight" if "model.head.1.weight" in state else "head.1.weight"
        if head_key in state:
            detected = state[head_key].shape[0]
            if detected != config.num_classes:
                print(f"  [INFO] Checkpoint tiene {detected} clases → usando {detected}")
                config.num_classes = detected
    elif "head.1.weight" in ckpt:
        detected = ckpt["head.1.weight"].shape[0]
        if detected != config.num_classes:
            print(f"  [INFO] Checkpoint tiene {detected} clases → usando {detected}")
            config.num_classes = detected

    model = create_model(
        "wound_classifier",
        num_classes=config.num_classes,
        pretrained=False,
        dropout=config.dropout,
    )
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
        state = {k.replace("model.", ""): v for k, v in state.items() if k.startswith("model.")}
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.to(DEVICE).eval()
    _classifier_cache[checkpoint_path] = (model, config)
    print(f"  Clasificador cargado: {Path(checkpoint_path).name} -> {DEVICE.upper()} ({config.num_classes} clases)")
    return model, config


def _classify_bbox(
    image_bgr: np.ndarray,
    bbox: tuple,
    mask_crop: np.ndarray,
    classifier: torch.nn.Module,
    config: "ClassificationConfig",
) -> tuple[str, float]:
    """Classify a single YOLO bbox using its U-Net mask.

    Args:
        image_bgr: Full BGR image.
        bbox: (x1, y1, x2, y2) in image coordinates.
        mask_crop: Binary mask (0/255) for this bbox, same size as crop.
        classifier: WoundClassifier in eval mode.
        config: ClassificationConfig.

    Returns:
        (class_name, confidence).
    """
    from albumentations.pytorch import ToTensorV2
    import albumentations as A

    x1, y1, x2, y2 = bbox
    crop_rgb = image_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2RGB)
    crop_mask_f = (mask_crop > 127).astype(np.float32)

    H, W = int(config.image_size[0]), int(config.image_size[1])

    img_transform = A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img_tensor = img_transform(image=crop_rgb)["image"]

    mask_resized = cv2.resize(crop_mask_f, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

    x = torch.cat([img_tensor, mask_tensor], dim=0).unsqueeze(0)  # (1, 4, H, W)

    try:
        x = x.to(DEVICE)
        with torch.inference_mode():
            log_probs = classifier(x)
        probs = torch.exp(log_probs).squeeze(0).cpu()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("  [WARN] CUDA OOM — fallback a CPU", file=sys.stderr)
            classifier_cpu = classifier.cpu()
            x_cpu = x.cpu()
            with torch.inference_mode():
                log_probs = classifier_cpu(x_cpu)
            probs = torch.exp(log_probs).squeeze(0)
            classifier.to(DEVICE)
        else:
            raise

    max_conf, pred_idx = probs.max(dim=0)
    max_conf_f = float(max_conf.item())
    pred_idx_i = int(pred_idx.item())

    if max_conf_f < config.confidence_threshold:
        class_name = "desconocido"
    else:
        class_name = config.class_names[pred_idx_i]

    return class_name, max_conf_f


def pipeline(image_path: str, output_dir: str = None, tipo_checkpoint: str = None):
    """YOLO detect -> Crop -> U-Net segment -> Stitch back.
    
    If --tipo is provided, also classify each bbox and write _tipo.txt.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: no se pudo leer {image_path}")
        return
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    yolo = _get_yolo()
    unet = _get_unet()

    # 1. YOLO: detectar bbox
    results = yolo(img, conf=0.15, verbose=False)
    boxes = results[0].boxes

    zero_detections = boxes is None or len(boxes) == 0

    if zero_detections:
        print("YOLO: SIN DETECCION — pasando imagen completa a U-Net")
        bboxes = [(0, 0, w, h)]
    else:
        bboxes = []
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4].astype(int)
            bw, bh = x2 - x1, y2 - y1
            pad_w, pad_h = int(bw * 0.2), int(bh * 0.2)
            x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
            bboxes.append((x1, y1, x2, y2))
        print(f"YOLO: {len(bboxes)} herida(s) detectada(s)")

    # 2. Por cada bbox: crop -> U-Net -> stitch
    full_mask = np.zeros((h, w), dtype=np.uint8)
    transform = get_default_transforms((384, 384))
    per_bbox_masks: list[np.ndarray] = []

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            per_bbox_masks.append(np.zeros((bh, bw), dtype=np.uint8))
            continue

        crop = img_rgb[y1:y2, x1:x2]
        t = transform(image=crop)
        tensor = t["image"].unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            pred = torch.sigmoid(unet(tensor)).squeeze().cpu().numpy()

        mask_crop = (pred > 0.5).astype(np.uint8) * 255
        mask_crop = cv2.resize(mask_crop, (bw, bh))
        per_bbox_masks.append(mask_crop)

        # Stitch back
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask_crop)

    # 3. Classification (--tipo)
    out = Path(output_dir) if output_dir else Path(image_path).parent
    name = Path(image_path).stem
    tipo_path = out / f"{name}_tipo.txt"

    if tipo_checkpoint:
        if zero_detections:
            tipo_path.write_text("SIN_DETECCION", encoding="utf-8")
            print(f"[TIPO] YOLO sin detecciones → SIN_DETECCION en {tipo_path}")
        else:
            classifier, clf_config = _load_classifier(tipo_checkpoint)
            lines: list[str] = []
            print(f"\n[TIPO] Clasificando {len(bboxes)} bbox(es):")
            for i, (bbox, mask) in enumerate(zip(bboxes, per_bbox_masks)):
                if mask.max() == 0:
                    cls_name, conf = "sin_deteccion", 0.0
                else:
                    cls_name, conf = _classify_bbox(img, bbox, mask, classifier, clf_config)
                inst_id = i + 1
                lines.append(f"{inst_id}: {cls_name} ({conf:.2f})")
                print(f"  [{inst_id}] {cls_name} (conf={conf:.2f})")
            tipo_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  → {tipo_path}")

    # 4. Guardar resultados
    total = h * w
    px = (full_mask > 0).sum()
    print(f"U-Net: {px}px ({px/total*100:.2f}%) de herida segmentada")

    cv2.imwrite(str(out / f"{name}_original.png"), img)
    cv2.imwrite(str(out / f"{name}_mascara_yolo_unet.png"), full_mask)

    overlay = img.copy()
    overlay[full_mask > 0] = [255, 200, 0]
    overlay = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # Dibujar bboxes de YOLO en el overlay
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imwrite(str(out / f"{name}_superpuesto_yolo_unet.png"), overlay)

    print(f"Resultados: {name}_superpuesto_yolo_unet.png")
    print(f"  BBox YOLO: amarillo | Mascara U-Net: verde")

    return full_mask, bboxes


# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline YOLO+U-Net con clasificacion opcional de tipo")
    parser.add_argument("imagen", type=Path, help="Ruta a la imagen a analizar")
    parser.add_argument("--tipo", type=Path, default=None, help="Checkpoint del clasificador de tipo de herida (best.pth)")
    parser.add_argument("--salida", type=Path, default=None, help="Directorio de salida (default: junto a la imagen)")
    args = parser.parse_args()

    if not args.imagen.exists():
        print(f"[ERROR] No se encontro: {args.imagen}", file=sys.stderr)
        sys.exit(1)

    if args.tipo is not None and not args.tipo.exists():
        print(f"[ERROR] Checkpoint de clasificador no encontrado: {args.tipo}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  {args.imagen}")
    print(f"{'='*60}")
    pipeline(
        str(args.imagen),
        output_dir=str(args.salida) if args.salida else None,
        tipo_checkpoint=str(args.tipo) if args.tipo else None,
    )
