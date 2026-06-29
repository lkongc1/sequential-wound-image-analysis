#!/usr/bin/env python3
"""Semantic segmentation + Watershed instance segmentation demo.

Loads a trained segmentation model, runs inference on a wound image,
converts the probability map into binary masks at multiple thresholds,
applies Watershed to separate touching regions into individual instances,
computes per-instance statistics, and exports an overlay image + CSV.

Usage:
    python scripts/inference/demo_watershed.py <image_path>
    python scripts/inference/demo_watershed.py wound.jpg --threshold 0.4 --output results/
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import create_model
from src.datasets.wound_dataset import get_default_transforms
from src.inference.postprocessing import watershed_instances, compute_instance_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic segmentation + Watershed instance segmentation demo"
    )
    parser.add_argument(
        "image", type=Path,
        help="Path to the wound image to process"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "models" / "screening" / "FPN_EfficientNetB3_best.pth",
        help="Path to model checkpoint (default: models/screening/FPN_EfficientNetB3_best.pth)"
    )
    parser.add_argument(
        "--model", default="fpn",
        help="Model architecture name (default: fpn)"
    )
    parser.add_argument(
        "--encoder", default="efficientnet-b3",
        help="Encoder backbone (default: efficientnet-b3)"
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.3, 0.4, 0.5],
        help="Threshold values to try (default: 0.3 0.4 0.5)"
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "output" / "watershed_demo",
        help="Output directory (default: output/watershed_demo)"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device to run inference on (default: cuda)"
    )
    parser.add_argument(
        "--dist-threshold", type=float, default=0.3,
        help="Distance threshold ratio for Watershed (default: 0.3)"
    )
    parser.add_argument(
        "--min-area", type=int, default=50,
        help="Minimum instance area in pixels (default: 50)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(
        str(args.checkpoint),
        map_location=args.device,
        weights_only=False,
    )
    model = create_model(args.model, encoder_name=args.encoder, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(args.device).eval()

    # Load image
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Error: could not read image: {args.image}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_stem = args.image.stem

    # Predict
    transform = get_default_transforms((384, 384))
    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(args.device)
    with torch.inference_mode():
        pred = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

    # Try multiple thresholds
    for threshold in args.thresholds:
        mask = (pred > threshold).astype(np.uint8) * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        instance_map = watershed_instances(
            mask,
            dist_threshold_ratio=args.dist_threshold,
            min_instance_area=args.min_area,
            kernel_close=3,
        )
        stats = compute_instance_stats(instance_map)

        pos_px = int((mask > 0).sum())
        print(f"Threshold={threshold}: {pos_px}px positive, {len(stats)} instances")
        for s in stats:
            sid = s["instance_id"]
            apx = s["area_px"]
            bx, by, bw, bh = s["bbox_x"], s["bbox_y"], s["bbox_w"], s["bbox_h"]
            print(f"  [{sid}] area={apx}px bbox=({bx},{by},{bw}x{bh})")

        # Export overlay and CSV for the primary threshold (first one by default, 0.4 historically)
        export_threshold = 0.4 if 0.4 in args.thresholds else args.thresholds[0]
        if threshold == export_threshold:
            overlay = img.copy()
            colors = [(255, 200, 0), (0, 200, 255), (200, 0, 255)]
            for s in stats:
                sid = s["instance_id"]
                color = colors[(sid - 1) % len(colors)]
                mask_i = (instance_map == sid).astype(np.uint8) * 255
                overlay[mask_i > 0] = color
                contours, _ = cv2.findContours(
                    mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
                if contours:
                    M = cv2.moments(contours[0])
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(
                            overlay, str(sid), (cx - 5, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                        )

            overlay_img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
            overlay_path = args.output / f"{image_stem}_watershed_instances.png"
            cv2.imwrite(str(overlay_path), overlay_img)
            print(f"\nOverlay saved: {overlay_path} ({overlay_img.shape})")

            csv_path = args.output / f"{image_stem}_watershed_instances.csv"
            fields = [
                "instance_id", "area_px", "area_pct",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "centroid_x", "centroid_y",
            ]
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for s in stats:
                    w.writerow(s)
            print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
