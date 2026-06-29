"""Postprocessing of predicted masks."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def clean_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Remove small blobs from predicted mask."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask_uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)
    return (cleaned > 127).astype(np.float32)


def apply_morphology(mask: np.ndarray, operation: str = "close", kernel_size: int = 5) -> np.ndarray:
    """Apply morphological operations to clean mask edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_uint8 = (mask * 255).astype(np.uint8)
    if operation == "close":
        result = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    elif operation == "open":
        result = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    else:
        result = mask_uint8
    return (result > 127).astype(np.float32)


def watershed_instances(
    binary_mask: np.ndarray,
    dist_threshold_ratio: float = 0.3,
    min_instance_area: int = 50,
    kernel_close: int = 3,
) -> np.ndarray:
    """Separate touching blobs in a binary mask via watershed.

    Pipeline:
        1. Morphological close to merge nearby fragments.
        2. Distance transform → threshold for sure-foreground markers.
        3. Sure-background via dilation.
        4. Unknown region = sure_bg - sure_fg.
        5. Connected components on sure-fg → marker image.
        6. cv2.watershed() on a colourised version of the markers.

    Args:
        binary_mask: Binary uint8 mask (0/255) or bool / float mask.
        dist_threshold_ratio: Fraction of max distance for sure-fg markers
            (lower = more instances, default 0.3).
        min_instance_area: Minimum pixel area to retain an instance
            (default 50).
        kernel_close: Kernel size for morphological close before distance
            transform (default 3).

    Returns:
        uint16 instance map with same spatial shape as input:
        0 = background, 1..N = instance IDs.
    """
    # Normalise to uint8 0/255
    if binary_mask.dtype != np.uint8 or binary_mask.max() > 1:
        mask_uint8 = binary_mask.astype(np.uint8)
    else:
        mask_uint8 = (binary_mask * 255).astype(np.uint8)

    # Edge case: empty mask
    if cv2.countNonZero(mask_uint8) == 0:
        return np.zeros(mask_uint8.shape, dtype=np.uint16)

    # Edge case: already multiple disconnected blobs (no watershed needed)
    # Return each CC as a separate instance for speed.
    num_labels, cc_labels = cv2.connectedComponents(mask_uint8, connectivity=8)
    if num_labels > 2:  # label 0 = background, 1..N = blobs
        instance_map = np.zeros(mask_uint8.shape, dtype=np.uint16)
        for label_id in range(1, num_labels):
            mask = cc_labels == label_id
            if int(mask.sum()) >= min_instance_area:
                instance_map[mask] = label_id
        return instance_map

    # 1. Morphological close to smooth boundaries
    if kernel_close >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
        mask_proc = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    else:
        mask_proc = mask_uint8.copy()

    # 2. Distance transform
    dist = cv2.distanceTransform(mask_proc, cv2.DIST_L2, 5)
    dist = np.clip(dist, 0, None).astype(np.float32)

    max_dist = float(dist.max())
    if max_dist < 1e-6:
        # Degenerate case — mask exists but distance is near zero (single line)
        instance_map = np.zeros(mask_uint8.shape, dtype=np.uint16)
        instance_map[mask_uint8 > 0] = 1
        return instance_map

    # 3. Sure-foreground markers via distance threshold
    thr_dist = dist_threshold_ratio * max_dist
    _, sure_fg = cv2.threshold(dist, thr_dist, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # 4. Sure-background via dilation
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv2.dilate(mask_proc.astype(np.uint8), kernel_dil, iterations=3)

    # 5. Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Build marker image for watershed
    #    cv2.connectedComponents: 0 = background, 1..N = components
    #    cv2.watershed expects markers > 0, where 1 = background.
    #    So shift: old bg(0) → 1, old comp(1) → 2, old comp(2) → 3, ...
    num_markers, markers = cv2.connectedComponents(sure_fg, connectivity=8)
    markers = markers + 1
    # Mark unknown region as 0 (watershed treats 0 as unlabelled)
    markers[unknown > 0] = 0

    # 7. Build 3-ch image for watershed input (mask as grayscale tripled)
    marker_colour = np.stack([mask_uint8] * 3, axis=-1).astype(np.uint8)

    # 8. Apply watershed — markers are mutated in-place
    markers = cv2.watershed(marker_colour, markers)

    # 9. Convert watershed output to uint16 instance map
    #    watershed returns: -1 = boundaries, 1 = original background,
    #    2..num_markers+1 = instances (some may be missing if merged)
    instance_map = np.zeros_like(markers, dtype=np.uint16)
    # Labels from CC: label 1 is the original CC background, labels 2..N are
    # the markers that watershed assigned as instances
    for label_id in range(2, markers.max() + 1):
        instance_mask = markers == label_id
        area = int(instance_mask.sum())
        if area >= min_instance_area:
            instance_map[instance_mask] = label_id - 1  # shift to 1-based

    return instance_map


def compute_instance_stats(
    instance_map: np.ndarray,
    image_shape: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Compute per-instance statistics from an instance map.

    Args:
        instance_map: uint16 map where 0 = background, 1..N = instance IDs.
        image_shape: (height, width) of the original image. If None, uses
            instance_map.shape. Used for area_pct calculation.

    Returns:
        List of dicts, one per instance, with keys:
            instance_id, area_px, area_pct, bbox_{x,y,w,h}, centroid_{x,y}.
        Returns empty list if no instances found.
    """
    h, w = image_shape or instance_map.shape[:2]
    total_area = h * w

    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]  # skip background

    if len(unique_ids) == 0:
        return []

    stats: list[dict[str, Any]] = []
    for inst_id in unique_ids:
        mask = instance_map == inst_id
        area_px = int(mask.sum())

        # Bounding box from non-zero coordinates
        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0).tolist()
        y_max, x_max = coords.max(axis=0).tolist()
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1

        # Centroid
        centroid_y = float(coords[:, 0].mean())
        centroid_x = float(coords[:, 1].mean())

        stats.append({
            "instance_id": int(inst_id),
            "area_px": area_px,
            "area_pct": round(area_px / total_area * 100, 4),
            "bbox_x": x_min,
            "bbox_y": y_min,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "centroid_x": round(centroid_x, 2),
            "centroid_y": round(centroid_y, 2),
        })

    return stats
