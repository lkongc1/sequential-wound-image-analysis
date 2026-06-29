"""SAM2 post-processor for instance segmentation.

Takes a binary mask prediction from any model (FPN, U-Net, etc.) and uses
SAM2 (Segment Anything Model 2) to separate individual wound instances.

Usage:
    from src.inference.sam2_seg import SAM2InstanceSegmenter
    segmenter = SAM2InstanceSegmenter()
    instance_map = segmenter.separate_instances(image_rgb, binary_mask)
    stats = segmenter.compute_stats(instance_map)
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Cache for the SAM2 model (singleton pattern)
_SAM2_MODEL: dict[str, Any] = {}


def get_sam2_model(
    model_type: str = "sam2.1_hiera_l",
    device: str | None = None,
) -> torch.nn.Module:
    """Get or create the SAM2 model singleton via HuggingFace auto-download.

    Args:
        model_type: One of:
            - 'sam2.1_hiera_l'   (best, 224M params)
            - 'sam2.1_hiera_b+'  (balanced)
            - 'sam2.1_hiera_s'   (fast)
            - 'sam2_hiera_l'     (SAM2 v1 large)
            - 'sam2_hiera_b+'    (SAM2 v1 balanced)
        device: 'cuda' or 'cpu'. Auto-detects if None.

    Returns:
        SAM2 model in eval mode.
    """
    global _SAM2_MODEL

    cache_key = f"{model_type}_{device}"
    if cache_key in _SAM2_MODEL:
        return _SAM2_MODEL[cache_key]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Map model_type to HuggingFace model ID
    HF_MODEL_ID_MAP = {
        # SAM 2.1 (best)
        "sam2.1_hiera_l":  "facebook/sam2.1-hiera-large",
        "sam2.1_hiera_b+": "facebook/sam2.1-hiera-base-plus",
        "sam2.1_hiera_s":  "facebook/sam2.1-hiera-small",
        # SAM 2.0
        "sam2_hiera_l":    "facebook/sam2-hiera-large",
        "sam2_hiera_b+":   "facebook/sam2-hiera-base-plus",
        "sam2_hiera_s":    "facebook/sam2-hiera-small",
    }

    if model_type not in HF_MODEL_ID_MAP:
        raise ValueError(
            f"Unknown model type: {model_type}. Options: {list(HF_MODEL_ID_MAP.keys())}"
        )

    hf_model_id = HF_MODEL_ID_MAP[model_type]

    logger.info(f"Inicializando SAM2 ({model_type} = {hf_model_id}) en {device.upper()}...")

    from sam2.build_sam import build_sam2_hf

    model = build_sam2_hf(hf_model_id, device=device)
    model.eval()

    _SAM2_MODEL[cache_key] = model
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"SAM2 listo ({param_count:.0f}M params, device={device})")
    return model


class SAM2InstanceSegmenter:
    """Post-processes binary masks with SAM2 to separate wound instances.

    Uses SAM2's automatic mask generation to find all object masks in the
    image, then filters those that overlap with the binary mask prediction.
    Each remaining SAM2 mask becomes one wound instance.

    Args:
        model_type: SAM2 variant ('sam2.1_hiera_l', 'sam2.1_hiera_b+', etc.)
        pred_iou_thresh: Filter threshold for mask quality (lower = more masks)
        stability_score_thresh: Filter for stable masks (lower = more masks)
        min_mask_region_area: Minimum pixel area to keep a mask
        overlap_thresh: Minimum fraction of SAM2 mask pixels that must overlap
            the binary prediction (0.0-1.0). Lower = more permissive.
        points_per_side: Grid density for SAM2 mask generation
    """

    def __init__(
        self,
        model_type: str = "sam2.1_hiera_l",
        pred_iou_thresh: float = 0.6,
        stability_score_thresh: float = 0.7,
        min_mask_region_area: int = 50,
        overlap_thresh: float = 0.3,
        nms_thresh: float = 0.5,
        points_per_side: int = 32,
    ):
        self.model_type = model_type
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.overlap_thresh = overlap_thresh
        self.nms_thresh = nms_thresh
        self.points_per_side = points_per_side
        self._model = None
        self._mask_generator = None

    def _lazy_init(self):
        """Initialize model on first use."""
        if self._model is not None:
            return
        # Clear global cache to ensure fresh model with current params
        global _SAM2_MODEL
        _SAM2_MODEL.clear()
        self._model = get_sam2_model(self.model_type)
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        self._mask_generator = SAM2AutomaticMaskGenerator(
            model=self._model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
            box_nms_thresh=0.7,
            output_mode="binary_mask",
        )

    def separate_instances(
        self,
        image: np.ndarray,
        binary_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Separate wound instances in the image.

        Args:
            image: RGB image (H, W, 3) uint8.
            binary_mask: Optional binary uint8 mask (0/255) from model prediction.
                If None, uses all SAM2 masks directly.

        Returns:
            uint16 instance map: 0 = background, 1..N = instance IDs.
        """
        self._lazy_init()
        assert self._mask_generator is not None

        h, w = image.shape[:2]

        # SAM2 automatic mask generation
        logger.info("SAM2: generando máscaras automáticas...")
        sam2_masks = self._mask_generator.generate(image)
        logger.info(f"SAM2: {len(sam2_masks)} máscaras encontradas")

        if not sam2_masks:
            return np.zeros((h, w), dtype=np.uint16)

        candidates: list[np.ndarray] = []

        if binary_mask is not None:
            # Normalize binary mask
            if binary_mask.dtype != np.uint8 or binary_mask.max() <= 1:
                binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
            else:
                binary_mask_uint8 = binary_mask
            # Resize binary mask to match the original image resolution
            if binary_mask_uint8.shape[:2] != (h, w):
                binary_mask_uint8 = cv2.resize(
                    binary_mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST
                )
            binary_mask_bool = binary_mask_uint8 > 0
            total_pred_px = int(binary_mask_bool.sum())
        else:
            binary_mask_bool = None
            total_pred_px = 0

        for sam_mask in sam2_masks:
            seg = sam_mask["segmentation"]  # bool array (H, W)
            if binary_mask_bool is not None and seg.sum() > 0:
                # What fraction of this SAM2 mask overlaps the binary prediction?
                # This avoids penalizing small wounds for the "global" IoU.
                overlap = int((seg & binary_mask_bool).sum()) / max(int(seg.sum()), 1)
            else:
                overlap = 1.0  # No filter if no binary mask

            if overlap >= self.overlap_thresh:
                candidates.append(seg)

        # Sort candidates by area (largest first)
        candidates.sort(key=lambda m: int(m.sum()), reverse=True)

        # NMS: skip masks that overlap significantly with already-kept masks
        kept_masks: list[np.ndarray] = []
        for mask in candidates:
            mask_bool = mask.astype(bool)
            if self.nms_thresh >= 1.0:
                kept_masks.append(mask_bool)
                continue
            # Check IoU against all kept masks
            overlaps_other = False
            for kept in kept_masks:
                intersection = int((mask_bool & kept).sum())
                smaller_area = min(int(mask_bool.sum()), int(kept.sum()))
                if smaller_area > 0 and intersection / smaller_area > self.nms_thresh:
                    overlaps_other = True
                    break
            if not overlaps_other:
                kept_masks.append(mask_bool)

        # Build instance map from kept masks
        instance_map = np.zeros((h, w), dtype=np.uint16)
        instance_id = 1
        for mask in kept_masks:
            # Ensure mask is bool
            mask_bool = mask.astype(bool)
            new_region = mask_bool & (instance_map == 0)
            area = int(new_region.sum())
            if area >= self.min_mask_region_area:
                instance_map[new_region] = instance_id
                instance_id += 1

        logger.info(f"SAM2: {instance_id - 1} instancias de herida finales (de {len(candidates)} candidatos)")
        return instance_map

    def compute_stats(
        self, instance_map: np.ndarray, image_shape: tuple[int, int] | None = None
    ) -> list[dict[str, Any]]:
        """Compute per-instance statistics matching the format of compute_instance_stats."""
        from src.inference.postprocessing import compute_instance_stats

        return compute_instance_stats(instance_map, image_shape)

    def __del__(self):
        """Cleanup."""
        global _SAM2_MODEL
        _SAM2_MODEL.clear()
