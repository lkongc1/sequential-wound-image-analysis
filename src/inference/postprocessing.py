"""Postprocessing of predicted masks."""
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
