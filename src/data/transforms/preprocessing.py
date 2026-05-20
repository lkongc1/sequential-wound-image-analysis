"""Image preprocessing: normalization, illumination correction, resizing."""
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def normalize_image(
    image: np.ndarray,
    alpha: float = 0.0,
    beta: float = 255.0,
    norm_type: int = cv2.NORM_MINMAX,
    dtype: int = cv2.CV_32F,
) -> np.ndarray:
    """Normalizes image to range [alpha, beta] with explicit destination array.

    Avoids Pylance warning about Optional[MatLike] from dst=None.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty or None image received")
    # Ensure dtype is compatible with cv2.normalize
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    # Create explicit destination to avoid None dst warning
    normalized: np.ndarray = np.zeros_like(image, dtype=np.float32)
    cv2.normalize(src=image, dst=normalized, alpha=alpha, beta=beta, norm_type=norm_type, dtype=dtype)
    return normalized


def correct_illumination(image: np.ndarray, method: str = "gray") -> np.ndarray:
    """Correct non-uniform illumination using background estimation."""
    if method == "gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        cv2.divide(gray, blur, scale=255, dst=gray)
        normalized: np.ndarray = np.zeros_like(image, dtype=np.float32)
        cv2.normalize(src=image, dst=normalized, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
        return normalized
    elif method == "retinex":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        cv2.addWeighted(gray, 1.5, blur, -0.5, 0, dst=gray)
        normalized: np.ndarray = np.zeros_like(image, dtype=np.float32)
        cv2.normalize(src=image, dst=normalized, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
        return normalized
    return image


def resize_with_padding(image: np.ndarray, target_size: Tuple[int, int] = (384, 384)) -> np.ndarray:
    """Resize image keeping aspect ratio with zero padding."""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    result = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return result


def calculate_image_stats(image_path: Path) -> Dict[str, Any]:
    """Calculate basic image statistics."""
    from PIL import Image

    stats: Dict[str, Any] = {
        "width": 0,
        "height": 0,
        "mode": "UNKNOWN",
        "channels": 0,
        "brightness_mean": 0.0,
        "brightness_std": 0.0,
        "contrast": 0.0,
        "is_grayscale": False,
        "file_size_kb": 0.0,
    }
    try:
        if not image_path.exists():
            return stats
        stats["file_size_kb"] = round(image_path.stat().st_size / 1024, 2)
        img = Image.open(image_path)
        stats["width"], stats["height"] = img.size
        stats["mode"] = img.mode
        img_np = np.array(img)
        stats["channels"] = len(img_np.shape) if len(img_np.shape) > 2 else 1
        stats["is_grayscale"] = img.mode == "L" or (len(img_np.shape) == 2)
        if stats["is_grayscale"]:
            brightness = img_np.astype(np.float32)
        else:
            brightness = (
                0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
            ).astype(np.float32)
        stats["brightness_mean"] = round(float(np.mean(brightness)), 2)
        stats["brightness_std"] = round(float(np.std(brightness)), 2)
        stats["contrast"] = stats["brightness_std"]
    except Exception as e:
        logger.warning("Error processing %s: %s", image_path.name, e)
    return stats
