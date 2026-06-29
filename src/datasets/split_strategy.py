"""Data split strategies: train/val/test splits."""
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from pathlib import Path


def train_val_test_split(
    image_paths: List[Path],
    mask_paths: List[Path],
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List, List, List, List]:
    """Split image/mask pairs into train/val/test sets."""
    test_size_adj = test_size / (1 - val_size)
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=val_size + test_size_adj,
        random_state=seed,
        shuffle=True,
    )
    val_size_adj = val_size / (val_size + test_size_adj)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs,
        temp_masks,
        test_size=val_size_adj,
        random_state=seed,
        shuffle=True,
    )
    return train_imgs, val_imgs, test_imgs, train_masks, val_masks, test_masks
