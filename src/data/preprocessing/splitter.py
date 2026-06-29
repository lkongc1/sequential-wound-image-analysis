"""Data splitting strategies for train/val/test division.

SOLID principles:
- SRP: DataSplitter handles only data splitting logic
- DIP: Group extraction abstracted via _extract_group_id (can be overridden)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


class DataSplitter:
    """Splits a DataFrame containing dataset metadata into train/val/test sets.

    Uses GroupShuffleSplit to ensure patient-level isolation across splits.
    If no patient_id column exists, falls back to source-based grouping as
    minimum safeguard against cross-domain contamination.

    SOLID:
    - SRP: Only handles data splitting logic
    - DIP: Group extraction abstracted via _extract_group_id (can be overridden)
    """

    KNOWN_SOURCES: frozenset[str] = frozenset({"fusc", "medetec", "wsnet"})

    def __init__(
        self,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        """Initialize the splitter.

        Args:
            val_size: Fraction of training data for validation (default 0.2 = 20%).
            test_size: Fraction for test set (default 0.1 = 10%).
            random_state: Random seed for reproducibility.
        """
        if not 0.0 < val_size < 1.0:
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        if val_size + test_size >= 1.0:
            raise ValueError(
                f"val_size ({val_size}) + test_size ({test_size}) must be < 1.0"
            )
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

    def _extract_group_id(self, filename: str) -> str:
        """Extract group identifier from filename using pattern matching.

        Heuristic priority:
        1. If filename matches prefix_number pattern and prefix is a known source
           (fusc, medetec, wsnet) -> use prefix as group (domain-level isolation).
        2. Otherwise extract prefix_number as a unique patient ID.

        Args:
            filename: Image filename (e.g., 'fusc_0045.png')

        Returns:
            Group identifier string
        """
        name = Path(filename).stem if isinstance(filename, (str, Path)) else str(filename)
        match = re.match(r'^([a-zA-Z]+)_(\d+)$', name)
        if match:
            prefix, _ = match.groups()
            if prefix.lower() in self.KNOWN_SOURCES:
                return prefix.lower()
            return f"{prefix}_{match.group(2)}"
        return name

    def _get_groups(self, df: pd.DataFrame) -> pd.Series:
        """Resolve group identifiers for each row.

        Priority:
        1. Explicit 'patient_id' column
        2. Inferred from filename via _extract_group_id
        3. Fallback to 'source' column (domain split)
        """
        if "patient_id" in df.columns:
            return df["patient_id"]
        if "filename" in df.columns:
            return df["filename"].apply(self._extract_group_id)
        if "source" in df.columns:
            return df["source"]
        raise ValueError(
            "Cannot determine groups: no 'patient_id', 'filename', or 'source' column found."
        )

    def split_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split training samples into train and validation sets (group-aware).

        Takes samples marked as 'train', splits them into train/val using
        GroupShuffleSplit to keep patients isolated.

        Args:
            df: DataFrame with 'split' column containing at least 'train' samples.

        Returns:
            DataFrame with 'split' column updated to include 'train', 'val', 'test'.

        Raises:
            ValueError: If no training samples found or no groups can be determined.
        """
        if "split" not in df.columns:
            raise ValueError("DataFrame must have a 'split' column")

        train_mask = df["split"] == "train"
        if not train_mask.any():
            raise ValueError("No samples found with split='train'")

        train_df = df[train_mask].copy()
        other_df = df[~train_mask].copy()
        if len(train_df) == 0:
            raise ValueError("No training samples to split")

        groups = self._get_groups(train_df).values

        gss = GroupShuffleSplit(n_splits=1, test_size=self.val_size, random_state=self.random_state)
        train_idx, val_idx = next(gss.split(train_df, groups=groups))

        train_subset = train_df.iloc[train_idx].copy().reset_index(drop=True)
        val_subset = train_df.iloc[val_idx].copy().reset_index(drop=True)
        train_subset["split"] = "train"
        val_subset["split"] = "val"

        result = pd.concat([train_subset, val_subset, other_df], ignore_index=True)
        return result

    def split_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split ALL samples into train/val/test using GroupShuffleSplit.

        This method is used when all samples are marked as 'train'
        and need complete group-aware redistribution.

        Args:
            df: DataFrame to split. Must contain filename or source column for grouping.

        Returns:
            DataFrame with redistributed 'split' column ('train', 'val', 'test').

        Raises:
            ValueError: If no groups can be determined.
        """
        groups = self._get_groups(df).values

        # Step 1: separate test set
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_val_groups = groups[train_val_idx]

        # Step 2: separate val from train_val
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=self.val_size, random_state=self.random_state
        )
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_groups))

        train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        return pd.concat([train_df, val_df, test_df], ignore_index=True)
