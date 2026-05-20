"""Kaggle data source implementation.

SRP: Handles Kaggle data downloading only.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.sources.base import BaseDataSource

logger = logging.getLogger(__name__)


class KaggleSource(BaseDataSource):
    """Kaggle dataset source implementation.

    Example:
        >>> source = KaggleSource(dataset_slug="user/dataset")
        >>> source.download(Path("data/raw"))
    """

    def __init__(
        self,
        dataset_slug: str,
        credentials_path: Optional[Path] = None,
        timeout: int = 7200,
    ):
        self.dataset_slug = dataset_slug
        self._dataset_name = dataset_slug.split("/")[-1]
        self._credentials_path = credentials_path
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"KaggleSource({self.dataset_slug})"

    def download(
        self,
        output_dir: Path,
        unzip: bool = True,
        force: bool = False,
        expected_files: Optional[List[str]] = None,
    ) -> bool:
        """Download dataset from Kaggle."""
        output_dir = Path(output_dir).resolve()
        start_time = time.time()

        if not force and self._is_already_downloaded(output_dir):
            logger.info(f"Dataset already present: {output_dir}")
            return True

        logger.info(f"Starting download: {self.dataset_slug}")
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["kaggle", "datasets", "download", "-d", self.dataset_slug, "-p", str(output_dir)]
        if unzip:
            cmd.append("--unzip")

        try:
            process = subprocess.run(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=False,
                text=True,
                timeout=self._timeout
            )
            elapsed = time.time() - start_time

            if process.returncode != 0:
                logger.error(f"Download failed with exit code {process.returncode}")
                return False

            logger.info(f"Download completed in {elapsed:.1f}s")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Download timed out after {self._timeout}s")
            return False
        except FileNotFoundError:
            logger.error("'kaggle' command not found. Install with: pip install kaggle")
            return False

    def _is_already_downloaded(self, output_dir: Path) -> bool:
        if not output_dir.exists():
            return False
        has_content = any(
            p.name.lower() in ("images", "masks", "train", "val", "test")
            or p.suffix in (".png", ".jpg", ".jpeg")
            for p in output_dir.rglob("*")
        )
        return has_content

    def load(self, data_dir: Path) -> Tuple[List[Path], List[Path]]:
        """Load image and mask paths from downloaded dataset."""
        data_dir = Path(data_dir)

        image_patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
        mask_patterns = ["**/*mask*.png", "**/*mask*.jpg", "**/masks/**"]

        image_paths = []
        mask_paths = []

        for pattern in image_patterns:
            image_paths.extend(data_dir.glob(pattern))
        for pattern in mask_patterns:
            mask_paths.extend(data_dir.glob(pattern))

        image_paths = sorted(set(image_paths))
        mask_paths = sorted(set(mask_paths))

        logger.info(f"Loaded {len(image_paths)} images and {len(mask_paths)} masks")
        return image_paths, mask_paths

    def validate(self, data_dir: Path) -> Dict[str, Any]:
        """Validate downloaded dataset."""
        data_dir = Path(data_dir)

        result = {"valid": True, "errors": [], "warnings": [], "stats": {}}

        if not data_dir.exists():
            result["valid"] = False
            result["errors"].append(f"Directory does not exist: {data_dir}")
            return result

        image_count = len(list(data_dir.glob("**/*.png")))
        image_count += len(list(data_dir.glob("**/*.jpg")))
        mask_count = len(list(data_dir.glob("**/*mask*")))

        result["stats"] = {
            "image_count": image_count,
            "mask_count": mask_count,
        }

        if image_count == 0:
            result["warnings"].append("No images found")
        if mask_count == 0:
            result["warnings"].append("No masks found")

        return result
