#!/usr/bin/env python3
"""CLI for downloading wound segmentation dataset from Kaggle."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sources import KaggleSource

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download wound segmentation dataset from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/1_download_dataset.py
  python scripts/1_download_dataset.py --data-dir ./data/raw --verbose
  python scripts/1_download_dataset.py --force
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "data_wound_seg",
        help="Output directory for raw data (default: %(default)s)",
    )
    parser.add_argument(
        "--creds",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "api-json" / "kaggle.json",
        help="Path to Kaggle credentials JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default="leoscode/wound-segmentation-images",
        help="Kaggle dataset slug (default: %(default)s)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if dataset exists",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    logger.info(f"Starting download: {args.slug}")
    logger.info(f"Output directory: {args.data_dir}")

    source = KaggleSource(
        dataset_slug=args.slug,
        credentials_path=args.creds,
    )

    start_time = time.time()
    success = source.download(
        output_dir=args.data_dir,
        unzip=True,
        force=args.force,
    )
    elapsed = time.time() - start_time

    if success:
        logger.info(f"Download completed in {elapsed:.1f}s")
        result = source.validate(args.data_dir)
        if result["warnings"]:
            for w in result["warnings"]:
                logger.warning(w)
        logger.info(f"Dataset valid: {result['valid']}")
        logger.info(f"Images: {result['stats'].get('image_count', 0)}, Masks: {result['stats'].get('mask_count', 0)}")
        return 0
    else:
        logger.error("Download failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
