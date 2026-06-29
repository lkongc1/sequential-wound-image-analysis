"""Path resolution for images and masks following SOLID principles.

DIP: High-level modules depend on abstractions (PathResolverProtocol).
SRP: ImageResolver only resolves image paths, MaskResolver only resolves masks.
OCP: Easy to extend with new resolution strategies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class PathResolverProtocol(Protocol):
    """Protocol for path resolvers - defines the interface."""

    def resolve(self, filename: str) -> Optional[Path]:
        """Resolve a filename to its full path."""
        ...

    def resolve_many(self, filenames: List[str]) -> Dict[str, Optional[Path]]:
        """Resolve multiple filenames at once."""
        ...


@dataclass
class ImageResolver:
    """Resolves image file paths using recursive search.

    This resolver handles the critical bug where filenames from CSV
    don't match flat directory assumptions. It searches recursively
    and caches results for performance.

    Attributes:
        root_dir: Root directory to search in.
        extensions: File extensions to search for.
        case_sensitive: Whether to match case exactly.

    Example:
        >>> resolver = ImageResolver(Path("data/raw/data_wound_seg"))
        >>> path = resolver.resolve("fusc_0001.png")
        >>> print(path)  # Could be: data/raw/data_wound_seg/test_images/fusc_0001.png
    """

    root_dir: Path
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    case_sensitive: bool = False
    _cache: Dict[str, Path] = field(default_factory=dict, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate root directory exists."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {self.root_dir}")

    def _build_cache(self) -> None:
        """Build the filename-to-path cache by walking the directory tree."""
        if self._initialized:
            return

        logger.info(f"Building image path cache from {self.root_dir}...")
        self._cache.clear()

        for ext in self.extensions:
            for path in self.root_dir.rglob(f"*{ext}"):
                filename = path.name
                if not self.case_sensitive:
                    filename = filename.lower()
                self._cache[filename] = path
                # Also index by stem for flexible matching
                stem = path.stem
                if not self.case_sensitive:
                    stem = stem.lower()
                self._cache[stem] = path

        logger.info(f"Image cache built: {len(self._cache)} entries")
        self._initialized = True

    def resolve(self, filename: str) -> Optional[Path]:
        """Resolve a single filename to its full path.

        Args:
            filename: The filename to resolve (e.g., "fusc_0001.png").

        Returns:
            The full path if found, None otherwise.
        """
        self._build_cache()

        # Try exact match first
        search_name = filename if self.case_sensitive else filename.lower()

        # Check direct match
        if search_name in self._cache:
            return self._cache[search_name]

        # Try without extension (stem match)
        stem = Path(search_name).stem
        if stem in self._cache:
            return self._cache[stem]

        # Try with different extensions
        path = Path(filename)
        stem = path.stem
        for ext in self.extensions:
            alt_name = f"{stem}{ext}"
            if alt_name in self._cache:
                return self._cache[alt_name]

        logger.warning(f"Could not resolve image: {filename}")
        return None

    def resolve_many(self, filenames: List[str]) -> Dict[str, Optional[Path]]:
        """Resolve multiple filenames at once.

        Args:
            filenames: List of filenames to resolve.

        Returns:
            Dictionary mapping filename to resolved Path (or None if not found).
        """
        self._build_cache()
        return {fname: self.resolve(fname) for fname in filenames}

    def get_unresolved(self, filenames: List[str]) -> List[str]:
        """Get list of filenames that could not be resolved.

        Args:
            filenames: List of filenames to check.

        Returns:
            List of filenames that were not found.
        """
        resolved = self.resolve_many(filenames)
        return [fname for fname, path in resolved.items() if path is None]


@dataclass
class MaskResolver:
    """Resolves mask file paths corresponding to images.

    Strategy for mask resolution:
    - Same name with _mask suffix: image.png -> image_mask.png
    - In masks/ subdirectory: image.png -> masks/image.png
    - In _masks/ subdirectory: image.png -> _masks/image.png
    - With mask_ prefix: image.png -> mask_image.png

    Attributes:
        image_resolver: Resolver for finding related images.
        search_strategies: List of strategies to try for finding masks.
    """

    image_resolver: ImageResolver
    search_strategies: tuple[str, ...] = field(
        default_factory=lambda: ("_mask", "masks/", "_masks/", "mask_")
    )
    mask_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    # FIX: Changed Dict[str, Path] → Dict[str, Optional[Path]] to allow None values
    _mask_cache: Dict[str, Optional[Path]] = field(default_factory=dict, repr=False)

    def resolve(self, image_path: Path) -> Optional[Path]:
        """Resolve the mask path for a given image path.

        Args:
            image_path: Path to the image file.

        Returns:
            Path to the mask if found, None otherwise.
        """
        cache_key = str(image_path)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        image_dir = image_path.parent
        stem = image_path.stem
        mask_path: Optional[Path] = None

        # Strategy 1: Same directory with _mask suffix
        for ext in self.mask_extensions:
            candidate = image_dir / f"{stem}_mask{ext}"
            if candidate.exists():
                mask_path = candidate
                break

        # Strategy 2: masks/ subdirectory
        if mask_path is None:
            candidate = image_dir / "masks" / image_path.name
            if candidate.exists():
                mask_path = candidate

        # Strategy 3: _masks/ subdirectory
        if mask_path is None:
            candidate = image_dir / "_masks" / image_path.name
            if candidate.exists():
                mask_path = candidate

        # Strategy 4: mask_ prefix in same directory
        if mask_path is None:
            for ext in self.mask_extensions:
                candidate = image_dir / f"mask_{stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break

        # Strategy 5: Look in parent directory's masks folder
        if mask_path is None:
            parent_masks = image_dir.parent / "masks" / image_path.name
            if parent_masks.exists():
                mask_path = parent_masks

        # FIX: Now safe to store Optional[Path] in _mask_cache
        self._mask_cache[cache_key] = mask_path
        return mask_path

    def resolve_many(self, image_paths: List[Path]) -> Dict[Path, Optional[Path]]:
        """Resolve masks for multiple images.

        Args:
            image_paths: List of image paths.

        Returns:
            Dictionary mapping image path to mask path (or None).
        """
        return {img_path: self.resolve(img_path) for img_path in image_paths}

    def get_missing_masks(self, image_paths: List[Path]) -> List[Path]:
        """Get images that don't have corresponding masks.

        Args:
            image_paths: List of image paths to check.

        Returns:
            List of image paths without masks.
        """
        resolved = self.resolve_many(image_paths)
        return [img for img, mask in resolved.items() if mask is None]