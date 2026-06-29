#!/usr/bin/env python3
"""Final validation tests for SOLID structure."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    print("=" * 60)
    print("TESTING SOLID STRUCTURE IMPORTS")
    print("=" * 60)

    errors = []

    # Test src.data facade
    print("\n[DATA FASCADE]")
    try:
        from src.data import (
            ImageStatisticsCalculator,
            MaskStatisticsCalculator,
            DatasetAuditor,
            AuditConfig,
            ImageQualityPipeline,
            ResolutionChecker,
            KaggleSource,
            normalize_image,
        )
        print("  OK: All data facade imports")
    except ImportError as e:
        errors.append(f"Data facade: {e}")
        print(f"  ERR: {e}")

    # Test EDA
    print("\n[EDA]")
    try:
        from src.data.eda import (
            CSVReporter,
            DatasetStatisticsCalculator,
            ImageStatisticsCalculator,
            JSONReporter,
            MaskStatisticsCalculator,
            WoundDistributionVisualizer,
        )
        _ = (DatasetStatisticsCalculator, JSONReporter)
        print("  OK: EDA module")
    except ImportError as e:
        errors.append(f"EDA: {e}")
        print(f"  ERR: {e}")

    # Test Audit
    print("\n[AUDIT]")
    try:
        from src.data.audit import AuditConfig, DatasetAuditor
        auditor = DatasetAuditor(
            config=AuditConfig(
                data_dir=Path("data"),
                output_dir=Path("output"),
            ),
            image_calculator=ImageStatisticsCalculator(),
            mask_calculator=MaskStatisticsCalculator(),
            visualizer=WoundDistributionVisualizer(Path("output")),
            csv_reporter=CSVReporter(),
        )
        assert auditor is not None
        print("  OK: DatasetAuditor instantiated with DIP")
    except Exception as e:
        errors.append(f"Audit instantiation: {e}")
        print(f"  ERR: {e}")

    # Test Quality
    print("\n[QUALITY]")
    try:
        from src.data.quality import ImageQualityPipeline, ResolutionChecker
        pipeline = ImageQualityPipeline()
        pipeline.add_checker(ResolutionChecker())
        print("  OK: ImageQualityPipeline with OCP")
    except Exception as e:
        errors.append(f"Quality: {e}")
        print(f"  ERR: {e}")

    # Test Sources
    print("\n[SOURCES]")
    try:
        from src.data.sources import KaggleSource
        source = KaggleSource(dataset_slug="test/dataset")
        print(f"  OK: KaggleSource({source.name})")
    except Exception as e:
        errors.append(f"Sources: {e}")
        print(f"  ERR: {e}")

    # Test Transforms
    print("\n[TRANSFORMS]")
    try:
        from src.data.transforms import get_training_augmentation, normalize_image
        _ = (normalize_image, get_training_augmentation)
        print("  OK: Transforms module")
    except ImportError as e:
        errors.append(f"Transforms: {e}")
        print(f"  ERR: {e}")

    # Test Logging
    print("\n[LOGGING]")
    try:
        from src.utils.logging import AuditLogger, get_logger, setup_logging
        _ = (setup_logging, AuditLogger, get_logger)
        print("  OK: Logging utilities")
    except ImportError as e:
        errors.append(f"Logging: {e}")
        print(f"  ERR: {e}")

    # Test new scripts
    print("\n[SCRIPTS]")
    scripts = [
        "scripts/pipeline/download_dataset.py",
        "scripts/2_audit_dataset.py",
        "scripts/3_visualize_data.py",
    ]
    for script in scripts:
        path = PROJECT_ROOT / script
        if path.exists():
            print(f"  OK: {script}")
        else:
            errors.append(f"Missing: {script}")
            print(f"  ERR: Missing {script}")

    # Test old files removed
    print("\n[OLD FILES REMOVED]")
    old_files = [
        "src/data/acquisition/kaggle.py",
        "src/data/validators.py",
        "src/data/eda.py",
        "src/data/quality_control.py",
        "src/models/losses.py",
        "src/utils/logging_config.py",
        "scripts/0.cleanup_duplicates.py",
        "scripts/1.validate_structure.py",
        "scripts/1.download_dataset.py",
        "scripts/2.audit_dataset.py",
        "scripts/3.data_visualiser.py",
    ]
    for file_path in old_files:
        path = PROJECT_ROOT / file_path
        if not path.exists():
            print(f"  OK: Removed {file_path}")
        else:
            errors.append(f"Still exists: {file_path}")
            print(f"  ERR: Still exists {file_path}")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAIL: {len(errors)} ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("SUCCESS: ALL SOLID STRUCTURE TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(test_imports())
