# Data Repository Specification

## Purpose

Typed data access layer wrapping `pd.read_csv()` with automatic path resolution via `PathResolver`. Consumers query typed records without knowing CSV file locations or performing manual path construction.

## Requirements

### Requirement: Segmentation Dataset Repository

The system MUST provide a `SegmentationDatasetRepo` class that loads a segmentation CSV, resolves all paths, and returns `SegmentationRecord` instances.

#### Scenario: Load segmentation CSV with relative paths

- GIVEN a CSV at `data/processed/segmentation.csv` containing rows with relative `image_path` and `mask_path` columns
- WHEN `SegmentationDatasetRepo(csv_path="data/processed/segmentation.csv")` is instantiated and records are iterated
- THEN every record's `image_path` and `mask_path` are absolute `Path` objects resolved from the relative CSV values

#### Scenario: Filter by split

- GIVEN a loaded segmentation repo with `train`, `val`, and `test` records
- WHEN `repo.get_by_split("train")` is called
- THEN only records where `split == "train"` are returned

#### Scenario: Filter by patient

- GIVEN a loaded segmentation repo with records for patients P001, P002, P003
- WHEN `repo.get_by_patient("P001")` is called
- THEN only records where `patient_id == "P001"` are returned

### Requirement: Classification Dataset Repository

The system MUST provide a `ClassificationDatasetRepo` class that loads a classification CSV, resolves paths, and returns `ClassificationRecord` instances. The `mask_path` field SHALL be `None` when the CSV column is empty.

#### Scenario: Load classification CSV

- GIVEN a classification CSV at `data/processed/classification.csv`
- WHEN `ClassificationDatasetRepo(csv_path="data/processed/classification.csv")` is instantiated
- THEN all records are returned as `ClassificationRecord` instances with resolved absolute `image_path`

#### Scenario: Filter by source

- GIVEN a loaded classification repo with records from sources `"roboflow"` and `"custom"`
- WHEN `repo.get_by_source("roboflow")` is called
- THEN only records with `source == "roboflow"` are returned

### Requirement: CSV Path Resolution Transparency

Consumers of repositories SHALL NOT construct or resolve CSV paths manually. The repository constructor accepts a config-relative or absolute path and resolves it internally through the `PathResolver`.

#### Scenario: Repository resolves its own CSV path

- GIVEN `SegmentationDatasetRepo` is constructed with a relative CSV path
- WHEN the repository loads the CSV
- THEN it resolves the relative path to absolute internally; consumers never call `PathResolver` directly for the CSV path

### Requirement: Missing CSV Handling

The repository SHALL raise a clear `FileNotFoundError` when the CSV file does not exist after path resolution.

#### Scenario: CSV file does not exist

- GIVEN `csv_path="data/missing.csv"` that does not resolve to an existing file
- WHEN `SegmentationDatasetRepo` is instantiated
- THEN `FileNotFoundError` is raised with a message including the resolved path
