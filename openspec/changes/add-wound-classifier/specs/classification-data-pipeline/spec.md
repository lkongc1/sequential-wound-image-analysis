# Classification Data Pipeline Specification

## Purpose

Download, normalize, balance, and split 7-class wound classification data from Roboflow into `data-clasificador/`.

## Requirements

### Requirement: Roboflow Dataset Download

The system MUST download three Roboflow datasets (SmartHeal, Basic Wound Classify, Burn Wound Classification) via the `roboflow` Python SDK, using API key `DVnQXr1udAOhcikIJWnJ`, into `data-clasificador/`. Downloaded images MUST be cached locally after first download.

#### Scenario: Successful download

- GIVEN a valid Roboflow API key
- WHEN the download script runs
- THEN images from all 3 datasets are saved under `data-clasificador/raw/`
- AND a manifest CSV logs source dataset and count per class

#### Scenario: Network failure with local cache

- GIVEN a previous download completed successfully
- WHEN the script runs without network access
- THEN it SHALL use the cached `data-clasificador/raw/` directory
- AND print a warning that no fresh data was fetched

### Requirement: Label Normalization

The system MUST map English Roboflow labels to the 7-class Spanish taxonomy via an explicit mapping table. Any label not in the mapping SHALL cause the script to fail with a descriptive error listing the unknown label.

| English Source       | Spanish Target |
|----------------------|----------------|
| abrasion             | raspón         |
| bruise               | hematoma       |
| burn                 | quemadura      |
| cut                  | corte          |
| laceration           | laceración     |
| stab                 | punción        |
| normal_skin          | piel_sana      |

#### Scenario: All labels recognized

- GIVEN Roboflow datasets using known English labels
- WHEN labels are normalized
- THEN ALL labels map successfully to the 7-class Spanish taxonomy

#### Scenario: Unknown label encountered

- GIVEN a Roboflow dataset contains "avulsion" which is not in the taxonomy
- WHEN normalization runs
- THEN the script MUST fail with an error message listing "avulsion" as unknown
- AND suggest adding it to the mapping or excluding the class

### Requirement: Class Balancing

The system SHALL undersample majority classes to match the minority class target of ~600 images per class. Images for removal SHALL be selected randomly with a fixed seed for reproducibility.

#### Scenario: Majority class exceeds target

- GIVEN class "quemadura" has 2000 images and class "punción" has 150
- WHEN balancing is applied targeting ~600/class
- THEN "quemadura" is randomly undersampled to ~600
- AND "punción" is kept at 150 (below target)
- AND the total dataset size is logged

### Requirement: Stratified Train/Val/Test Split

The system MUST split the balanced dataset into 70% train, 15% validation, 15% test using stratified sampling to preserve class distribution. Split manifests SHALL be saved as `train.csv`, `val.csv`, `test.csv` in `data-clasificador/` with columns `image_path` and `label`.

#### Scenario: Stratified split preserves ratios

- GIVEN a balanced dataset with 7 classes
- WHEN 70/15/15 stratified split is performed
- THEN each split contains all 7 classes
- AND per-class ratios deviate <2% from the global class distribution
