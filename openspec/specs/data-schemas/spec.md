# Data Schemas Specification

## Purpose

Typed dataclass contracts replacing hardcoded column name strings across CSV producers and consumers. Defines field names, types, and optionality for segmentation and classification records.

## Requirements

### Requirement: Segmentation Record Schema

The system MUST provide a `SegmentationRecord` dataclass with fields matching the segmentation CSV columns.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `filename` | `str` | Yes | Original image filename |
| `source` | `str` | Yes | Data source identifier |
| `split` | `str` | Yes | train/val/test |
| `image_path` | `str` | Yes | Relative path to image file |
| `mask_path` | `str` | Yes | Relative path to mask file |
| `patient_id` | `str` | No | Patient identifier (empty string if unknown) |
| `wound_percentage` | `float` | No | Wound area ratio; NaN if unknown |
| `has_wound` | `bool` | Yes | Whether wound tissue is present |
| `dataset` | `str` | Yes | Dataset name |

#### Scenario: Record from complete CSV row

- GIVEN a CSV row with all fields populated including `patient_id="P001"`, `wound_percentage=12.5`, `has_wound="True"`, `image_path="data/images/img_001.jpg"`, `mask_path="data/masks/mask_001.png"`
- WHEN `SegmentationRecord` is constructed from that row
- THEN all fields are populated with correct types; `image_path` and `mask_path` are relative strings

#### Scenario: Record with missing optional fields

- GIVEN a CSV row where `patient_id` and `wound_percentage` are empty strings
- WHEN `SegmentationRecord` is constructed
- THEN `patient_id` is `""` and `wound_percentage` is `NaN`

### Requirement: Classification Record Schema

The system MUST provide a `ClassificationRecord` dataclass with fields matching the classification CSV columns.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `filename` | `str` | Yes | Original image filename |
| `image_path` | `str` | Yes | Relative path to image file |
| `mask_path` | `str` | No | Relative path to mask (None if absent) |
| `label` | `str` | Yes | Classification label |
| `split` | `str` | Yes | train/val/test |

#### Scenario: Record with mask_path populated

- GIVEN a CSV row with `mask_path="data/masks/cls_mask_001.png"`
- WHEN `ClassificationRecord` is constructed
- THEN `mask_path` is `"data/masks/cls_mask_001.png"`

#### Scenario: Record with missing optional mask_path

- GIVEN a CSV row where `mask_path` is empty or missing
- WHEN `ClassificationRecord` is constructed
- THEN `mask_path` is `None`
