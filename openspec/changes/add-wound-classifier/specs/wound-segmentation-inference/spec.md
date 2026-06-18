# Wound Segmentation Inference Specification

## Purpose

CLI inference for wound segmentation with optional type classification via `--tipo` flag. Covers `predecir.py` and `predecir_yolo_unet.py`.

## Requirements

### Requirement: Baseline Segmentation CLI

The system MUST run segmentation inference via `predecir.py` and `predecir_yolo_unet.py` without a `--tipo` flag, producing mask PNGs, overlay PNGs, and instance CSVs. Existing behavior SHALL remain unchanged when `--tipo` is absent.

#### Scenario: Segmentation-only inference

- GIVEN a wound image and a trained U-Net checkpoint
- WHEN `predecir.py image.png` runs without `--tipo`
- THEN mask and overlay PNGs are saved
- AND no classifier is loaded

### Requirement: Type Classification Flag

The system MUST accept `--tipo CHECKPOINT` in `predecir.py` and `predecir_yolo_unet.py`. When present, after segmentation each wound instance SHALL be classified using the checkpoint model. Output SHALL be written to `{image_stem}_tipo.txt` as one line per instance: `{instance_id}: {class_name} ({confidence:.2f})`.

#### Scenario: Single wound classified

- GIVEN an image with one wound and `--tipo models/classifier/best.pth`
- WHEN the pipeline runs YOLO→U-Net→Classifier
- THEN `imagen_tipo.txt` contains one line: `1: laceración (0.87)`

#### Scenario: Multiple wounds classified per-instance

- GIVEN an image with 3 YOLO-detected wounds and `--tipo`
- WHEN the pipeline runs
- THEN `_tipo.txt` contains 3 lines, one per instance
- AND each line includes the instance ID assigned by the instance separation step

#### Scenario: --tipo with --instancias flag

- GIVEN `--tipo` and `--instancias` both active
- WHEN inference runs
- THEN instance IDs in `_tipo.txt` SHALL match those in `_instancias.csv`

### Requirement: Abstention on Low Confidence

The system MUST return "desconocido" when the classifier's top-1 confidence is below `confidence_threshold` (default 0.5). No forced classification SHALL be made below threshold.

#### Scenario: Low-confidence input triggers abstention

- GIVEN a wound crop the classifier assigns max confidence 0.32
- WHEN `confidence_threshold=0.5`
- THEN the output is `1: desconocido (0.32)`

#### Scenario: Confidence exactly at threshold

- GIVEN top-1 confidence is exactly 0.50
- WHEN threshold is 0.5
- THEN the predicted class name is returned (not "desconocido")

### Requirement: No-Detection Classifier Skip

The system MUST skip classifier inference entirely when YOLO detects zero wounds. In `predecir_yolo_unet.py`, when no bboxes are found (empty or fallback full-image pass), the classifier SHALL NOT run. In `predecir.py`, when the mask has zero positive pixels, the classifier SHALL NOT run. In both cases, `_tipo.txt` SHALL be written with the content `SIN_DETECCION`.

#### Scenario: Empty image, YOLO detects nothing

- GIVEN an image with no wounds and `--tipo` active
- WHEN the pipeline runs
- THEN no classifier forward pass occurs
- AND `_tipo.txt` contains `SIN_DETECCION`
