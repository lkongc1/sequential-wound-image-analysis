# CLI Entry Points Specification

## Purpose

`pyproject.toml` with `[project.scripts]` section defining installable CLI entry points for the pipeline. Enables `pip install -e .` workflow so all commands are available in the PATH without manual `PYTHONPATH` manipulation.

## Requirements

### Requirement: Pipeline Entry Points

The system MUST define five CLI entry points in `pyproject.toml` under `[project.scripts]`, each mapping a command name to a callable function within the `scripts` package.

| Command | Target Module | Description |
|---------|--------------|-------------|
| `wound-download` | `scripts.download_classification_data:main` | Download wound classification data |
| `wound-build-dataset` | `scripts.2_build_dataset:main` | Build segmentation dataset |
| `wound-train` | `scripts.3_train_wound_segmentation:main` | Train wound segmentation model |
| `wound-evaluate` | `scripts.4_evaluate:main` | Evaluate trained model |
| `wound-infer` | `scripts.inference.predecir:main` | Run inference on new images |

#### Scenario: All entry points registered

- GIVEN `pyproject.toml` contains the `[project.scripts]` section with all five commands
- WHEN `pip install -e .` is executed successfully
- THEN all five commands are available in the shell environment

#### Scenario: Entry point invokable from command line

- GIVEN the package is installed in editable mode
- WHEN `wound-download --help` is invoked
- THEN the script's help text or expected output is displayed (no `ModuleNotFoundError`)

### Requirement: Editable Install Workflow

The `pyproject.toml` SHALL include complete project metadata (`name`, `version`, `dependencies`, `requires-python`) sufficient for `pip install -e .` to succeed without additional configuration.

#### Scenario: Editable install succeeds

- GIVEN a fresh virtual environment with Python ≥ 3.10
- WHEN `pip install -e .` is executed from the project root
- THEN the package installs without errors and all dependencies are resolved

### Requirement: Script Functions Exposed

Each entry point target function MUST be importable at the dotted path specified in `[project.scripts]`. Functions SHALL accept zero arguments (argparse handles CLI parsing internally).

#### Scenario: Target function is importable

- GIVEN `pyproject.toml` maps `wound-build-dataset` to `scripts.2_build_dataset:main`
- WHEN `from scripts.2_build_dataset import main` is executed
- THEN the import succeeds and `main` is a callable

#### Scenario: Missing target function fails install

- GIVEN `pyproject.toml` references a non-existent target function
- WHEN `pip install -e .` is executed
- THEN installation fails with a clear error identifying the missing function
