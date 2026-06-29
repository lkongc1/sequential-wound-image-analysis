# Proposal: Solid Data Pipeline Refactor

## Intent

The codebase has 30+ files duplicating `PROJECT_ROOT`, 3 disconnected config systems (`.env` and `config/*.yaml` never imported), hardcoded absolute machine paths in CSVs, and no data access abstraction. Scripts call `pd.read_csv()` directly with brittle column-name contracts. A hardcoded Roboflow API key sits in source. The repo is non-portable across machines.

## Scope

### In Scope
- `src/core/path_resolver.py` — single `PROJECT_ROOT`, `resolve()`, `relativize()` functions
- `src/config/manager.py` — `ConfigManager` loading `.env` + `config/*.yaml` with env-specific overrides
- `src/data/schemas.py` — dataclass contracts for CSV columns (`SegmentationRecord`, `ClassificationRecord`)
- `src/data/repository.py` — `SegmentationDatasetRepo`, `ClassificationDatasetRepo` wrapping `pd.read_csv` + path resolution
- Fix 4 CSV generators to write relative paths via `PathResolver.relativize()`
- Fix ~10 CSV consumers to resolve paths via `PathResolver.resolve()`
- Move hardcoded Roboflow API key to `ROBOFLOW_API_KEY` env var
- Add `pyproject.toml` with `[project.scripts]` entry points
- Unit tests for new modules + smoke tests for key pipeline scripts
- Update `.env.example` with new variables and load precedence docs
- Migration: auto-detect absolute paths in CSVs and handle both formats during transition

### Out of Scope
- Script consolidation (making scripts thin orchestrators)
- DVC pipeline integration
- Rotating the exposed Roboflow API key (manual follow-up)
- Refactoring `src/config.py` dataclasses — only add ConfigManager alongside them

## Capabilities

### New Capabilities
- `path-resolver`: Centralized PROJECT_ROOT, resolve/relativize functions used by all scripts and modules
- `config-manager`: Unified config loaded from `.env` + YAML with environment-specific overrides, precedence: YAML base → YAML env → `.env`
- `data-schemas`: Dataclass contracts defining CSV column names, types, and constraints
- `data-repository`: Data access abstraction wrapping `pd.read_csv` with automatic path resolution and typed record filtering
- `cli-entry-points`: `pyproject.toml` with `[project.scripts]` entry points for key pipeline commands

### Modified Capabilities
None — no existing specs to modify.

## Approach

Introduce 4 new `src/` modules (PathResolver, ConfigManager, data schemas, data repository), then fix all CSV generators to write relative paths and all consumers to resolve through PathResolver. PathResolver auto-detects absolute paths for backward compatibility. Existing `src/config.py` dataclasses remain — ConfigManager loads alongside them. Scripts retain current structure; only path/config/data-access calls change. Smoke tests validate each pipeline phase after refactoring.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/core/path_resolver.py` | New | Central path resolution service |
| `src/config/manager.py` | New | `.env` + YAML config loader |
| `src/data/schemas.py` | New | CSV column contract dataclasses |
| `src/data/repository.py` | New | Typed data access with path resolution |
| `scripts/2_build_dataset.py` | Modified | Write relative paths via PathResolver |
| `scripts/10_integrate_new_datasets.py` | Modified | Write relative paths via PathResolver |
| `scripts/download_classification_data.py` | Modified | Relative paths + env var API key |
| `scripts/9_add_negatives.py` | Modified | Relative paths via PathResolver |
| `src/datasets/wound_dataset.py` | Modified | Resolve paths via PathResolver |
| `src/datasets/classification_dataset.py` | Modified | Resolve paths via PathResolver |
| ~10 training/inference scripts | Modified | Resolve CSV paths via PathResolver |
| `pyproject.toml` | New | Project metadata + entry points |
| `.env.example` | Modified | Document new env vars + precedence |
| `scripts/inference/predecir.py` | Modified | Remove hardcoded MODEL_PATH |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Existing absolute-path CSVs break consumers | Medium | PathResolver auto-detects absolute paths; backward-compatible during transition |
| Config loading order ambiguity | Low | Documented precedence: YAML base → YAML env → `.env` |
| Script breakage during refactor | Medium | Smoke tests per pipeline phase; incremental migration |
| PR exceeds 400-line budget | High | See chained-PR forecast; split into autonomous slices |

## Rollback Plan

1. Revert all new `src/` modules (4 files, no existing consumers before this change)
2. Revert CSV generators to write absolute paths via `Path.resolve()`
3. Revert consumers to construct `Path()` directly from CSV columns
4. Remove `pyproject.toml` entry points
5. Restore hardcoded API key to `download_classification_data.py`
6. Keep `.env.example` update (harmless) or revert

## Dependencies

- `python-dotenv` and `pyyaml` already installed
- No external system or API changes required

## Success Criteria

- [ ] `python -c "from src.core.path_resolver import PROJECT_ROOT, resolve, relativize"` succeeds
- [ ] `python -c "from src.config.manager import ConfigManager; cm = ConfigManager(); print(cm.get('paths.data_dir'))"` loads config
- [ ] All 4 CSV generators produce files with portable relative paths
- [ ] `wound_dataset.py` and `classification_dataset.py` load data from relative-path CSVs
- [ ] `scripts/download_classification_data.py` reads API key from env var (no hardcoded value)
- [ ] `pyproject.toml` entry points are invokable via `pip install -e .`
- [ ] All new modules have ≥80% unit test coverage
- [ ] Smoke tests pass for download → build → train → evaluate pipeline phases
- [ ] No machine-specific paths committed (verified via `rg "C:\\\\Users"` returning zero results)
