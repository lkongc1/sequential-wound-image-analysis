# Tasks: Solid Data Pipeline Refactor

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~950 (new: ~457 / modified: ~135 / tests: ~360) |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR 1: Foundation → PR 2: Config → PR 3: Data+CLI → PR 4: Integration |
| Delivery strategy | auto-forecast → auto-chain |
| Chain strategy | feature-branch-chain |

Decision needed before apply: No
Chained PRs recommended: Yes
Chain strategy: feature-branch-chain
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | PathResolver + Schemas + unit tests | PR 1 | Base: `feature/solid-pipeline` tracker. ~180 lines. |
| 2 | ConfigManager + unit tests | PR 2 | Base: PR 1 branch. Depends on PathResolver. ~290 lines. |
| 3 | DataRepository + CLI + pyproject.toml + unit tests | PR 3 | Base: PR 2 branch. ~410 lines (near budget). |
| 4 | Script fixes + datasets + .env + requirements + smoke | PR 4 | Base: PR 3 branch. ~175 lines. |

## Phase 1: Foundation — PathResolver + Schemas (WU1)

- [x] 1.1 Create `src/core/path_resolver.py` — export `PROJECT_ROOT` constant, `resolve()` and `relativize()` per path-resolver spec Requirements 1–3 (Scenarios: relative→absolute, absolute as-is, `..` normalization, outside-root ValueError)
- [x] 1.2 Create `src/data/schemas.py` — `SegmentationRecord` and `ClassificationRecord` dataclasses per data-schemas spec Requirements 1–2 (Scenarios: complete row, missing optionals, mask_path=None)

## Phase 2: Config Layer (WU2)

- [x] 2.1 Create `src/config/__init__.py` — **DEVIATION**: skipped — Python namespace collision with existing `src/config.py` module. ConfigManager placed in `src/core/config_manager.py` instead (see design-deviation note below).
- [x] 2.2 Create `src/core/config_manager.py` — `ConfigManager` with `WOUND_ENV` detection, YAML+.env deep merge (precedence: base → env-specific → `.env`), typed getters per config-manager spec Requirements 1–4 (Scenarios: all-sources, missing env YAML, missing .env, nested key access, type mismatch). **File relocated from `src/config/manager.py` to `src/core/config_manager.py` to avoid namespace collision with existing `src/config.py`.**
- [x] 2.3 Update `src/data/__init__.py` — re-export `SegmentationRecord`, `ClassificationRecord` from `.schemas`

## Phase 3: Data Layer + CLI (WU3)

- [x] 3.1 Create `src/data/repository.py` — `SegmentationDatasetRepo` and `ClassificationDatasetRepo` per data-repository spec Requirements 1–4 (Scenarios: load+resolve paths, filter-by-split, filter-by-patient, filter-by-source, FileNotFoundError, backward-compat absolute-path fallback)
- [x] 3.2 Create `src/cli.py` — `main_*` wrapper functions using `importlib` for all entry-point scripts per design D5 (scripts with leading digits can't be imported directly)
- [x] 3.3 Create `pyproject.toml` — project metadata, dependencies, 5 `[project.scripts]` entry points mapping to `src.cli:main_*` per cli-entry-points spec Requirements 1–3 (Scenarios: editable install succeeds, all commands available)

## Phase 4: Integration — Fix Pipelines (WU4)

- [x] 4.1 Fix CSV generators (`scripts/2_build_dataset.py`, `10_integrate_new_datasets.py`, `9_add_negatives.py`, `download_classification_data.py`) — import `PathResolver.relativize()`, write relative paths
- [x] 4.2 Fix datasets (`src/datasets/wound_dataset.py`, `src/datasets/classification_dataset.py`) — use `PathResolver.resolve()` for path resolution from CSV columns per design
- [x] 4.3 Fix ~10 training/inference scripts — import `PathResolver.resolve()` for CSV path resolution (search for `pd.read_csv` + `Path(p)` patterns)
- [x] 4.4 Fix `scripts/download_classification_data.py` — `os.getenv("ROBOFLOW_API_KEY")` replaces hardcoded key
- [x] 4.5 Fix `scripts/inference/predecir.py` — use `resolve("models/screening/...")` instead of hardcoded PROJECT_ROOT path

## Phase 5: Configuration Files (WU4)

- [x] 5.1 Update `.env.example` — add `WOUND_ENV`, `ROBOFLOW_API_KEY`, config precedence documentation
- [x] 5.2 Update `requirements.txt` — add `python-dotenv`, `pyyaml` as explicit dependencies

## Phase 6: Testing (spread across WUs)

- [x] 6.1 (WU1) Write `tests/unit/test_path_resolver.py` — per path-resolver spec Scenarios 1–5 (relative→absolute, absolute as-is, `..` normalization, relativize inside/outside root)
- [x] 6.2 (WU2) Write `tests/unit/test_config_manager.py` — per config-manager spec Scenarios 1–7 (all-sources, missing env, missing .env, type coercion, type error, nested keys, deep merge). 33 tests, all passing.
- [x] 6.3 (WU1) Write `tests/unit/test_schemas.py` — per data-schemas spec Scenarios 1–4 (complete row, empty optionals, mask populated, mask missing→None)
- [x] 6.4 (WU3) Write `tests/unit/test_repository.py` — per data-repository spec Scenarios 1–7 (load+resolve, split filter, patient filter, source filter, FileNotFoundError, backward-compat fallback, CSV path transparency)
- [x] 6.5 (WU4) Write `tests/smoke/test_cli.py` — per cli-entry-points spec Scenario 2 (editable install + each command invokable without `ModuleNotFoundError`)
- [x] 6.6 (WU4) Verify — run `rg "C:\\Users"` returns zero results (no machine-specific paths committed)
