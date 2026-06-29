# Design: Solid Data Pipeline Refactor

## Technical Approach

Introduce 4 standalone modules at the bottom of the dependency graph, then thread them upward through scripts and datasets. PathResolver replaces 30+ duplicated `PROJECT_ROOT` computations. ConfigManager unifies `.env` + YAML config loading. Data schemas define typed CSV contracts. DataRepository wraps `pd.read_csv` with automatic path resolution. Existing `src/config.py` dataclasses remain untouched per proposal scope — ConfigManager coexists alongside them.

## Architecture Decisions

| Decision | Choice | Rejected | Rationale |
|----------|--------|----------|-----------|
| **D1: ConfigManager vs. existing dataclasses** | **Option C**: coexist. ConfigManager for new code, dataclasses keep `__post_init__` pattern | A. Replace entirely (breaking, out of scope) / B. Dataclasses consume ConfigManager via `__post_init__` (coupling, out of scope) | Proposal explicitly excludes refactoring `src/config.py`. Each script imports what it needs. No migration risk. |
| **D2: PathResolver.`resolve()` auto-detection** | **Two-layer**: `PathResolver` uses simple `path.is_absolute()` per spec. `DataRepository` wraps with existence-check fallback. | B. Always resolve against PROJECT_ROOT (breaks valid absolute external paths) / C. Combine in one function (violates spec clarity, complicates pure function) | `is_absolute()` is Python's built-in — works correctly on all platforms. The fallback lives at the data layer where file existence is known, keeping PathResolver pure and spec-compliant. |
| **D3: Config loading precedence** | `WOUND_ENV` env var detects environment (default: `development`). Deep merge for nested YAML. `.env` overrides with `__` as separator for nested keys (e.g., `PATHS__DATA_DIR`). | `APP_ENV` (too generic), `NODE_ENV` (node-specific), shallow merge (blows away sibling keys) | Deep merge preserves keys not overridden. `WOUND_ENV` is project-specific. Double-underscore separator avoids ambiguity with single-underscore YAML keys. |
| **D4: Module dependency graph** | `PathResolver` ← `ConfigManager` (for config file paths) ← `DataRepository` (for CSV + record paths). `schemas` is leaf with zero deps. No circular dependencies. | ConfigManager depends on DataRepository (unnecessary) / DataRepository depends on ConfigManager for defaults (tight coupling) | PathResolver is the foundation — both ConfigManager and DataRepository need it. ConfigManager resolves YAML paths. DataRepository resolves CSV paths + record paths. Schemas are pure data. |
| **D5: pyproject.toml entry points** | Thin `src/cli.py` wrapper with `main_*` functions that import and call script entry points via `importlib`. Entry points map to `src.cli:main_build_dataset` etc. | Direct `[project.scripts]` mapping to `scripts.2_build_dataset:main` (invalid Python: module names can't start with digits) / Rename all scripts (breaking) | Scripts stay as-is. `cli.py` bridges the gap. Adds one file, zero script changes for CLI mapping. |
| **D6: Backward compatibility** | DataRepository resolves paths via `PathResolver.resolve()`, then checks `path.exists()`. If false and path is absolute, tries `PROJECT_ROOT / path.name` as fallback. Logs warning when fallback fires. | Handle both formats forever (unclean state) / Drop absolute-path support immediately (breaks existing CSVs) | Phase 1 (this PR): fallback active. Phase 2 (~2 weeks): remove fallback, add deprecation warning. Phase 3: strict resolve only. |

## Data Flow

```
CSV Generator (script)              CSV Consumer (dataset/training)
        │                                      │
        │ PathResolver.relativize()            │ DataRepository.load()
        ▼                                      ▼
  Relative paths in CSV ────────▶ CSV file ◀── PathResolver.resolve()
  "data/images/img_001.jpg"                     → PROJECT_ROOT / "data/images/img_001.jpg"
                                                → (or as-is if absolute + exists)
```

```
ConfigManager ──▶ YAML base + YAML env + .env ──▶ unified dict
       │
       ├── cm.get_str("paths.raw") → "data/raw/data_wound_seg"
       └── cm.get_int("training.batch_size") → 8
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/core/path_resolver.py` | Create | `PROJECT_ROOT`, `resolve()`, `relativize()` |
| `src/config/__init__.py` | Create | Package init for config module |
| `src/config/manager.py` | Create | `ConfigManager` with YAML+.env deep merge, typed getters |
| `src/data/schemas.py` | Create | `SegmentationRecord`, `ClassificationRecord` dataclasses |
| `src/data/repository.py` | Create | `SegmentationDatasetRepo`, `ClassificationDatasetRepo` |
| `src/cli.py` | Create | Wrapper functions for `[project.scripts]` entry points |
| `src/data/__init__.py` | Modify | Add schema + repo exports |
| `pyproject.toml` | Create | Project metadata, deps, entry points |
| `scripts/2_build_dataset.py` | Modify | Import PathResolver, write relative paths |
| `scripts/10_integrate_new_datasets.py` | Modify | Import PathResolver, write relative paths |
| `scripts/download_classification_data.py` | Modify | `os.getenv("ROBOFLOW_API_KEY")` + relative paths |
| `scripts/9_add_negatives.py` | Modify | Import PathResolver, write relative paths |
| `src/datasets/wound_dataset.py` | Modify | `create_dataset_from_csv` uses DataRepository |
| `src/datasets/classification_dataset.py` | Modify | Constructor uses DataRepository |
| `~10 training/inference scripts` | Modify | Import PathResolver for CSV path resolution |
| `.env.example` | Modify | Add `WOUND_ENV`, `ROBOFLOW_API_KEY`, precedence docs |
| `requirements.txt` | Modify | Add `python-dotenv`, `pyyaml` as explicit deps |

## Interfaces

```python
# PathResolver (src/core/path_resolver.py)
PROJECT_ROOT: Path  # resolves to repo root at import time
def resolve(path: str | Path) -> Path: ...
def relativize(path: str | Path) -> str: ...  # raises ValueError if outside PROJECT_ROOT

# ConfigManager (src/config/manager.py)
class ConfigManager:
    def __init__(self, env: str | None = None, project_root: Path | None = None): ...
    def get_str(self, key: str, default: str | None = None) -> str: ...
    def get_int(self, key: str, default: int | None = None) -> int: ...
    def get_float(self, key: str, default: float | None = None) -> float: ...
    def get_bool(self, key: str, default: bool | None = None) -> bool: ...
    def get_path(self, key: str, default: Path | None = None) -> Path: ...

# DataRepository (src/data/repository.py)
class SegmentationDatasetRepo:
    def __init__(self, csv_path: str | Path): ...  # raises FileNotFoundError
    def get_all(self) -> list[SegmentationRecord]: ...
    def get_by_split(self, split: str) -> list[SegmentationRecord]: ...
    def get_by_patient(self, patient_id: str) -> list[SegmentationRecord]: ...

class ClassificationDatasetRepo:
    def __init__(self, csv_path: str | Path): ...
    def get_all(self) -> list[ClassificationRecord]: ...
    def get_by_source(self, source: str) -> list[ClassificationRecord]: ...
```

## Testing Strategy

| Layer | What | Tool |
|-------|------|------|
| Unit | `resolve()` / `relativize()` edge cases (absolute, relative, `..`, outside root) | pytest + tmp_path |
| Unit | `ConfigManager` deep merge, missing files, type coercion, nested key access | pytest + tmp YAML/.env fixtures |
| Unit | `SegmentationRecord` / `ClassificationRecord` from dicts, missing optionals | pytest |
| Unit | `DataRepository` CSV loading, filter methods, missing CSV error, backward-compat fallback | pytest + tmp CSV |
| Integration | `DataRepository` + real CSVs from `data/processed/` | pytest (requires data) |
| Smoke | `pip install -e .` + each CLI entry point invoked | manual / CI |

## Migration / Rollout

**Phase 1 (this PR)**: All 4 modules created. CSV generators write relative paths. Consumers resolve through PathResolver + DataRepository fallback. Both absolute and relative CSVs work. Roboflow key moved to env var.

**Phase 2 (follow-up PR)**: Regenerate all CSVs with relative paths. Remove existence-check fallback from DataRepository. Add `DeprecationWarning` for absolute paths in `resolve()`.

**Rollback**: Revert 4 new modules + cli.py + pyproject.toml. Revert path writing lines in CSV generators. Restore API key. Existing dataclasses untouched — no rollback needed for them.

## Open Questions

- [ ] Is `WOUND_ENV` the right env var name, or should it be configurable in `.env` itself?
- [ ] Should `ConfigManager` also read from `config/defaults.yaml` (not currently in repo — only `data_config.yaml` and `model_config.yaml` exist)?
