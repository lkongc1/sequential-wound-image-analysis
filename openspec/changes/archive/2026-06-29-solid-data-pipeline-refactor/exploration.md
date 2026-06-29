## Exploration: solid-data-pipeline-refactor

### Current State

The codebase has a split personality: the `src/` package applies SOLID principles well (dataclass configs, factory pattern, dependency injection, abstract bases), but the 34 scripts in `scripts/` operate as a loosely coupled chain of ad-hoc Python files. Every script duplicates path resolution, CSV reading, and configuration logic. The root cause of the PR issue — hardcoded `C:\Users\LEGIO\...` paths — is that `scripts/2_build_dataset.py`, `scripts/10_integrate_new_datasets.py`, and `scripts/download_classification_data.py` use `Path.resolve()` when writing CSV columns, embedding absolute machine-specific paths into the generated artifact files (dataset CSVs). These CSVs are then consumed by datasets (`src/datasets/wound_dataset.py`, `src/datasets/classification_dataset.py`), training scripts, inference scripts, and YOLO preparation — making the repo non-portable.

Three config systems coexist disconnected: `.env` + `config/*.yaml` files (never imported by any code), `src/config.py` dataclasses (used by comparative eval and classifier training), and hardcoded constants in every script. There is no Makefile, `pyproject.toml`, or DVC pipeline locking — the only automation is `run_all_training.py` which shell-calls other scripts.

### Affected Areas

#### Generators (produce CSVs with absolute paths)
- `scripts/2_build_dataset.py:96-97` — `str(img_path.resolve())` and `str(mask_path.resolve())` → writes `dataset_final.csv`
- `scripts/10_integrate_new_datasets.py:117-118,201` — `str(path.resolve())` → writes `co2wounds_v2_integrated.csv`, `classification_dataset.csv`
- `scripts/download_classification_data.py:185,261` — `str(img_file.resolve())` → writes `train.csv`, `val.csv`, `test.csv`
- `scripts/9_add_negatives.py:39-40` — uses non-resolved `str(path)`, slightly better but still fragile

#### Consumers (read CSV paths)
- `src/datasets/wound_dataset.py:244-268` — `create_dataset_from_csv()` reads `image_path`, `mask_path` columns as `Path(p)`
- `src/datasets/classification_dataset.py:182-206` — `ClassificationDataset.__getitem__()` reads `image_path`, `mask_path`
- `scripts/4_train_models.py:66-67` — `df["image_path"]`, `df["mask_path"]`
- `scripts/6_train_unet_final.py:121-122` — same pattern
- `scripts/6b_train_unet_r18.py:121` — same pattern
- `scripts/11_screening_architectures.py:278,284` — same pattern
- `scripts/train_individual.py:82,88` — same pattern
- `scripts/generate_pseudo_masks.py:178-179` — reads `image_path`, writes `mask_path`
- `scripts/8_prepare_yolo.py:129` — `df["image_path"]`
- `scripts/inference/predecir.py:40` — hardcoded `MODEL_PATH`

#### Configuration landscape (disconnected)
- `.env` (21 lines) — never imported by any Python code
- `.env.example` (26 lines) — template, also unused
- `config/data_config.yaml` — data paths, preprocessing, augmentation — never imported
- `config/model_config.yaml` — model/loss hyperparameters — never imported
- `config/clinical_config.yaml` — FDA/clinical thresholds — never imported
- `config/environments/development.yaml` — dev env overrides — never imported
- `config/environments/production.yaml` — prod env overrides — never imported
- `src/config.py` — 6 dataclasses (ComparativeConfig, QualityConfig, PathsConfig, EDAConfig, TrainingConfig, ClassificationConfig, InstanceConfig) — used by comparative eval and classifier training, but ignored by most scripts
- `src/core/constants.py` — standalone constants, no connection to config system

#### SOLID violations (specifics)
- **SRP violation — path resolution duplicated**: 30+ files define `PROJECT_ROOT = Path(__file__).resolve().parent.parent` independently
- **SRP violation — CSV column contracts**: Every script hardcodes column names (`"image_path"`, `"mask_path"`, `"split"`) — no centralized CSV schema
- **SRP violation — DataLoader creation duplicated**: `scripts/4_train_models.py`, `scripts/6_train_unet_final.py`, `scripts/11_screening_architectures.py`, `scripts/train_individual.py` all independently build DataLoaders from the same CSV
- **Hardcoded API key**: `scripts/download_classification_data.py:34` — `API_KEY: str = "DVnQXr1udAOhcikIJWnJ"` (security issue)
- **No DIP for data access**: Scripts directly call `pd.read_csv()` with hardcoded paths instead of depending on a data access abstraction
- **Missing ConfigManager**: No single source of truth for config — values are split across `.env` (unused), `config/*.yaml` (unused), dataclass defaults, and script constants

#### Scripts by pipeline phase
| Phase | Script | Manual/Auto | Depends On | Produces |
|-------|--------|-------------|------------|----------|
| **Download** | `1_download_dataset.py` | Manual | KaggleSource | `data/raw/data_wound_seg/` |
| **Download** | `download_classification_data.py` | Manual | Roboflow SDK | `data-clasificador/{train,val,test}.csv` |
| **Build** | `2_build_dataset.py` | Manual | `1_download` | `data/processed/dataset_final.csv` |
| **Build** | `10_integrate_new_datasets.py` | Manual | `2_build` | updates `dataset_final.csv` + `classification_dataset.csv` |
| **Build** | `9_add_negatives.py` | Manual | `2_build` | adds negatives to `dataset_final.csv` |
| **Build** | `8_prepare_yolo.py` | Manual | `2_build` | `data/yolo/` |
| **Build** | `generate_pseudo_masks.py` | Manual | screening model | updates classification CSVs with mask_path |
| **EDA** | `3_eda.py` | Manual | `2_build` | console report only |
| **Train** | `4_train_models.py` | Manual | `2_build` | `models/{name}_final.pth` (4 models) |
| **Train** | `6_train_unet_final.py` | Manual | `2_build` | `models/unet_final_pretrained.pth` |
| **Train** | `11_screening_architectures.py` | Manual | `2_build` | `models/screening/` (12 combos) |
| **Train** | `train_classifier.py` | Manual | `download_classification_data` | `models/classifier/best.pth` |
| **Train** | `train_*.py` (10 variant scripts) | Manual | `2_build` | `models/screening/` |
| **Train** | `run_all_training.py` | Semi-auto | all training scripts | orchestrates via subprocess |
| **Train** | `monitor_training.py` | Manual | training dirs | console monitoring |
| **Eval** | `5_evaluate.py` | Manual | `4_train` | console report |
| **Eval** | `7_evaluate_pretrained.py` | Manual | pretrained models | console report |
| **Infer** | `inference/predecir.py` | Manual | `models/screening/FPN_EfficientNetB3_best.pth` | output images |
| **Infer** | `inference/predecir_yolo_unet.py` | Manual | YOLO + U-Net models | output images |
| **Infer** | `inference/predecir_yolo_seg.py` | Manual | YOLO seg models | output images |
| **Infer** | `inference/comparar_modelos_screening.py` | Manual | screening models | comparison images |
| **Infer** | `inference/test_instancias.py` | Manual | instance models | test output |
| **Infer** | `inference/test_miimage.py` | Manual | models | test output |

### Approaches

#### 1. Minimal Relative-Path Fix (CSV Relativization)

Change all CSV generators to write paths relative to PROJECT_ROOT, and ensure all consumers resolve relative paths against PROJECT_ROOT.

**What changes:**
- `scripts/2_build_dataset.py`, `scripts/10_integrate_new_datasets.py`, `scripts/download_classification_data.py`, `scripts/9_add_negatives.py` — replace `str(path.resolve())` with `str(path.relative_to(PROJECT_ROOT))` or store just the relative path
- `src/datasets/wound_dataset.py:create_dataset_from_csv()` — add `Path(csv_path).parent / p` resolution for relative paths in columns
- `src/datasets/classification_dataset.py:ClassificationDataset.__getitem__()` — same relative resolution
- All training/inference scripts that build Path objects from CSV — centralize resolution

**Pros:**
- Smallest possible change: ~10 files, ~30 lines changed
- CSVs become portable (machine-independent)
- Low risk of breaking existing workflows
- Can be completed in one session

**Cons:**
- Does not fix the config chaos (3 disconnected systems)
- Does not fix duplicated logic across scripts
- Does not fix the hardcoded API key
- Adds ambiguity: consumer must know what root to resolve against
- No automation improvements

**Effort:** Low

---

#### 2. Centralized Config + PathResolver Abstraction (Recommended)

Introduce a `ConfigManager` that loads `.env` and YAML configs, provide a `PathResolver` service that handles path resolution uniformly, centralize CSV schema as a dataclass, and relocate shared data pipeline logic into `src/data/` modules that scripts call.

**What changes:**
1. **Config unification** — Create `src/config/` package:
   - `ConfigManager` — reads `.env`, merges with `config/*.yaml`, exposes typed config
   - Loads environment-specific overrides from `config/environments/{ENV}.yaml`
   - Replaces `src/config.py` dataclass defaults with config-loaded values
   - `src/core/constants.py` folded into config system

2. **PathResolver service** — Create `src/core/path_resolver.py`:
   - Single `PROJECT_ROOT` constant from `src/core/__init__.py`
   - `resolve_data_path(relative_path: str) -> Path` — project-root-relative
   - `relativize_path(absolute_path: Path) -> str` — stores relative in CSVs
   - Used by ALL scripts and modules

3. **CSV schema dataclass** — Create `src/data/schemas.py`:
   ```python
   @dataclass
   class SegmentationRecord:
       filename: str
       source: str
       split: str
       image_path: str  # relative
       mask_path: str   # relative
       wound_percentage: float
       # ... all columns
   ```
   - Single source of truth for column names and types

4. **Data access layer** — Create `src/data/repository.py`:
   - `SegmentationDatasetRepo` — wraps `pd.read_csv()` + path resolution + filtering
   - `ClassificationDatasetRepo` — wraps classification CSVs + path resolution
   - Used by all dataset classes and scripts instead of raw `pd.read_csv()`

5. **Script refactoring** — Each script becomes thin:
   - Import config, path resolver, repository from `src/`
   - Remove duplicated `PROJECT_ROOT` and path construction
   - Remove hardcoded hyperparameters (use config)
   - Remove `API_KEY` from `download_classification_data.py` (use env var)

6. **Automation** — Add `pyproject.toml` with `[project.scripts]`:
   ```toml
   [project.scripts]
   download-dataset = "scripts.1_download_dataset:main"
   build-dataset = "scripts.2_build_dataset:main"
   train = "scripts.train:main"
   ```

**Pros:**
- Single source of truth for paths, config, CSV schemas
- Scripts become thin orchestrators calling src/ modules
- CSVs become portable (relative paths)
- Fixes all SOLID violations (SRP, DIP, OCP)
- Enables future DVC pipeline integration
- Config changes propagate everywhere without touching scripts
- `.env` + YAML files actually work

**Cons:**
- High effort: ~30+ files changed
- Risk of breaking existing scripts during refactor
- Requires careful testing of every pipeline phase
- Migration of existing absolute-path CSVs (backward compat needed)
- May exceed 400-line PR budget

**Effort:** High

---

#### 3. Hybrid: Path Fix + Config Bootstrap

Fix paths first (Approach 1), then bootstrap the config system incrementally without touching all scripts at once.

**What changes:**
- **Phase A** (this change): Fix absolute paths in all CSV generators, add `PathResolver` in `src/core/`, add relative-path resolution to consumers
- **Phase B** (follow-up): Create `ConfigManager`, migrate scripts one-by-one
- **Phase C** (follow-up): Create data repository layer, migrate scripts

**Pros:**
- Lower immediate risk than Approach 2
- Fixes the PR complaint directly
- Scaffolds infrastructure for future cleanup
- Can be done in smaller PRs

**Cons:**
- Three changes instead of one
- Config chaos persists between phases
- Duplicated logic persists until Phase C

**Effort:** Medium (Phase A only), High (all phases)

### Recommendation

**Approach 2 (Centralized Config + PathResolver)**, but scoped to the core path/config/data-access issues, leaving non-essential script consolidation for follow-up. The key deliverables:

1. **`src/core/path_resolver.py`** — one global `PROJECT_ROOT`, `resolve()`, `relativize()` functions
2. **`src/config/manager.py`** — `ConfigManager` that loads `.env` + `config/*.yaml` → typed config
3. **`src/data/schemas.py`** — dataclass for CSV column contracts
4. **`src/data/repository.py`** — `SegmentationDatasetRepo`, `ClassificationDatasetRepo`
5. **Fix all 4 CSV generators** to write relative paths
6. **Fix all CSV consumers** (datasets + scripts) to resolve paths via PathResolver
7. **Remove hardcoded API key** → `.env`
8. **Add `pyproject.toml`** with `[project.scripts]` and project metadata

The reason for choosing this over Approach 3: the config system scaffolding is necessary to make the path fix robust — without a central PathResolver, every consumer needs its own relative-to-absolute logic, which is fragile. The config unification is the enabler, not a nice-to-have.

### Risks

- **Existing absolute-path CSVs break**: Users with existing `dataset_final.csv` containing absolute paths will need to regenerate or run a migration script. Mitigation: auto-detect absolute paths and resolve them regardless, or provide a `migrate_csvs.py` script.
- **CSV column contract changes**: If `image_path` changes from absolute to relative, all scripts reading it must use PathResolver. Mitigation: the repository layer auto-resolves.
- **Config loading order ambiguity**: `.env` vs YAML vs CLI args precedence must be defined. Mitigation: YAML base → YAML env overrides → `.env` overrides → CLI args (most specific wins).
- **Script breakage during refactor**: 34 scripts depend on current path resolution. Mitigation: make PathResolver backward-compatible — if a path is already absolute, return it as-is.
- **Regeneration required**: `data-clasificador/{train,val,test}.csv` are tracked in git (not ignored per `.gitignore`). After the fix they must be regenerated with relative paths. The absolute-path versions in git will be replaced.
- **Roboflow API key exposure**: Current hardcoded key in `download_classification_data.py` (line 34) is already in git history. Mitigation: move to `.env`, rotate key after this change.

### Ready for Proposal

Yes. The exploration is complete: the full audit of hardcoded paths (generators + consumers), the complete pipeline map (34 scripts), the SOLID analysis with specific file:line references, the config landscape, and the consumer impact analysis are all documented. At least 2 approaches with tradeoffs are presented. The recommended approach (centralized config + path resolver) is scoped and actionable.
