# Config Manager Specification

## Purpose

Unified configuration system merging `.env` environment variables with `config/*.yaml` files, supporting environment-specific overrides. Replaces the current disconnected state where `.env` and `config/*.yaml` are never imported together.

## Requirements

### Requirement: Multi-Source Config Loading

The system MUST provide a `ConfigManager` class that loads configuration from three sources with defined precedence: `.env` variables override environment-specific YAML, which overrides base YAML.

#### Scenario: All sources present

- GIVEN `config/defaults.yaml` defines `paths.data_dir: "data"`, `config/environments/development.yaml` defines `paths.data_dir: "data_dev"`, and `.env` defines `PATHS__DATA_DIR=/custom/path`
- WHEN `ConfigManager()` is instantiated
- THEN `cm.get("paths.data_dir")` returns `"/custom/path"` (`.env` wins)

#### Scenario: Only base YAML present

- GIVEN `config/defaults.yaml` exists but no environment YAML and no `.env`
- WHEN `ConfigManager()` is instantiated
- THEN values from `config/defaults.yaml` are returned for all keys

### Requirement: Graceful Handling of Missing Files

The system SHALL NOT raise errors when optional config files are absent. Missing `.env` or missing environment-specific YAML files MUST be treated as empty sources.

#### Scenario: Missing environment-specific YAML

- GIVEN `config/environments/production.yaml` does not exist
- WHEN `ConfigManager(env="production")` is instantiated
- THEN base YAML values are used without error

#### Scenario: Missing .env file

- GIVEN no `.env` file exists in the project root
- WHEN `ConfigManager()` is instantiated
- THEN YAML values are returned without error; no `.env` overrides apply

### Requirement: Typed Getter Methods

`ConfigManager` MUST expose typed getter methods (`get_str`, `get_int`, `get_float`, `get_bool`, `get_path`) that return configuration values in the expected type.

#### Scenario: Typed getter returns correct type

- GIVEN `config/defaults.yaml` defines `training.batch_size: 32`
- WHEN `cm.get_int("training.batch_size")` is called
- THEN the integer `32` is returned

#### Scenario: Type mismatch raises error

- GIVEN a key whose value is the string `"not_a_number"`
- WHEN `cm.get_int("key")` is called
- THEN a `ValueError` or `TypeError` SHALL be raised

### Requirement: Nested Key Access

The system SHALL support dot-notation nested key access for YAML hierarchies (e.g., `"paths.data_dir"` resolves `paths: { data_dir: ... }`).

#### Scenario: Nested key from YAML

- GIVEN `config/defaults.yaml` contains `paths: { data_dir: "data/raw" }`
- WHEN `cm.get_str("paths.data_dir")` is called
- THEN `"data/raw"` is returned
