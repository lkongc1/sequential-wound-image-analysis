# Path Resolver Specification

## Purpose

Centralized path resolution replacing 30+ duplicated `PROJECT_ROOT` definitions. Provides a single source of truth for converting between relative paths (stored in data files) and absolute paths (used at runtime), with automatic absolute-path detection for backward compatibility.

## Requirements

### Requirement: Project Root Constant

The system MUST expose a `PROJECT_ROOT` constant that resolves to the repository root directory at import time. All modules and scripts SHALL reference this constant instead of computing their own root.

#### Scenario: Root resolves to repo directory

- GIVEN the module is imported from anywhere within the project tree
- WHEN `PROJECT_ROOT` is evaluated
- THEN it returns an absolute `Path` pointing to the repository root

### Requirement: Absolute Path Resolution

The system MUST provide a `resolve(path) -> Path` function that converts a given path to an absolute path. If the input is already absolute, it SHALL be returned unchanged.

#### Scenario: Relative path becomes absolute

- GIVEN a relative path `"data/raw/images/img_001.jpg"`
- WHEN `resolve("data/raw/images/img_001.jpg")` is called
- THEN the result is `PROJECT_ROOT / "data/raw/images/img_001.jpg"` as an absolute `Path`

#### Scenario: Absolute path is returned as-is

- GIVEN an absolute path `"C:\\Users\\X\\project\\data\\img.jpg"` (or `/home/user/project/data/img.jpg`)
- WHEN `resolve(path)` is called
- THEN the same absolute `Path` is returned without modification

#### Scenario: Path with `.` and `..` segments

- GIVEN a relative path `"scripts/../data/file.csv"`
- WHEN `resolve(path)` is called
- THEN the result is a normalized absolute path with `..` resolved

### Requirement: Relative Path Generation

The system MUST provide a `relativize(path) -> str` function that computes a portable relative path from an absolute one, relative to `PROJECT_ROOT`.

#### Scenario: Absolute path converted to relative

- GIVEN an absolute path inside the project `PROJECT_ROOT / "data/raw/file.csv"`
- WHEN `relativize(path)` is called
- THEN the result is the POSIX-style relative string `"data/raw/file.csv"`

#### Scenario: Already-relative path returned as normalized string

- GIVEN a relative path `"data/processed/file.csv"`
- WHEN `relativize(path)` is called
- THEN it returns the normalized relative string

#### Scenario: Path outside project root

- GIVEN an absolute path outside `PROJECT_ROOT`
- WHEN `relativize(path)` is called
- THEN a `ValueError` SHALL be raised
