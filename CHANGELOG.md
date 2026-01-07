# Changelog

All notable changes to the RDT Kernel project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-01-07

### Added
- Comprehensive test suite with 20+ tests covering:
  - Basic functionality
  - Device detection (CPU/GPU/TPU)
  - Numerical stability
  - Parameter validation
  - Edge cases
  - Periodic boundary conditions
- Complete docstrings for all functions
- Benchmarks and Stability section in README
- Variance plotting script (`examples/rdt_variance_plot.py`)
- Demo Jupyter notebook (`examples/demo.ipynb`)
- `.gitignore` file for Python projects
- `CHANGELOG.md` to track version history
- Development dependencies (pytest, matplotlib)

### Changed
- Exposed `clamp_min` parameter in `step()` function (default: 1.001)
- Updated README to document:
  - Periodic boundary conditions
  - Recommended parameter ranges
  - Performance benchmarks
  - Clamp behavior
- Bumped version from 1.0.6 to 1.1.0

### Fixed
- License inconsistency (standardized to Apache 2.0 across all files)
- Empty test file now contains comprehensive tests
- Empty demo notebook now contains working examples

## [1.0.6] - 2025

### Changed
- Force sync PyPI release
- Updated README

## Earlier Versions

See git history for details on versions prior to 1.0.6.
