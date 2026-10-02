# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.3] - 2026-10-02

### Added

- `spxmod` exports `XModel`, `Space`, `VariableBuilder`, the dimension classes
  and `__version__` at the package level (#29).
- GitHub Actions workflow running ruff and pytest on Python 3.11 and 3.12
  (#29).
- README with install instructions, a runnable example and the configuration
  schema (#29).

### Changed

- Package metadata moved from `setup.py` to `pyproject.toml`. `requires-python`
  is `>=3.11`. Dependency floors are `regmod>=0.1.2,<0.2` and `msca>=0.2.0`,
  the first versions that provide what spxmod imports (#26, #29).
- Direct dependencies `numpy`, `scipy`, `pandas` and `xspline` are declared
  explicitly (#26).

### Removed

- The Cython extension `spxmod.linalg`. spxmod is now a pure-Python package
  with no compiler needed at install time (#26).
- `.pre-commit-config.yaml`, in favor of CI (#27).

### Fixed

- Spline variables on a multi-cell space had their design-matrix columns in a
  different order than the variables and smoothing priors, so priors coupled
  the wrong coefficient pairs. Results for spline-on-space models change (#25).
- `predict(..., return_ui=True)` always raised. The prediction-interval path
  referenced a missing attribute and relied on a Cython helper that did not
  accept sparse input and used a NumPy API removed in 2.0 (#26).
- `Dimension.set_span` failed on column names that are not valid Python
  identifiers (#28).
- scipy deprecation and future warnings from `block_diag` and `diags` (#27).

[Unreleased]: https://github.com/ihmeuw-msca/spxmod/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/ihmeuw-msca/spxmod/compare/v0.3.2...v0.3.3
