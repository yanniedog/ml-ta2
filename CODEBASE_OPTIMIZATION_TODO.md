# Codebase Optimization and Refactoring Plan

This document outlines a TODO list for reducing waste, refactoring the codebase, and improving adherence to software engineering best practices such as DRY (Don't Repeat Yourself).

## 1. Waste Reduction: Consolidate Redundant Test Scripts

The project's root directory contains a large number of test-related scripts. Many of these appear to be temporary, redundant, or superseded by more comprehensive tests. This creates clutter and makes it difficult to maintain a clear and reliable testing strategy.

**TODO:**

- [ ] **Review and remove `simple_test.py`:** This script re-implements core logic from the `src` directory for testing purposes. The functionality is better covered by `comprehensive_test.py`, which tests the `src` modules directly. Deleting this file will eliminate redundant code and reduce maintenance overhead.

- [ ] **Consolidate `phaseX_quality_gate.py` and `direct_phaseX_test.py` scripts:** These scripts were likely used for manual, phase-specific testing. Their logic should be migrated into the main test suite under the `tests/` directory using a testing framework like `pytest`. Once migrated, these files should be deleted.

- [ ] **Investigate and remove one-off scripts:** Files like `quick_test.py`, `fix_ab_testing.py`, and `fix_ab_testing2.py` appear to be temporary or for debugging. Their purpose should be understood, and if they are no longer needed, they should be removed.

- [ ] **Standardize on a single test runner:** All testing efforts should be unified under a single framework like `pytest`. Test files should be moved to the `tests/` directory and follow a consistent naming convention (e.g., `test_*.py`).

## 2. Refactoring for DRY Principle

The codebase contains instances of duplicated logic, particularly in test scripts. This violates the DRY (Don't Repeat Yourself) principle, which leads to increased maintenance overhead and potential for inconsistencies.

**TODO:**

- [ ] **Refactor `simple_test.py` to use `src.indicators`:** The `simple_test.py` script contains local implementations of `calculate_rsi` and `calculate_bollinger_bands`. These functions are already implemented more robustly in `src/indicators.py`. The test should be refactored to import and use the functions from the `src` module, and the duplicated local implementations should be deleted.

- [ ] **Audit other test scripts for duplicated logic:** Perform a broader review of all test scripts (including those in the `tests/` directory) to identify other instances of duplicated helper functions or setup code. Consolidate these into shared test utilities.

## 3. Codebase and Project Structure Improvements

The project's structure can be improved by organizing files more logically and reducing clutter in the root directory.

**TODO:**

- [ ] **Relocate top-level scripts:** Move scripts from the root directory into more appropriate subdirectories to create a cleaner, more organized project structure:
  - Move `demo.py` and `feature_engineering_demo.py` to `notebooks/` or a new `examples/` directory.
  - Move `launch.bat` and `launch_local.py` to the `scripts/` directory.
  - Move `web_frontend.py` into the `web_app/` directory.

- [ ] **Review `src/` module granularity:** Some modules in the `src/` directory are very large (e.g., `model_serving.py`, `security_audit.py`). Consider breaking these down into smaller, more focused modules to improve readability and maintainability.

- [ ] **Standardize configuration management:** Ensure the configuration approach is consistent across all `*.yaml` files in the `config/` directory. Document the purpose of each configuration file (`development.yaml`, `production.yaml`, etc.) to avoid ambiguity.
