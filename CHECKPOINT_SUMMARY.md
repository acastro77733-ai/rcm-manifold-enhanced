# Checkpoint Summary

Date: 2026-07-12

## Before
- The repository contained the unpacked RCM project with a basic ablation study and HRM-related modules.
- The experimental runner did not fully isolate conditions, did not clone identical checkpoints, and lacked strong paired controls or explicit HRM validation.

## After
- Added a regression test suite for experimental isolation.
- Refactored the enhanced ablation study to:
  - generate deterministic, pre-generated trial inputs,
  - clone identical initial checkpoints per condition,
  - isolate HRM behavior cleanly across full/no-HRM/field-control baselines,
  - evaluate next-state prediction, reconstruction, classification, recovery, and recall,
  - save paired-comparison and HRM-validation artifacts.
- Verified the new isolation logic with:
  - `pytest -q tests/test_experimental_isolation.py` → 4 passed
