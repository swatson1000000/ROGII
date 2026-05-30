---
description: "Use when writing or editing the Kaggle submission notebook or submission pipeline."
applyTo: "src/build_inference_notebook.py, jupyter/**"
---

# Inference & Submission Guidelines

## Submission format
- `submission.csv` with columns `id, tvt`.
- One row per scored horizontal-well row, i.e. rows `[PS, end]` of each test well.
- `id = f"{well_id}_{row_index}"` (0-based row index into that well's horizontal file).
- The id set must match `data/raw/sample_submission.csv` exactly — validate before writing.

## Pipeline
- Load trained artifacts from a Kaggle dataset (no internet at submit time).
- Use the **same feature-builder** as training. Anchor predictions on `TVT_input` at PS.
- Confirm Kaggle's runtime/internet constraints on the competition page before finalising
  (record them in `plan.md` once verified — currently unverified).

## Notebook hygiene
- Notebooks live in `jupyter/` and are the only place notebooks are used.
- Keep the notebook a thin wrapper over `src/` logic; don't fork the feature code into the
  notebook.
