---
description: "Use when writing or editing data loading, feature engineering, GR/type-well correlation, or cross-validation splits."
applyTo: "src/dataset.py, src/utils.py, src/eda.py"
---

# Dataset & Feature Guidelines

## Files (per well, in data/raw/{train,test}/)

| File | Columns | Notes |
|------|---------|-------|
| `<id>__horizontal_well.csv` | `MD, X, Y, Z, GR, TVT_input` (+`TVT` + markers in train) | the drilled lateral, row-ordered by MD (~1 ft step) |
| `<id>__typewell.csv` | `TVT, GR` (+`Geology` in train) | vertical reference well; one per horizontal |

- `TVT` — **target** (train only), fully filled.
- `TVT_input` — known TVT up to the **Prediction Start (PS)** point; equals `TVT` there, blank after.
- Markers `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA` — formation-top depths, **train only**.
- `Geology` — formation code per type-well depth, **train only**.

## Prediction Start (PS)
`ps = first row index where TVT_input is NaN` (see `dataset.prediction_start_index`). Rows
`[ps, end]` are the submission rows; their id is `f"{well_id}_{i}"`.

## Cross-validation
- **Split by well id (GroupKFold on `well_id`)** — never split rows of a well across folds.
  Geology is spatially autocorrelated; row-level splits leak the answer.
- Consider grouping by spatial proximity / type-well so neighbouring wells don't straddle folds.
- `N_FOLDS = 5`, `SEED = 42` (see `config.py`).

## Data hygiene
- **`GR` may be NaN** — impute or mask explicitly; don't let NaNs propagate into features.
- Coordinates `X, Y, Z` are absolute survey coordinates; derive trajectory features (azimuth,
  inclination, dip) as deltas, not raw values.
- Validate that reconstructed submission ids match `data/raw/sample_submission.csv` exactly.
