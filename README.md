# ROGII — Wellbore Geology Prediction (Kaggle)

Predict horizontal-well **TVT** past the prediction-start point. Metric: RMSE (ft); null
(repeat last-known TVT) ≈ 15.91. Code competition (notebook submission).

## Current best — LB **11.921** (`ROGII_11.921`)
konbu-style recipe: exact within-well identity `TVT = −Z + ANCC + b_well`, with ANCC
spatially imputed (FormationPlaneKNN through nearest well centroids + full-density RowKNN),
~78 features (incl. GR rolling/lag, GR-vs-typewell offset-diffs + beam, formation geometry),
**LightGBM ×3 seeds + XGBoost → Ridge stack** (OOF 11.885). Trained locally
(`experiments/konbu_prod.py`), served by `jupyter_konbu/rogii_konbu_inference.py`.

## Layout
- `src/` — base pipeline (harness, features, spatial imputers, train/predict)
- `experiments/` — session scripts (training driver + re-audit probes)
- `jupyter/`, `jupyter_konbu/` — Kaggle inference kernels
- `plan.md` — source of truth: score history, experiment log, prioritized plan
- `docs/konbu_recipe/` — trained blend weights + feature list (provenance)

Data, model weights, and Kaggle build staging are git-ignored (reproducible / in the Kaggle dataset).
