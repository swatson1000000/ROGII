---
description: "Use when writing or editing training scripts. Covers CV, logging convention, OOF, and the geosteering anchor."
applyTo: "src/train*.py"
---

# Training Guidelines

## Objective
Minimise MSE/RMSE of `dTVT = TVT_true − TVT_pred` over scored rows (rows ≥ PS). Anchor each
prediction on the known TVT at PS — predict the **deviation** along the lateral, not absolute
TVT from scratch, so errors stay bounded near the landing point.

## Cross-validation
- **GroupKFold by `well_id`** (5 folds, seed 42). Write per-fold OOF predictions to
  `data/processed/oof_*.csv` (columns `id, tvt`) for honest local scoring via `src/evaluate.py`.
- Local metric MUST mirror the LB: score only rows ≥ PS.

## Epoch / iteration logging (required when training is iterative)
Emit a summary line per epoch/round, implemented via a callback:

```
========================================
Epoch N/E: train_loss=X.XXXX val_rmse=X.XXXX time=Xm XXs  YYYY-MM-DD HH:MM:SS ★ BEST
========================================
```
- Time as `Xm XXs` (use `divmod(elapsed, 60)`), include `time.strftime('%Y-%m-%d %H:%M:%S')`.
- Append ` ★ BEST` on a new best validation RMSE.

## Guardrails
- Set a fold-0 validation gate before committing to a full 5-fold run (record it in `plan.md`).
- Never feature on train-only columns (`TVT`, markers, `Geology`) at inference — they don't
  exist in test. Keep a single feature-builder shared by train and inference paths.
