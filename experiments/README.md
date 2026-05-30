# Experiments — 2026-05-29 session (LB 12.624 → 11.921)

Standalone scripts (originated in `/tmp`, hardcoded absolute paths — run from repo root
in the `kaggle-arch` conda env). Provenance for the `ROGII_11.921` milestone.

- **konbu_prod.py** — productionized konbu recipe: rebuilds FormationPlaneKNN + full-density
  RowKNN imputers, parallel per-well feature build (caches to `data/processed/konbu/`),
  GPU-LightGBM ×3 seeds + GPU-XGBoost, Ridge stack. Banked **OOF 11.885**. Saves the 20
  models + `blend.json`/`feature_cols.json` (the latter two preserved in `docs/konbu_recipe/`).
  The Kaggle inference kernel is `jupyter_konbu/rogii_konbu_inference.py`.
- **phase4_reexam.py** — ablation that overturned the "GR conclusively dead" verdict: the
  GR-vs-typewell matching features contribute **+0.252 ft** as GBM inputs (we had tested them
  only as a point-estimator and deleted them).
- **ncc_feature_ablation.py** — re-audit probe: multi-scale NCC as GBM features on the konbu
  base (Phase-2 had killed NCC as a linear blend — the wrong frame).

Recipe summary, score history, and the corrected conclusions are in `plan.md`.
