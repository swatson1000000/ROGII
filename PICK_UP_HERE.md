# PICK UP HERE
_Last updated: 2026-05-29 ~21:30 — **LB 11.921 banked + pushed to GitHub (tag `ROGII_11.921`)**. konbu
recipe productionized & submitted (OOF 11.885). GPU LightGBM built. Two "dead" verdicts re-audited
(GR overturned, NCC confirmed dead). Backups fixed (side-quest). All clean; nothing running._

## ⏯️ RESUME HERE — pick the next lever (the big question)
We went **12.624 → 11.921** (−0.70 ft) today by reproducing+banking konbu's recipe. Frontier is **≈7.5**,
so we're **~4.4 ft back**. Today proved part of our gap was self-inflicted (we'd thrown away the GR
features and wrongly declared a 12.1 ceiling). But the re-audit yields are now small scraps (~0.25 ft
GR, already banked; NCC was a null). **The ~4.4 ft frontier gap is still unexplained.**

**Decision to make first thing:**
- **(A) Test a kriging / Gaussian-Process spatial anchor.** This is the ONE hypothesis big enough to
  explain a 4-ft gap and the only thing we've literally never tried. Our `FormationPlaneKNN` is
  line-for-line identical to konbu's, so the anchor has never been pushed past plane-KNN. Competitors
  publish `rogii-*-gp` notebooks (e.g. `innerf1re/rogii-ultranote-v6-gp-main`, pulled to
  `/tmp/pull_innerf1re_rogii-ultranote-v6-gp-main/`) — GP is plausibly the lever the 7.5 leaders pull.
  **Recommended.** Cheapest test: GP-impute ANCC at eval (X,Y) vs the plane-KNN, compare per-formation
  imputation RMSE + ΔOOF on the cached konbu matrix.
- **(B) Finish the re-audit (lower EV):** postproc probe #2 on the **konbu** OOF (we only ever tested
  postproc on the inferior 12.76 OOF; konbu's own pipeline used Optuna shrinkage + PS-fade + Sav-Golay).
  Artifacts ready: `/tmp/konbu_oof.csv` (275 MB, the 11.87 OOF). Then normalized-DTW-as-feature / seq
  blend (both LOW).
- **(C) Cheap ensemble gains:** more LGB seeds + CatBoost (not installed — `pip install catboost`),
  more features, on the cached `data/processed/konbu/train_feats.parquet` (no rebuild needed).

The honest read (from §`Currently working on` in plan.md): re-auditing nets ~0.25-ft scraps; a better
anchor is the only candidate matching a 4-ft gap. **Lean (A).**

## ✅ Done this session (2026-05-29)
- **LB 11.921** (public), OOF 11.885, gap +0.036 → CV trustworthy. Prior 12.624. Recorded in
  `plan.md` §1 + LB Submission History + Experiment Log.
- **Diagnosed the konbu gap** (pulled his actual notebook): our anchor ≡ his; deficit = deleted GR/beam
  feats + full-density RowKNN (5.05M vs our stride-3 1.68M) + tighter reg (89lv+heavy L2) + LGB×3+XGB
  Ridge stack. **Overturned the banked "signal-limited at 12.11" conclusion.**
- **Productionized:** `experiments/konbu_prod.py` (parallel feature build → `data/processed/konbu/`,
  GPU-LGB×3 + GPU-XGB, saves 20 models → `models/konbu/`). Inference kernel
  `jupyter_konbu/rogii_konbu_inference.py` (validated locally to 7.6e-6 ft, ran COMPLETE on Kaggle).
- **GPU LightGBM built** on the GB10 (CUDA/sm_121 from source; 6.73× faster, identical RMSE). xgboost
  now installed in `kaggle-arch` (GPU works). Recipe in memory `lgb-cuda-build-skynet`.
- **Re-audit of "dead" verdicts** (`experiments/phase4_reexam.py`, `experiments/ncc_feature_ablation.py`):
  Phase-4 GR "dead" → **+0.25 ft** (overturned, in recipe); multi-scale NCC "dead" → **+0.004 ft**
  (confirmed dead — redundant with konbu's matching feats). Lesson: re-test in the GBM-feature frame;
  it can confirm *or* overturn.
- **Pushed to GitHub** `swatson1000000/ROGII`, commit `052c296`, tag `ROGII_11.921` (40 files; data +
  model weights gitignored).
- **Backups fixed** (unrelated): nightly was silently failing 5 days (orphaned UID-1001 ownership on
  deepthought `/mnt/mypassport/backup_*_skynet` — reclaimed via chown). `~/bin/backup` hardened
  (`send_email` retry + 465 fallback + `last_email_fail` sentinel). ⚠️ The 6 AM email failure root cause
  is unproven (works interactively); tomorrow's 6 AM run is the real test — check `~/.msmtp.log` +
  the sentinel file.

## 📁 Key artifacts (for the next lever)
- Cached konbu feature matrices: `data/processed/konbu/{train_feats,test_feats}.parquet` (78 feats +
  id/well/target). **Use these for any ablation — no 35-min rebuild needed.**
- konbu OOF (for postproc probe): `/tmp/konbu_oof.csv`.
- Models: `models/konbu/` (3 LGB seeds ×5 folds + XGB ×5 folds + `blend.json` + `feature_cols.json`).
- Recipe provenance in repo: `docs/konbu_recipe/{blend,feature_cols}.json`.
- GPU LGB: `device_type="cuda"` works in `kaggle-arch`; **fresh `lgb.Dataset` per device** (reusing one
  across cpu→cuda segfaults — see memory `lgb-cuda-build-skynet`).

## ⚠️ Settled this session — don't re-litigate
- konbu recipe = current best (OOF 11.885 / LB 11.921). Anchor is plane-KNN (≡ konbu).
- GR-matching: dead as a *point estimator* (aliasing), but the *features* are worth +0.25 ft — KEPT.
- Multi-scale NCC: adds ~0 on top of konbu's base (redundant). Don't re-add.
- Genuinely dead (don't revisit): absolute-TVT target (19.5); GR/NCC/DTW as point estimators.

## Read order for a cold start
`PICK_UP_HERE.md` (this) → `plan.md` §1 + Experiment Log (newest first) → `SUMMARY.md` (facts) →
`plan_summary.md` (approach). GitHub: `swatson1000000/ROGII` @ tag `ROGII_11.921`.
