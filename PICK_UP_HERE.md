# PICK UP HERE
_Last updated: 2026-05-28 ~22:45 (FIRST LB submission made — score PENDING; code-comp pipeline built; Phase 4 & postproc both dead; LB top moved to 7.5)_

## ⏯️ RESUME HERE FIRST — (1) get the LB score, (2) then pick the lever
**FIRST: read our first-ever LB score.**
```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-arch
kaggle competitions submissions -c rogii-wellbore-geology-prediction | head -4
```
A background poller was running (`log/` / task `b78jaf32m`) and may have logged it already. Submission
was made 2026-05-28 ~02:41 UTC from kernel **`stevewatson999/rogii-inference` v4** (our Phase-3 OOF
**12.76** model). **Record the score in `plan.md` LB Submission History + §1 table**, then decide.

**State of play (all settled this session):**
- **Phase 3 banked: OOF 12.7608.** Formation-KNN family ceiling (signal-limited, not capacity-limited).
- **Phase 4 (GR-matching: NCC/beam/PF/DTW) CONCLUSIVELY DEAD** — 3 probes. GR signal is real (oracle
  corr +0.72 at the true TVT) but **aliases**: fingerprint autocorrelation ~4 ft ≪ our anchor error
  ~12.7 ft, so any blind search locks the wrong fringe. **Do not build PF/beam/DTW.**
- **Phase 6 postproc DEAD too** — shrinkage/PS-fade/Sav-Golay on `oof_phase3` gave **+0.0025 ft**
  (honest nested CV); shrinkage optimum is α=1 (the GBM OOF is already calibrated).
- **The public LB top moved 9.25 → 7.5** since the plan's research (Deotte 8.6). Our 12.76 is far back.
  Two hypothesized levers (GR-matching, postproc) are both ~0, so the live frontier lever looks like a
  **much tighter spatial anchor** (we never pushed past konbu's ~12.1). A tighter anchor would *also*
  unlock GR refinement (the aliasing is anchor-quality-dependent).

**Decision after the score lands** (depends on the CV↔LB gap):
- If **LB ≈ OOF (~12–13)**: we're genuinely ~5 ft back → invest in a much better spatial anchor
  (kriging/GP instead of IDW-plane, more formations, better `b_well`) and/or **scout the 7.5 frontier**
  (recent public notebooks — the plan's research is 2 weeks stale).
- If **LB ≪ OOF**: distribution gap is favorable → richer formation-KNN (→ konbu 12.1) + LGB×seeds +
  CatBoost ensemble + hill-climb may already be competitive; build that, resubmit.

## ✅ Done so far
- **Phase 0 (2026-05-27).** `src/harness.py`: GroupKFold-by-well, simulate-test-on-train,
  null baseline = **15.9099 ft**. `python -m src.harness` reproduces it + confirms 14,151 test ids.
- **Phase 1 (2026-05-27).** Base pipeline: `src/features.py` 44 drift features; `src/train.py`
  LightGBM on drift, GKF-5; `src/predict.py` → submission. Phase-1 OOF = **14.98 ft**.
- **Phase 3 (2026-05-28) — PRODUCTIONIZED.** OOF **12.76 ft** (−2.19 vs Phase 1).
  - **`src/spatial.py`** (NEW): `FormationPlaneKNN` (distance-weighted 2-D plane through K=10 nearest
    **well centroids** per formation top, raw coords, LOO self-exclusion) + `RowKNN` (dense row-level
    ANCC, K=20 IDW, stride-3 ref ≈1.68M pts, LOO over-query buffer). Both built once from train wells.
  - **`src/features.py`** (refactored): two cached layers merged on `id` in `build_feature_matrix` —
    `build_base_features` → `features_base_{split}.parquet` (44 cols) and `build_spatial_features` →
    `spatial_{split}.parquet` (23 cols). **Train = LOO** (`self_wid=wid`); **test = full train-ref**
    (`self_wid=None`, mirrors how the hidden test is scored). `rk_tvt_formula` (`−Z+ANCC+b_well`,
    dense) is the #1 feature by gain by 4.5×.
  - **`src/train.py` + `src/predict.py`**: heavy config (160lv / lr .03 / 4000-ES200), `phase3`-tagged
    artifacts (`lgb_phase3_fold*.txt`, `oof_phase3.parquet`, `submission_phase3.csv`).
  - Matrix verified end-to-end: **67 feats, 0 rows missing spatial** on train (3.78M) + test (14,151).

## ▶️ Next steps (start here)
1. ✅ **Phase 3 banked** (OOF 12.7608), **Phase 4 ABANDONED** (GR aliases → ~0), **Phase 6 postproc
   DEAD** (+0.0025 ft). **First LB submission MADE — score pending** (see RESUME HERE FIRST).
2. **Read the LB score, record it, pick the lever** per the decision tree in RESUME HERE FIRST.
3. **Most likely real lever: a much tighter spatial anchor.** Current KNN ~12.1–12.76; LB frontier 7.5.
   Ideas: kriging / Gaussian-process on the 6 formation tops instead of IDW-plane; richer `b_well`
   (early/mid/late/WLS variants — see 9.251 `seg_b_well` in `/tmp/nihilisticneuralnet_*.code.py`);
   dense-ANCC std/dist/bias features. A tighter anchor lowers RMSE directly AND could unlock GR refine.
4. **Scout the 7.5 frontier** — the plan's research (9.25 top) is 2 weeks stale. Pull current public
   notebooks (`kaggle kernels list --competition rogii-wellbore-geology-prediction --sort-by scoreDescending`)
   to find the missing lever (better spatial method? leak? external data? typewell used differently?).
5. **Ensemble** (after a better anchor): LGB×3-seeds + CatBoost (⚠️ **xgboost/catboost NOT installed
   locally** — `pip install` in `kaggle-arch`, or run on deepthought), GKF-5, hill-climb/NNLS blend.

## 🚀 Kaggle submission pipeline (BUILT — code competition; reuse for every resubmit)
- **It IS a code competition** — CSV upload via CLI is rejected (HTTP 400). You submit by pushing a
  notebook that writes `/kaggle/working/submission.csv`, then **Submit to Competition on the web** (the
  final submit is web-only; CLI can't do it).
- Artifacts dataset (private): **`stevewatson999/rogii-artifacts`** = `src/` (zipped) + 5 gz fold models.
  Rebuild from `kaggle_artifacts/` then `kaggle datasets version -p kaggle_artifacts -m "..."`.
- Kernel: **`jupyter/rogii_inference.py`** + `jupyter/kernel-metadata.json`. Reuses `src/` via shims
  (`C.RAW`→comp input, `C.MODELS`→decompressed models, `C.PROC`→working) and **self-locates** inputs.
  Resubmit: `kaggle kernels push -p jupyter/` → wait for COMPLETE (`kaggle kernels status ...`) → submit on web.
- **Mount gotchas (cost 4 kernel versions):** CLI-attached sources mount **NESTED** at
  `/kaggle/input/competitions/<slug>/` and `/kaggle/input/datasets/<user>/<slug>/` (NOT flat — the kernel
  recurses to find them); Kaggle **auto-extracts** uploads (`src.zip`→`src/src/...`, `*.gz`→`*.txt`);
  brand-new private datasets take a few min to propagate before they attach.
- Validate locally before pushing: `/tmp/test_kernel.py` mirrors the kernel against Kaggle's layout.

## ⚠️ Productionization gotchas (preserved in src/spatial.py — don't regress)
- **RowKNN LOO buffer:** a well's own rows are its nearest neighbors, so the KD-tree query over-fetches
  `max(400, maxself + k + 5)` (maxself ≈ 4047 at stride-3) before masking self, else you leak / run
  short of neighbors.
- **Plane fit is in RAW (X,Y) through well CENTROIDS** — never dense per-row points (collinear along
  one trajectory → intercept extrapolates to garbage, saw 2107 ft).
- **Test uses the full train reference (no self-exclusion).** For the 3 visible test wells (blanked
  train wells) this is a mild optimism, consistent with predict.py's already-flagged optimistic sanity
  check; for the real hidden test there is no leak.
- **Cache layout:** `features_base_{split}.parquet` (base 44) + `spatial_{split}.parquet` (spatial 23).
  `build_feature_matrix` merges them on `id` each call (cheap; not re-cached). Delete a layer to force
  its rebuild. `spatial_train.parquet` was seeded by copying the validated `phase3_feats_s3.parquet`.

## ⚠️ Compute / environment gotchas
- **Conda env (skynet, aarch64): `kaggle-arch`** (`source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate kaggle-arch`). base/`kaggle` CLI is broken. **xgboost not installed** locally.
- **Neither skynet nor deepthought has a GPU-enabled LightGBM build** (both CPU-only). For this GBM
  workload skynet CPU is the right lane. ROGII path does NOT exist on deepthought (rsync to dispatch).
- Run scripts with `nohup` + timestamped `log/`. Never `conda run` (buffers logs).

## Verification items (plan §10)
- ✅ **Code competition** (not CSV upload) — confirmed (CLI 400; pipeline built, see above).
- ✅ **Metric scale** — LB scores are RMSE-scale (top 7.5, null ~15.9); MSE label ranks identically.
- ⬜ **Exact daily submission quota** — likely 5/day (typical); confirm on the competition Submissions tab.
- ⬜ **Scoring-kernel runtime cap** — our kernel does 3 wells in ~3–4 min; ~200 hidden wells should fit
  a typical ≤9h cap (an over-run would *error*, not silently fail). Confirm if a future kernel gets heavy.
- ℹ️ Deadline **2026-08-05 23:59 UTC**.

## Read order for a cold start
`PICK_UP_HERE.md` (this) → `SUMMARY.md` (facts) → `plan_summary.md` (the approach) → `plan.md` (full detail).
