# ROGII — Wellbore Geology Prediction — Competition Plan

**Competition**: [ROGII — Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
**Host**: ROGII | **Category**: Featured | **Prize**: $50,000 | **Max team**: 5
**Opened**: 2026-05-05 | **Deadline**: **2026-08-05 23:59 UTC**
**Metric**: RMSE of `dTVT = TVT_true − TVT_pred` over hidden rows (Kaggle config reports "MSE"; same ranking — see `SUMMARY.md`)
**Target column**: `tvt` (True Vertical Thickness, feet)

> This plan is grounded in: the official task deck, the `KaggleBookSummary.md` winning workflow,
> EDA on the local data (269 wells with targets), and a deliberate study of 10 public notebooks
> spanning the leaderboard (Deotte starters, Giba/titericz baseline, the 9.251 DWT ensemble,
> Mitch's 9.398 drift-NCC writeup, the plane-fit-KNN and beam/PF notebooks). See §3.

---

## 0. TL;DR strategy

This is **not** a tabular per-row problem and **not** a sequence-model problem. It is a
**signal-alignment + spatial-imputation** problem stacked by a GBM. The proven recipe (multiple
top public notebooks converge on it):

1. **Predict drift, not absolute TVT.** Target = `TVT − last_known_TVT`. (Biggest single lever.)
2. **Align horizontal GR to the typewell GR "barcode"** via multi-scale normalized
   cross-correlation (NCC). This is the core signal (r ≈ 0.999 with true TVT).
3. **Exploit offset wells geometrically.** `TVT ≈ −Z + ANCC + b_well` is *exact* (r = 1.0000,
   resid 0.007 ft); impute the train-only formation top ANCC at test (X,Y) via plane-fit KNN.
4. **Add orthogonal sequence estimators** (particle filter, beam/Viterbi over typewell GR).
5. **Stack everything with a GBM ensemble** (LightGBM + XGBoost + CatBoost), GroupKFold-by-well.
6. **Blend (NNLS / hill-climb) + post-process** (Optuna shrinkage/fade + Savitzky-Golay).

Score ladder this produces: **null 15.9 → drift+NCC+KNN ≈ 12 → +beam/PF ≈ 10 → full ensemble + postproc ≈ 9.25–9.4 RMSE.**

---

## 1. Current State

| Item | Value |
|------|-------|
| Best Kaggle LB score | **11.921** (public) — 2026-05-30, kernel `stevewatson999/rogii-konbu-inference` v1 = konbu recipe (OOF 11.885; LB−OOF=+0.036, CV held). Prior: 12.624 (Phase-3). **−0.70 ft LB gain.** Frontier ≈ 7.5 → ~4.4 ft back. CV↔LB gap small & favorable → trust CV. |
| Best local CV (OOF) | **11.87 ft** (2026-05-29, konbu recipe reproduced on our data: 78 feats incl. GR/beam, full-density RowKNN, LGB×3+XGB Ridge stack) — NOT yet submitted/productionized. Prior banked: 12.76 (Phase 3, our single-LGB 67-feat). Phase 1 was 14.98. |
| Null baseline (local, **all 773 wells**) | **15.91 ft** RMSE (predict last_known_TVT) — matches published 15.91 |
| Public LB landscape | top ≈ **9.25** (DWT ensemble); strong solutions 9.4–9.96; null ≈ 15.9 |
| Currently running | nothing (LB score 12.624 received 2026-05-29; poller done). |
| Primary approach | drift target + formation KNN (plane + row) + GBM stack (NCC dead, see §5); Phase 4 PF/beam next |
| Target score | **≤ 9.4** (proven reachable), stretch **≤ 9.25** |
| Currently working on | **DONE FOR THE NIGHT 2026-05-29. LB 11.921 banked + pushed to GitHub (tag `ROGII_11.921`).** konbu recipe (OOF 11.885) submitted; kernel `rogii-konbu-inference` v1, dataset `rogii-konbu-artifacts`, models+cached feats under `models/konbu/`+`data/processed/konbu/`. Re-audit of "dead" verdicts ran 2 probes: **GR overturned (+0.25 ft, now in recipe); multi-scale NCC CONFIRMED dead (+0.004, redundant w/ konbu's matching feats).** **Next session — pick the lever (~4.4 ft to frontier 7.5):** (a) **kriging/GP spatial anchor** — the only untested hypothesis big enough to explain a 4-ft gap (competitors' `rogii-*-gp` notebooks; our KNN ≡ konbu's so it's never been beaten); (b) postproc probe #2 on the konbu OOF (LOW-MED, `/tmp/konbu_oof.csv` exists); (c) more feats/seeds/CatBoost. See PICK_UP_HERE.md. |

### LB Submission History
| Date | Approach | CV (OOF) | LB | Notes |
|------|----------|----------|----|-------|
| 2026-05-28 | Phase 3: 44 base + 23 formation plane/row-KNN, heavy LightGBM 160lv (kernel v4) | 12.7608 | **12.624** | First-ever submission. LB slightly **better** than OOF → gap small & favorable; trust CV. Frontier ≈7.5 → ~5 ft back. |
| 2026-05-30 | konbu recipe: 78 feats (incl. GR/beam), full-density RowKNN, LGB×3+XGB Ridge stack (kernel `rogii-konbu-inference` v1) | 11.885 | **11.921** | −0.70 ft vs prior. LB−OOF=+0.036 (CV held). Overturned the "signal-limited at 12.11" claim. ~4.4 ft to frontier. |

---

## 2. Problem restatement & the equations that matter

Each horizontal well is a sequence along measured depth `MD` (~1 ft steps). `TVT_input` is the
geologist's known TVT up to the **Prediction Start (PS)** point, then NaN. We predict `TVT` for
rows ≥ PS. A paired **typewell** (vertical reference) gives an unambiguous `GR`-vs-`TVT` profile.

Two facts dominate the solution (both EDA-verified locally):

- **Geometric identity (exact):** within a well, `TVT = −Z + ANCC + b_well` with Pearson
  r = 1.00000 and residual std ≈ 0.0065 ft. ANCC is a formation-top surface. It is **train-only**,
  so the lever is *spatial imputation* of ANCC at the eval (X,Y) from neighbouring wells.
- **GR barcode:** the GR trace is a stratigraphic fingerprint. Sliding the horizontal GR against
  the typewell GR locates TVT. Multi-scale NCC reaches r ≈ 0.999 with true TVT.

EDA numbers (269 local wells): null RMSE 16.18 ft; drift mean +2.1, std 16.0, range [−70, +99],
abs-drift p90 = 25.6 ft; eval-zone ≈ 4800 rows/well (median); **GR is 28% NaN on average**
(max 73%) — alignment must handle missing GR (interpolate + carry a missing flag).

---

## 3. Research findings (leaderboard landscape)

### Score ladder (from the notebooks studied)
| Approach | OOF RMSE | LB | Source |
|----------|----------|----|--------|
| Predict `last_known_TVT` (null) | 15.91 | — | titericz/Giba "Last Value Baseline" |
| Tree, 52 feats, **absolute TVT** target | 19.50 | — | Mitch writeup R2 (worse than null!) |
| **Drift target** + GR xcorr + formation KNN | 14.99 | — | Mitch R3 |
| + Viterbi beam / particle filter | 13.96 | — | Mitch R4 |
| + GBM ensemble (LGB+XGB+CatBoost) | 13.90 | — | Mitch R5 |
| Deotte XGB starter (drift + ~50 feats) | ~15 | — | cdeotte "CV 15" |
| Plane-fit formation-top KNN + LGB/XGB | 12.11 | 11.91 | konbu17 |
| 163 feats: multi-scale NCC + plane-KNN + beam/PF + Optuna + NNLS | 10.01 | 9.41 | Mitch R6 |
| Retuned CatBoost, NNLS blend | 10.05 | **9.398** | Mitch R7 |
| Full PF + beam×7 + DTW(multiscale+stochastic) + NCC + KNN → LGB+CB → hill-climb → Optuna + SG | — | **9.251** | nihilisticneuralnet (469 votes) |

CV↔LB gap is small and favourable (OOF ~10.0 → LB ~9.4); **trust CV** (Ch 7).

### The signals (orthogonal building blocks)
- **Multi-scale NCC** (Pearson r windows, half-widths 8/15/25, softmax-blended): the single most
  accurate signal. Amplitude-invariant — robust to GR scale differences between wells.
- **Formation plane-fit KNN**: for each of 6 formation tops (ANCC, ASTNU, ASTNL, EGFDU, EGFDL,
  BUDA), fit a weighted 2D plane through the K≈15 nearest training wells' tops, evaluate at eval
  (X,Y); calibrate per-well bias `b_well` from the anchor zone. Imputation RMSE ~17 ft (vs ~47 ft
  for naive IDW). Plus a row-level dense ANCC KNN over all anchor rows for finer resolution.
- **Particle filters** (≈500–600 particles): sequential Bayesian tracking of TVT down the well;
  GR-likelihood emission against the typewell grid + a motion model. Two flavours: ANCC-anchored
  and Z-velocity-regressed. PF(ANCC) is a *standalone* improver (13.4 ft).
- **Beam / Viterbi search** over the typewell GR (several emission/transition stiffness configs):
  large standalone error but orthogonal directional signal the GBM exploits.
- **Anchor-zone priors**: TVT rate over last ~200 known rows; linear/Z extrapolation;
  typewell-fit quality (affine calibration of GR vs typewell GR on the known prefix).
- **"Target-distance" features**: for hypothesised TVT anchors ± offsets, `GR − typewell_GR(anchor+offset)`.
  Lets the GBM judge which alignment is self-consistent.

### What does NOT work (verified by practitioners — avoid)
- **Absolute-TVT target**: 19.5 ft, *worse than null*. The per-well offset (11,000–12,000 ft)
  swamps the ±16 ft signal. → Always train on drift.
- **Sequence models (BiLSTM / TCN / 1D-CNN)**: ~14.6 ft OOF; Deotte's NN starter ~15.5. They
  memorise per-well GR fingerprints; **773 wells is too few to generalise**. This overturns the
  naive "it's sequential so use an RNN" intuition. Do not invest here early (maybe a late, heavily
  regularised ensemble member only).
- **Naive DTW (L2)** as the *only* alignment: can *regress* OOF (10.008 → 10.07 for one team)
  because L2 is amplitude-sensitive while NCC is amplitude-invariant. (The 9.251 notebook *did*
  use DTW successfully, but normalised and as one of many signals — treat DTW as optional stacking
  diversity, not a core signal.)
- **Elaborate multi-level stacking**: base models are 0.98–0.99 correlated → stacking overfits OOF
  noise. Prefer NNLS / hill-climbing blends.

---

## 4. Validation strategy (Kaggle Book Ch 7)

- **GroupKFold(5) by `well_id`** everywhere. Non-negotiable: each well has a distinctive GR
  fingerprint; random row KFold leaks it and inflates CV. (Confirmed by every serious notebook.)
- **Local metric must mirror the LB exactly**: RMSE of `TVT_true − pred` over **eval rows only**
  (rows ≥ PS). Build it in `src/evaluate.py` (already scaffolded) and score every experiment.
- **Simulate test on training wells**: keep the known prefix, hide the eval zone, predict, score
  against the held `TVT`. This is how all OOF numbers above were produced.
- **Adversarial validation** (Ch 7): train a classifier to separate train vs test wells on
  spatial/log summary features. The hidden test is ~200 wells — check whether they sit in the same
  (X,Y) region / GR regime as train; if not, weight CV toward test-like wells.
- **Trust CV over public LB.** Observed OOF↔LB gap is ~0.6 ft and stable. Select final submissions
  on CV (Ch 7 submission-selection).

---

## 5. The build plan (phased & gated)

Each phase has a **gate** (a CV bar to clear before moving on) and an **expected** score from the
research. Code goes in `src/`; runs are logged per CLAUDE.md (nohup + timestamped logs). Feature
generation is CPU/numba-heavy (~hours over 773 wells) — cache feature matrices to
`data/processed/`. Train GBMs on **deepthought** (GPU); run numba feature gen on **skynet** (CPU).

- ✅ **Phase 0 — Harness & null baseline.** *(done 2026-05-27)*
  Data pull complete (773 train + 3 test + 773 PNGs). `src/evaluate.py` scoring +
  `src/harness.py` (GroupKFold-by-well split, simulate-test-on-train loop, null baseline).
  **Null RMSE = 15.9099 ft** over all 773 wells (matches published 15.91). Test eval ids match
  `sample_submission.csv` exactly (14151). Folds balanced (155/155/155/154/154). Reproduce with
  `python -m src.harness`. *Gate PASSED.*

- ✅ **Phase 1 — Drift target + simple features + GBM.** *(done 2026-05-27)*
  `src/features.py` (44 feats: anchor/prefix slopes, position-from-PS, GR rolling/diff/lag +
  missing flag, typewell-GR-at-anchor) + `src/train.py` (LightGBM on drift, GroupKFold-5,
  lean params 63 leaves/lr .05) + `src/predict.py` (test → submission.csv). **OOF RMSE = 14.98**
  (null 15.91, +0.93 ft). *Gate PASSED (barely).* Caveats logged below.

- ❌ **Phase 2 — Multi-scale NCC GR alignment.** *(verified dead 2026-05-27 — NOT the core signal)*
  Implemented faithfully (physics-informed `multi_scale_ncc`, hws 8/15/25, stride 3, softmax;
  reproduces the literature pooled r=0.996). **But that r is a cross-well level-matching artifact.**
  Within-well, NCC has ~0 drift signal: `corr(ncc_drift, true_drift) ≈ -0.03`,
  `corr(ncc_drift, phase1_residual) ≈ 0`, best linear blend moves OOF 14.979→14.973 ft (noise),
  beats null in 0% of wells. Tested anchor-ref and typewell-ref, seeds 80/150, 0.5/1 ft resampling —
  robustly ~0. Reason: a near-horizontal lateral's GR(MD) is dominated by lateral heterogeneity, not
  the few ft of vertical motion, so correlating it against a vertical GR profile can't resolve drift;
  and the rough TVT level it *does* get is already free from `last_known_TVT`. **Lesson: the plan's
  "NCC = core signal, r≈0.999" over-attributed value to a misleading pooled statistic.** Pivoted to
  Phase 3 (the verified lever). Probes: `/tmp/ncc_probe{,2,3}.py`.

- ✅ **Phase 3 — Formation plane-fit KNN (offset-well geometry).** *(productionized 2026-05-28)*
  Built `src/spatial.py`: `FormationPlaneKNN` (distance-weighted 2D plane through K=10 nearest
  **well centroids** per formation top, raw coords, LOO self-exclusion) + `RowKNN` (dense row-level
  ANCC, K=20 IDW, stride-3 ref ≈1.68M pts, LOO over-query buffer). `src/features.py` now builds two
  cached layers — base (44) + spatial (23) — merged on `id`; train uses LOO, test uses the full
  train reference. 23 features incl. `rk_tvt_formula`/`fk_tvt_formula` (`−Z+ANCC+b_well` drift, the
  dominant feature by 4.5× gain), `rk_dist`, `fk_vs_rk_ANCC`. **OOF 12.76** (heavy 160lv LightGBM).
  **Gate reframed:** the plan's ≤11.5 was set off konbu's *LB* 11.91; konbu's comparable *OOF* is
  **12.11**, and even that fails — the formation-KNN family tops out near here. Verified
  **signal-limited, not capacity-limited** (lean 63lv = 12.786, heavy 160lv = 12.762; a noise-level
  delta). **Banked**; the gap to ≤10.5 needs Phase-4 orthogonal signal, not more model/feature work
  on this family.
  > ⚠️ **CORRECTION (2026-05-29): "signal-limited" was WRONG.** Reproducing konbu's recipe on our data
  > hit OOF **11.87** with NO new orthogonal signal — just a richer feature set (incl. the GR/beam
  > features we deleted), full-density RowKNN, tighter reg, and an LGB×3+XGB stack. The 63lv-vs-160lv
  > test only varied capacity *on our 67-feat matrix*; it never tested a richer feature set, so it
  > could not detect feature-limitation. We were capacity/feature-limited. See Experiment Log 2026-05-29.

- ⚠️ **Phase 4 — Particle filter + beam search estimators. PARTLY REOPENED 2026-05-29.** The 2026-05-28
  abandonment conflated two claims: (a) GR can't be a precise *point estimator* (TRUE — fingerprint
  aliasing, ~4 ft autocorrelation ≪ ~12.7 ft anchor error), and (b) GR-matching features contribute ~0
  (FALSE — as weak GBM inputs they add **+0.252 ft**, ablation 2026-05-29). So: still don't build PF/DTW
  as point estimators, but konbu's beam-delta / offset-diff / NCC-shift / prefix-tw-rmse FEATURES are
  worth keeping (they're in the banked 11.885 recipe). _Original (over-broad) verdict kept below:_
  ~~Do NOT build PF/beam/DTW/NCC: all are GR-vs-typewell matching, which contributes ~0... Signal is
  present (oracle corr +0.72) but practically unreachable. Skip straight to Phase 5/6.~~
  _(original plan, kept for context:)_
  Numba PF (ANCC-anchored + Z-velocity) and beam/Viterbi over typewell GR (several stiffness
  configs). Add each estimator's drift + its disagreement vs NCC/KNN + PF uncertainty (std).
  *Gate: OOF ≤ 10.5. Expected ≈ 10.*
  ⚠️ **PREMISE UPDATED 2026-05-28 — the Phase-2 "GR-matching is dead" thesis was FALSIFIED by the
  premise probe (see Experiment Log).** GR strongly localizes TVT (oracle test: per-well median
  corr +0.72, sharp to ±4 ft; ~52% of eval GR variance explained at the true TVT). The Phase-2 root
  cause ("lateral GR = heterogeneity, can't resolve vertical motion") is WRONG — NCC and the
  literature's diagonal-band DTW are *extractor* failures (DTW gave 257 ft RMSE — band maps the narrow
  lateral band across the whole typewell). **So don't rebuild the literature's PF/beam/DTW verbatim**
  (PF(ANCC) is also redundant with `rk_tvt_formula`). **The right Phase-4 move: a KNN-anchored GR
  extractor** — target-distance features `GR − tw_gr(rk_tvt_formula + o)` and/or a local windowed GR
  search seeded on the Phase-3 KNN estimate, so the GBM *refines* the ~12 ft KNN within a tight window
  where GR is sharp. **Gate this on an extractability probe first** (KNN-anchored GR features → mini
  GBM on the phase3 residual → does it predict residual / beat OOF 12.76?), since the oracle test
  proves a high ceiling but not blind extractability. If the extractability probe is null, *then* skip
  to Phase 5.

- ⬜ **Phase 5 — Full feature set + GBM ensemble + blend.**
  Assemble the ~120–163 feature matrix (all signals + disagreements + GR/trajectory + target-
  distance). Train **LightGBM ×(2–3 seeds) + XGBoost + CatBoost**, GroupKFold(5), early stopping.
  Blend OOF with **NNLS** (non-negative) or hill-climbing. *Gate: OOF ≈ 10.0; first real LB
  submission. Expected LB ≈ 9.4.*

- ⬜ **Phase 6 — Post-processing.**
  Optuna-tune: global shrinkage `alpha` (drift × α), fade-in ramp `1 − exp(−md_since/τ)` near PS,
  blend weight with raw PF `w_pf`; then **Savitzky-Golay** per-well smoothing (window ~17, order 3)
  of the drift trajectory. Tune on OOF with the same objective as the metric. *Gate: OOF improves;
  LB ≤ 9.4. Expected LB ≈ 9.25–9.4.*

- ⬜ **Phase 7 — Diversity & stretch.**
  Multi-scale + stochastic (Gumbel) DTW as extra stacking features; more PF/beam configs; seed
  ensembling; consider a single heavily-regularised sequence model purely for blend diversity
  (only if it clears a strict gate). Adversarial-validation-weighted CV. *Gate: each addition must
  improve OOF without widening the OOF↔LB gap.*

---

## 6. Feature families (reference)

1. **Anchor/prefix**: last_known_TVT, known TVT range/std, slope(TVT,MD) all & recent-200,
   slope(TVT,Z), prefix typewell-fit RMSE, affine GR↔typewell calibration (a,b).
2. **Position**: row_from_PS, row_frac (+ frac², √frac), md/x/y/z-from-PS, xy & xyz distance,
   dz/dmd, dx/dmd, dy/dmd (trajectory geometry).
3. **GR**: rolling mean/std (w = 5/21/51/101), diffs (1/2), lags & leads (1/5/15/30), envelope,
   energy, **gr_missing flag** (GR is 28% NaN — important).
4. **NCC**: per-scale TVT estimate + score, consensus, softmax-ensemble; `sc_vs_beam` etc.
5. **Formation geometry**: `−Z + form + b_well` per top (full/late/WLS bias variants), spatial
   KNN distance, per-formation imputation RMSE, dense-ANCC value/std/dist.
6. **Sequence estimators**: PF(ANCC), PF(Z) drift + std; beam configs drift; mean/std/median across
   estimators; pairwise disagreements (pf_vs_z, dtw_vs_beam, sig_std as a confidence proxy).
7. **Target-distance**: `GR − typewell_GR(anchor + offset)` for anchor ∈ {last_known, NCC, beam,
   PF, DTW} and offsets spanning ±80 ft — encodes alignment self-consistency.

Always train on **drift**; add `last_known_TVT` back at inference. Never feature on train-only
columns (`TVT`, formation markers, typewell `Geology`) directly — only via imputed/derived signals.

---

## 7. Modeling & ensembling (Ch 8–10)

- **Base models**: LightGBM (num_leaves ~255, lr 0.02–0.03, n_est ~8000 + early stop 250),
  XGBoost (hist/gpu, depth 5–6, lr ~0.035), CatBoost (depth 5–7, lr 0.02–0.03). GroupKFold(5).
- **HPO** (Ch 9): Optuna TPE, ~60 startup + warm-start trials per model; or manual bisection on
  lr / num_leaves / reg_lambda. Tune *after* a stable feature set exists, not before.
- **Ensembling** (Ch 10): models are 0.98–0.99 correlated → **NNLS or hill-climbing** blend on OOF,
  not deep stacking. Expect LightGBM to sometimes get ~0 weight; XGB+CatBoost carry the blend.
  Diversity from seeds, feature subsets, and the orthogonal alignment signals matters more than
  another GBM.

## 8. Post-processing
Optuna-tuned drift shrinkage + PS fade-in + raw-PF blend, then per-well Savitzky-Golay smoothing.
Tune on OOF against the exact metric. (Worth ~0.1–0.15 ft.)

## 9. Compute & workflow
- **skynet (local, aarch64, `kaggle-arch`)**: CPU/numba feature generation (PF/beam/DTW/NCC/KNN over
  773 wells — hours). Cache matrices to `data/processed/`.
- **deepthought (`runon deepthought`, `kaggle` env, RTX 4080)**: GBM GPU training + Optuna sweeps.
  Check `ssh deepthought nvidia-smi` first (multi-tenant). Verify the ROGII path exists there
  (the project path differs across machines — see CLAUDE.md / former rsync note; confirm before
  dispatching).
- Independent jobs per machine; **no cross-machine DDP** (not needed — GBM, not deep nets).

## 10. Submission logistics (verify on site)
- Strongly indicated to be a **code competition**: the visible 3-well `test/` is replaced by a
  hidden set (~200 wells per public writeups) at scoring; top notebooks attach pre-trained
  artifact datasets and run an **offline inference kernel** that builds `submission.csv` (`id,tvt`).
- Plan accordingly: separate **train/feature-artifact build** (offline, our machines) from a thin
  **Kaggle inference notebook** (`jupyter/`, no internet, reads artifacts, ~20 min for ~200 wells).
- `id = "<well_id>_<row_index>"`, scored rows = rows ≥ PS. Validate id set before submitting.
- **TODO**: confirm on the Kaggle site — code vs CSV, runtime/internet limits, public/private split,
  and whether the live metric label is MSE or RMSE (ranking identical either way).

## 11. Risks, pitfalls, open questions
- **GR missingness (28%, up to 73%)**: alignment degrades where GR is absent; impute + flag, and
  lean on the formation-geometry signal there.
- **Hidden-test distribution shift**: only 3 visible test wells; run adversarial validation early.
- **Feature-gen runtime**: numba JIT + 773 wells is slow; cache aggressively, parallelise per well.
- **Overfitting the blend**: keep it NNLS/hill-climb; re-check OOF↔LB gap on every change.
- **Don't chase sequence models** despite the sequential framing — the evidence says they don't
  generalise on 773 wells.
- **Open**: exact submission mechanics & metric label (above); whether external/regional geology
  data is allowed (check rules before using anything beyond the provided wells).

## 12. Kaggle Book references used
Ch 6 (metrics — build an exact local RMSE; optimise the metric directly), Ch 7 (validation —
GroupKFold, adversarial validation, trust-CV, submission selection), Ch 8 (tabular — GBDT
dominance, feature engineering, GBDT+DNN blends), Ch 9 (Optuna TPE / bisection HPO), Ch 10
(NNLS/hill-climb blending over deep stacking; diversity), Cross-cutting (seeds, experiment
tracking, the 10-step winning workflow).

---

## Experiment Log
_(newest first; append dated entries as work proceeds)_

- 2026-05-29 — **Re-audit probe #1: multi-scale NCC as GBM features = NULL (+0.004 ft).** Tested the
  Phase-2 "NCC dead" verdict in the production frame (9 multi-scale NCC feats — per-scale drift+score
  hws 8/15/25, softmax blend, cross-scale std — added to the cached konbu 78-feat matrix, single GPU-LGB,
  same shuffled GKF-5; `experiments/ncc_feature_ablation.py`). BASE 78 = 12.0960, BASE+NCC = 12.0920 →
  **+0.0039 ft (noise)**. **The frame-error hypothesis did NOT hold here** (unlike GR's +0.25). Reason:
  konbu's base ALREADY has the GR-vs-typewell matching features (`tw_diff_*`, `ncc_*_shift_well`, beam) —
  multi-scale NCC is **redundant** with them (same alignment signal). So NCC the signal isn't dead
  (~0.25 ft lives in those base features), but multi-scale NCC *specifically* adds nothing on top.
  **Refined lesson: re-testing in the right frame can CONFIRM a "dead" verdict, not only overturn it** —
  the discipline ("test as a GBM feature on the current best matrix") is what's required, but the answer
  is sometimes still dead. Remaining re-audit candidates (lower expected value now): postproc on the konbu
  OOF (probe #2, MEDIUM — base-model-dependent), normalized-DTW-as-feature (LOW), seq-model blend (LOW).

- 2026-05-29 — **Phase-4 "GR conclusively dead" verdict OVERTURNED (quantified).** Ablation on konbu's
  cached 78-feat matrix (`/tmp/phase4_reexam.py`, single GPU-LGB, same shuffled GKF-5): ALL-78 OOF
  **12.072** vs WITHOUT the 18 GR-vs-typewell *matching* features **12.324** → the "dead" features
  contribute **+0.252 ft**. Matching group = `beam_cons/loose_delta`, `beam_gap`, all 11 `tw_diff_*`
  offset-diffs, `ncc_med/mean_shift_well`, `prefix_tw_rmse/mae`. **Reconciliation:** our Phase-4 probe
  was right that GR can't be a precise *point estimator* (aliasing — can't pin TVT to the correct ~4 ft
  fringe), but that's a DIFFERENT question from "do these carry exploitable signal as weak GBM inputs."
  They do (~0.25 ft). The error was going from "bad estimator" → "delete the features." **Lesson: test a
  signal in the frame it will actually be used (GBM feature) before declaring it dead.** Productionized
  konbu run banked OOF **11.885** (GPU LGB; CPU diagnostic was 11.871 — noise-level identical, confirms
  GPU drop-in). Models + cached feats saved under `models/konbu/`, `data/processed/konbu/`.

- 2026-05-29 — **konbu's recipe reproduced on OUR data → OOF 11.8706. This OVERTURNS the banked
  "signal-limited, tops out at 12.11" conclusion.** Pulled konbu17's actual notebook
  (`rogii-plane-fit-formation-top-knn`, doc'd OOF 12.11 / LB 11.912) and ran it on our `data/raw`
  (`/tmp/konbu_repro.py`, GroupKFold-5 shuffled, XGB on GPU, LGB on CPU). Result: single LGB seeds
  **12.065 / 11.997 / 12.114**, single XGB **11.939**, simple-avg **11.903**, **Ridge stack 11.871**
  (weights lgb7 .35 / xgb .53 / lgb42 .12). **Every single model beats our banked 12.76 by ~0.7–0.9 ft
  — before any ensembling.** Diagnosis of the gap (audited konbu's code line-for-line):
  **(1)** our `FormationPlaneKNN` is IDENTICAL to konbu's → the anchor is NOT the deficit;
  **(2)** konbu keeps ~13 GR/beam/offset/FFT/NCC-shift features we DELETED in Phase 4 (78 feats vs our
  67); **(3)** full-density RowKNN (5.05M rows) vs our stride-3 (1.68M); **(4)** tighter regularization
  (89 leaves + heavy L2 vs our 160lv); **(5)** LGB×3 + XGB + Ridge stack vs our single LGB (ensemble is
  the SMALLEST lever — ~0.13 ft from single-XGB 11.94 → stack 11.87). **The plan's claim that the
  formation-KNN family is "signal-limited, not capacity-limited" and "tops out at 12.11, needs Phase-4
  orthogonal signal" was WRONG** — 11.87 was reached with NO new signal, just richer features (incl. the
  "dead" GR ones) + density + ensemble. ⚠️ **Re-examine the Phase-4 "GR conclusively dead" verdict** — we
  tested GR as a residual point-estimator, never as weak GBM inputs alongside everything else, which is
  how konbu uses it. Timing: 3.3 hr total (~1.8 hr full-density feature build, ~1.5 hr CPU LGB×3).
  ALSO this session: **GPU LightGBM now BUILT on skynet** (CUDA/sm_121 from source, 6.73× faster than CPU,
  identical RMSE — recipe in memory `lgb-cuda-build-skynet`); **xgboost now installed** in `kaggle-arch`
  (GPU works on GB10). Next: productionize konbu's recipe (cache feats + save models), inference kernel, submit.

- 2026-05-28 — **FIRST LB SUBMISSION — code-competition pipeline BUILT, validated, submitted (score
  pending).** Confirmed it's a **code competition** (CLI CSV submit → HTTP 400; only 3 visible test
  wells ship; top solutions are notebooks). Built the offline inference kernel `jupyter/rogii_inference.py`
  (+ `kernel-metadata.json`) that reuses our tested `src/` path via three shims — `C.RAW`→competition
  input, `C.MODELS`→decompressed fold models, `C.PROC`→/kaggle/working — and self-locates inputs.
  Packaged `src/` + the 5 fold models as private dataset `stevewatson999/rogii-artifacts`. Dry-ran the
  exact kernel logic locally (zipimport + gunzip + Kaggle's `src/src/` double-nesting) → byte-identical
  to `submission_phase3.csv`. Kernel v4 ran clean on Kaggle (14,151 rows). **Submitted to LB; score
  PENDING** (`kaggle competitions submissions`). **Mechanics gotchas (hard-won over 4 kernel versions):**
  (1) CLI-attached sources mount **NESTED** — `/kaggle/input/competitions/<slug>/` and
  `/kaggle/input/datasets/<user>/<slug>/`, NOT flat `/kaggle/input/<slug>/` (the web-UI convention) →
  kernel must recurse to find them; (2) Kaggle **auto-extracts** uploaded `.zip`/`.gz` — `src.zip`
  became `src/src/...`, models `.gz`→`.txt`; (3) brand-new private datasets need a few min to propagate
  before they attach; (4) final **Submit to Competition is web-only** (CLI can't submit code-comp output)
  — done by the user. **NB: the public LB top has moved 9.25 → 7.5 since the plan's research (Deotte 8.6)
  — our 12.76 OOF is far back; the live lever looks like a much tighter spatial anchor (see prior entry).**
- 2026-05-28 — **Phase 4 (GR-vs-typewell matching) CONCLUSIVELY DEAD for us — but for a deeper,
  correct reason than the plan first gave.** Three independent probes:
  (1) **premise/oracle** — GR localizes TVT strongly (corr +0.72 at the TRUE tvt) but only within
  **±4 ft**; (2) **extractability** — per-row target-distance anchored on the Phase-3 KNN (~12.7 ft off)
  → R² +0.001, RMSE 12.71→12.69 (noise); (3) **windowed-sequence match** seeded on KNN, ±15 ft local
  search (`/tmp/phase4_window_probe.py`) — corr(δ*, residual) **+0.007**, (anchor+δ*) RMSE 12.71→**15.51
  (worse)**, and even on the 39% of rows with windowed peak corr >0.5 the corr with residual is +0.004.
  **Root cause = fingerprint ALIASING:** the GR autocorrelation length (~4 ft) is *finer* than the
  anchor uncertainty (~12.7 ft ≈ 3 fringes), so any blind search locks onto the wrong fringe. The
  oracle only worked because it evaluated *at* the true TVT (no search). This is fundamental
  ill-posedness, not a fixable extractor — and it's why NCC, the literature's diagonal-band DTW
  (257 ft RMSE), beam, and PF all contribute ~0. **Implication for the published 9.251:** its
  GR-matching features face the same aliasing and contribute ~0; its 12.76→~10 OOF gain comes from
  **non-GR sources** — richer formation-KNN (konbu 12.11), the 6-model ensemble (LGB×3+CatBoost×3) +
  hill-climb, and especially Optuna drift-shrinkage + PS fade-in + Savitzky-Golay postproc. **Roadmap
  reversed: Phase 4 is abandoned; the real leverage is Phase 5 (richer KNN + ensemble + blend) and
  Phase 6 (postproc), which the plan under-weighted.** Probes: `/tmp/phase4_premise_probe.py`,
  `phase4_extract_probe.py`, `phase4_window_probe.py`.
- 2026-05-28 — **Phase-4 PREMISE PROBE: the "GR-matching is dead" thesis is FALSIFIED — signal
  exists, extraction is the bottleneck.** Probe (`/tmp/phase4_premise_probe.py`, 80 sample train
  wells; per-well table `/tmp/phase4_premise_probe_perwell.csv`). **[A] Oracle physics test:**
  corr(GR_eval[i], typewell_GR(TVT_true[i] + offset)) **peaks SHARPLY at offset 0** — mean +0.649,
  per-well median **+0.721** (p25 +0.576), collapsing to ~0 by ±4 ft; 94% of wells peak within ±8 ft;
  high-drift wells (range ≥30 ft) even stronger (+0.746). So **~52% of eval-zone GR variance is
  explained by the typewell GR-vs-TVT curve at the TRUE TVT, resolved to ~4 ft.** This directly
  refutes the Phase-2 root cause ("lateral GR ≈ heterogeneity, can't resolve vertical motion").
  Phase 2/this probe show NCC and DTW *extractors* fail, NOT that signal is absent. **[B]** DTW point
  estimate (lifted verbatim from 9.251) is garbage — pooled corr w/ true drift +0.059, beat-null 0/80,
  median RMSE **257 ft** (Sakoe-Chiba diagonal band maps the narrow lateral TVT band across the whole
  typewell range — geometrically broken; the GBM must ignore it). **[C]** DTW path slope corr +0.003
  (dead). **[D]** On top of phase3, the (broken-DTW, ill-posed single-row) extractors add ~0 — but both
  are known-bad, so D is uninformative about a *good* extractor. **Reconciliation:** the 9.251 gain
  isn't its DTW; it's **target-distance features** (`GR − tw_gr(anchor+o)`) over a GBM, which only work
  if anchored on a roughly-right per-row TVT. They anchor on `last_tvt` (constant, weak) + garbage
  estimators. **Untested promising path: anchor target-distance / a local windowed GR search on the
  Phase-3 KNN per-row estimate (~12 ft) and let the GBM refine within a tight window** — exactly where
  Test A says GR is sharp. Next probe: build KNN-anchored GR features → mini-GBM on the phase3 residual
  → does it predict residual / beat 12.76? (Caveat: A is an *oracle* ceiling, not proof of blind
  extractability — D's nulls keep the skeptical case alive until the extractability probe lands.)
- 2026-05-28 — **Phase 3 fully banked.** Productionized retrain finished clean (`log/train_20260528_164526.log`
  `[done]`, 6324s): **OOF RMSE 12.7608** (MSE 162.84, MAE 8.75, n=3,783,989; null 15.910, −3.149 ft),
  matching the push value — confirms productionization didn't regress. Per-fold TVT RMSE
  [10.77, 12.71, 13.02, 13.02, 14.01]; top gain still `rk_tvt_formula` (#1) ≫ `fk_tvt_formula` >
  `slope_tvt_md_recent`. `python -m src.predict` wrote `data/processed/submission_phase3.csv`
  (14,151 rows, TVT∈[11585.8, 12237.5], mean 11905.6; 3-visible-well in-fold sanity RMSE 4.795).
  Phase 3 closed. **Next = Phase 4 premise probe (cheap, GR-matching) — see §5 caveat.**
- 2026-05-28 — **Phase 3 PRODUCTIONIZED + push result; banked at family ceiling.** Overnight push
  (`/tmp/phase3_gbm_probe3.py`: heavy 160lv/lr.03/4000-ES200 over stride-3 RowKNN) → **OOF 12.762**,
  vs lean 12.786 — a noise-level delta, confirming Phase 3 is **signal-limited, not capacity-limited**.
  Pushing the formation family further is dead; gap to ≤10.5 needs Phase-4 orthogonal signal.
  **Productionized the probe into `src/`:** new `src/spatial.py` (`FormationPlaneKNN` centroid plane
  + `RowKNN` dense ANCC, both with LOO gotchas documented); `src/features.py` refactored into two
  cached layers — `features_base_{split}.parquet` (44) + `spatial_{split}.parquet` (23) — merged on
  `id` in `build_feature_matrix` (train=LOO self-exclusion, test=full train-ref). `src/train.py` +
  `src/predict.py` bumped to the heavy config + `phase3`-tagged artifacts. Matrix verified: 67 feats,
  **0 rows missing spatial** on train (3.78M) and test (14,151). `spatial_train.parquet` seeded from
  the validated `phase3_feats_s3.parquet` (identical code) to skip the 28-min recompute. **Retrain
  in flight** (`python -m src.train`) to regenerate productionized `models/lgb_phase3_fold*.txt` +
  `oof_phase3.parquet`; **predict not yet run** → `submission_phase3.csv` pending. Confirm OOF via
  `log/train_*.log` `[done]` + `data/processed/phase3_result.json` (expect ~12.76), then run
  `python -m src.predict`. **Next: Phase 4 — but probe its premise cheaply first (see §5 Phase 4
  caveat: beam/PF share NCC's dead GR-matching premise).**
- 2026-05-28 — **Phase 3 (formation KNN) validated: OOF 12.786 ft** (Phase-1 14.979, null 15.910;
  −2.19 ft). Probe added konbu-faithful **FormationPlaneKNN** (well-centroid K=10 weighted 2D plane,
  6 tops) + **RowKNN** (dense row-level ANCC, K=20 IDW, ref subsampled stride-5 → 1.01M pts), LOO,
  per-well `b_well` from anchor. Merged 23 features onto the 44-base matrix (67 total), retrained
  LightGBM GKF-5 (lean params, cap 3000). Steps: plane-only → 13.953; +RowKNN → **12.786**.
  `rk_tvt_formula` (dense `−Z+ANCC+b_well` drift) is the #1 feature by gain by 4.5×; `fk_tvt_formula`,
  `rk_dist` (impute quality), `fk_vs_rk_ANCC` (plane-vs-row agreement) also top-15. **Misses gate
  ≤11.5** (and ≤12.5 by 0.29) — attributable to: lean gate-model (folds converged in 46–154 rounds
  → not capacity-tuned; plan §7 heavy model is Phase-5), stride-5 RowKNN subsampling, and ~80-feat
  konbu set incl. beam/typewell (Phase 4). Probe code: `/tmp/phase3_gbm_probe2.py`. NOT yet
  productionized into `src/features.py`; submission not regenerated. Decision pending: bank+advance
  to Phase 4 vs push to ≤11.5 (full-density RowKNN + heavier model) first.
- 2026-05-27 — **Phase 2 NCC verified DEAD; pivoting to Phase 3.** Implemented the physics-informed
  `multi_scale_ncc` (hws 8/15/25, stride 3, softmax) faithfully — reproduces the literature's pooled
  r=0.996 of ncc_tvt vs true_tvt. **That r is a cross-well artifact** (NCC nails each well's ~11–12 k ft
  *level*, which `last_known_TVT` already gives). Within-well it carries no drift signal:
  `corr(ncc_drift, true_drift)=-0.03`, `corr(ncc_drift, phase1_residual)≈0` (typewell -0.029, anchor
  -0.001), best linear blend 14.979→14.973 ft, beats null in 0% of 80 sample wells. Robust across
  anchor- vs typewell-reference, seed radius 80/150, 0.5/1 ft typewell resampling, eval-range>30 ft
  subset. Root cause: near-horizontal lateral GR(MD) ≈ lateral heterogeneity, not vertical motion, so
  it can't match a vertical GR barcode at ft-scale. Confirms Phase-1's "GR features barely register"
  and the score ladder (Mitch NCC+KNN = 14.99 ≈ our 14.98). **Independently verified the Phase-3
  foundation instead:** `TVT = -Z + ANCC + b_well` exact (within-well resid std median 0.0065 ft, max
  0.0087, n=766); ANCC spatially smooth (LOO KNN-15-mean impute 78 ft vs 628 ft global). Probes left
  at `/tmp/ncc_probe{,2,3}.py`. Next: Phase-3 formation plane-fit KNN.
- 2026-05-27 — **Phase 1 done. OOF RMSE = 14.9793** (null 15.9099, +0.93 ft). LightGBM on drift,
  44 simple feats, GroupKFold-5, lean params (63 leaves, lr .05, ~700 iters avg). Gate (≤15) passed
  but **fragile**: folds 1 & 4 (16.16, 16.24) are *worse than null*; gain comes from folds 0/2/3.
  **Feature importance is dominated by anchor-zone trend-extrapolation + per-well constants**
  (slope_tvt_md_recent, slope_tvt_z_all, anchor_tvt, eval_len, ps_len); per-row GR features barely
  register. This is NOT a geosteering model — it extrapolates the landing trend. Confirms the plan
  thesis: the real lever is Phase-2 NCC alignment, not more simple GR stats.
  Pipeline now end-to-end: `src/{features,train,predict}.py` → submission_phase1.csv (14,151 rows,
  TVT∈[11597,12234]). Sanity vs 3 visible wells RMSE 4.07 (optimistic/in-fold). NOT yet submitted.
  Compute note: ran on skynet CPU (20 cores, 1190s) — neither env has a GPU LightGBM build.
- 2026-05-27 — **Phase 0 done.** Data download landed (773 train + 3 test + 773 PNGs). Built
  `src/harness.py`: GroupKFold-by-well, simulate-test-on-train, null baseline. Null RMSE =
  **15.9099 ft** over all 773 wells (vs 16.18 on the earlier 269-well subset — full data matches
  the published 15.91 exactly). Test eval ids match `sample_submission.csv` (14151). Gate passed.
- 2026-05-26 — Plan authored from deck + KaggleBookSummary + 10 public notebooks + local EDA
  (269 wells). Verified: null 16.18 ft, drift std 16 ft, `TVT=−Z+ANCC+b_well` r=1.0000, GR 28% NaN.
  Research notebooks cached at `/tmp/rogii_research/` (not committed).
