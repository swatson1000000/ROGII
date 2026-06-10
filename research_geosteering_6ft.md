# Research: Geosteering / Petrophysics Techniques to Close the ~2 ft RMSE Gap

**Competition:** ROGII — Wellbore Geology Prediction (predict TVT, RMSE in feet)
**Our LB:** 8.16 · **Best published notebook:** ~7.78 · **Actual leaders:** 5.99 (nothing published)
**Date:** 2026-06-07

---

## 0. TL;DR verdict (read this first)

The honest answer to "method vs. data" is **mostly method, but a narrow, specific method we are NOT
doing — and one data artifact we should rule out before chasing geology.**

1. **The single most promising untried lever is dip-aware / structural correlation, not a better GR
   matcher.** Everything we do (NCC, PF, beam, DTW) treats each well's GR-vs-typewell alignment as a
   *per-well 1-D shift/warp*. The petroleum-geology literature is emphatic that the horizontal-vs-typewell
   relationship is governed by **apparent dip along the lateral**, and that TVT is itself a *dip-distorted*
   quantity (TST = TVT·cos δ-type relations). A method that jointly estimates a **dip field across wells**
   (structural framework / trend surface from many geosteered laterals) and propagates it can place a well
   the single-typewell correlation cannot — this is exactly the difference between "TSP" geosteering (what
   we do) and "TST/structural-model" geosteering (what ROGII's own product and the SPE literature do). See
   §2, §3.

2. **ROGII's own inversion (StarSteer stochastic/deterministic) is NOT buildable here.** It is an
   *azimuthal-resistivity* forward-model + inversion. It requires resistivity curves and LWD tool
   specifications we do not have ([[no-resistivity-channel-in-data]]). The GR-only analog (azimuthal/imaging
   GR inversion) needs *azimuthal* GR (up/down sectors), which we also lack. This thread is a dead end. See §1.

3. **The full Bayesian / EnKF / RL geosteering machinery (Kullawan-Bratvold, Alyaev/NORCE, DISTINGUISH) is
   a decision-making and look-ahead framework, not a TVT-from-GR estimator.** Its *measurement-update* core
   (a particle filter / EnKF over formation-boundary position from GR likelihood) is precisely what our PF
   already is. We are not missing the estimator; we may be missing its **multi-well prior** (a structural
   surface with spatial covariance), which is the same point as #1. See §3.

4. **Rule out a data artifact first.** The leaders are at 5.99 with *nothing published* in a board that
   moved from ~9 to sub-7 fast ([[lb-board-moved-sub7]]). Before attributing 2 ft to geology, confirm it is
   not (a) a public/private split quirk, (b) a typewell↔well *pairing/ID* leak, or (c) a different ANCC
   imputation regime. The competition has an explicit "Leakage-Aware" notebook and a leakage discussion
   thread — that is a signal worth reading directly. See §5.

**Ranked shortlist of what to actually try** (full detail §6):

| Rank | Lever | Buildable from GR+traj+typewell only? | Expected | Difficulty |
|------|-------|----------------------------------------|----------|------------|
| 1 | **Cross-well dip field → structural ANCC surface** (universal/regression kriging of ANCC with a dip trend, replacing plane-fit/dense KNN) | Yes (ANCC train-only, but the *trend model* transfers) | Plausible 0.2–0.5 ft | Medium |
| 2 | **Apparent-dip-corrected GR matching** (estimate per-well apparent dip from the GR↔typewell stretch, feed dip as a feature + as a warp constraint) | Yes | Plausible 0.1–0.3 ft | Medium |
| 3 | **RGT / multi-well joint correlation** (Wheeler–Hale relative-geologic-time, least-squares-consistent shifts across all wells, not pairwise) | Yes | Speculative 0.1–0.3 ft | High |
| 4 | **GR-shape/texture features beyond NCC** (CWT/DWT detail-coefficient correlation, DFA exponent) as new stack features | Yes | Small, ≤0.1 ft | Low–Med |
| 5 | **Leakage / split diagnosis** (read the forum, audit pairing) | N/A | Unknown, possibly large | Low |

I cannot honestly say any single one of these closes the full 2 ft. The leaders' gap is more likely **a
combination of #1 (structural prior) + an exploit we cannot see (#5)** than one missing algorithm.

---

## 1. ROGII StarSteer inversion — what it actually is, and why it's a dead end here

**Method.** StarSteer offers **deterministic** and **stochastic** inversion of **ultra-deep azimuthal
resistivity (UDAR)** data. The forward model "combines LWD vendor tool-string specifications and a set of
resistivity curves, a theoretical earth model derived from one or more resistivity profiles based on offset
type wells, and the location/position of the wellpath inside the reservoir"
([ROGII, Stochastic Inversion in StarSteer](https://www.rogii.com/blog/stochastic-inversion);
[ROGII, UDAR inversion](https://www.rogii.com/blog/rogii-advances-reservoir-mapping-with-ultra-deep-azimuthal-resistivity-data-inversion)).
Stochastic inversion returns P10/P50/P90 boundary maps; deterministic returns a high-resolution resistivity
distribution. The math is a **Bayesian inverse problem**: given measured azimuthal-resistivity curves *d*,
find the layered earth model *m* (bed boundary depths, resistivities) whose forward-modeled response best
matches *d*, with uncertainty.

**Buildable from GR+traj+typewell only? NO.** Three hard blockers:
- It is keyed on **resistivity curves + tool specs**. We have neither ([[no-resistivity-channel-in-data]]).
- The GR analog ("A novel method for quantitative geosteering using azimuthal gamma-ray logging",
  [ScienceDirect S096980431400400X](https://www.sciencedirect.com/science/article/abs/pii/S096980431400400X);
  "A fast forward algorithm for real-time geosteering of azimuthal gamma-ray logging",
  [ScienceDirect S0969804317302580](https://www.sciencedirect.com/science/article/abs/pii/S0969804317302580))
  inverts the **up- vs. down-facing (azimuthal sector) GR readings** to get distance-to-boundary. Our GR is
  a single non-azimuthal trace — there is no front-to-back ratio to invert.
- The "theoretical earth model from offset type wells" *is* our typewell, and matching the wellpath to it
  *is* our NCC/PF. So the part of StarSteer we could mimic, we already do; the part that buys ROGII its
  accuracy (azimuthal-resistivity distance-to-boundary) is unbuildable.

**Verdict: dead.** Confirms the existing memory note. Do not spend time here.

Sources: [StarSteer product](https://www.rogii.com/products/starsteer),
[ROGII Geosteering](https://www.rogii.com/solutions/geosteering),
["Modeling and Inversion with Azimuthal Gamma Ray", SPE ATCE 2020](https://onepetro.org/SPEATCE/proceedings-abstract/20ATCE/20ATCE/D022S061R033/449842).

---

## 2. Structural / dip-based correlation — the strongest untried idea

This is the part of the geosteering literature our pipeline most clearly **does not** implement, and it
attacks TVT at the level of the *geometry that defines it*.

### 2.1 Why TVT is a dip problem, not just a matching problem

TVT (true vertical thickness) is **not** a dip-invariant quantity. The thickness/dip relations are explicit
in the petrophysics literature:

- For a vertical penetration, true (stratigraphic) thickness = H·cos D, with D the dip
  ([AAPG Wiki, Depth and thickness conversion](https://wiki.aapg.org/Depth_and_thickness_conversion)).
- In a deviated/horizontal well, **"bed thickness appears too great, the amount depending upon the direction
  of dip... and the drift angle and direction of the borehole. If the dip is in the same direction as the
  deviation, the unit appears thicker than it actually is; if opposite, it is shortened"**
  ([AAPG Wiki](https://wiki.aapg.org/Depth_and_thickness_conversion)).
- The TST↔TVD relation:
  `TST = (TVD_b − TVD_t)·cos δ' − sqrt((NSD_b−NSD_t)² + (EWD_b−EWD_t)²)·sin δ'`, where **δ' is the apparent
  dip in the direction of horizontal displacement**
  ([resdip.com, Calculating TST](https://resdip.com/docs/calculating%20TST.pdf);
  [Holt & Schoonover, SPWLA 1977, "True Vertical Depth, True Vertical Thickness and True Stratigraphic
  Thickness Logs"](https://onepetro.org/SPWLAALS/proceedings/SPWLA-1977/SPWLA-1977/SPWLA-1977-Y/19871)).
- Apparent vs. true dip: `tan β2 = cos(α)·tan β1` (β1 true dip, β2 apparent dip along a section of azimuth α)
  ([Modeling and Inversion with Azimuthal GR, ResearchGate 346223913](https://www.researchgate.net/publication/346223913_Modeling_and_Inversion_with_Azimuthal_Gamma_Ray_for_a_Better_Geosteering_Decision-Making)).

**Implication.** As a horizontal well steps out, the mapping from "position in the typewell GR" to TVT is
modulated by the *local apparent dip*, which varies along the lateral and between wells (folds, faults, dip
changes). Our NCC/PF find *where in the typewell* a GR segment matches; converting that to TVT correctly
needs the dip. We currently lean on the geometric identity `TVT = −Z + ANCC + b_well` and a per-well constant
`b_well`, with ANCC imputed by plane-fit/dense KNN. **A constant b_well + a locally-planar ANCC bakes in a
single dip per neighborhood; it cannot represent within-well dip change or fault throw**, which the
literature says is exactly where single-typewell correlation fails.

### 2.2 The TSP-vs-TST distinction = exactly our gap

The 2013 URTeC paper **"Geosteering Using True Stratigraphic Thickness"**
([resdip.com PDF](https://resdip.com/download/geosteeringUsingTST.pdf);
[OnePetro URTEC-1590259-MS](https://onepetro.org/URTECONF/proceedings-abstract/13URTC/All-13URTC/URTEC-1590259-MS/149030))
draws the precise line:

- **TSP (true stratigraphic position) geosteering** "uses relative position to stretch and squeeze the log
  in the well to match a template." **This is our method** (NCC/DTW/PF all stretch-squeeze GR to the typewell).
- **TST geosteering** "is used to **model the effect of changes in dip or faults** on a template log from an
  offset well" and is "**superior to TSP because it can look ahead of the bit**" — i.e., it carries an
  explicit **structural model** (dip + faults) rather than a per-well elastic warp.

That is the cleanest statement in the literature of *why* a structural-model approach beats per-well GR
matching: TSP confounds dip change with stratigraphic mismatch; TST separates them.

### 2.3 Multi-well structural framework / trend surface

The operational version of this — and the one that maps onto our 200-well inference set — is the **dynamic
structural-framework update from many geosteered laterals**:

- A horizontal-well-correlation workflow "allows update of a dynamic-framework structural model, **taking
  into account changes in the dip of the beds**" and flags faults when "large discrepancies between points
  on the modeled surface" appear
  ([JPT, "Horizontal-Well Correlation in Geosteering Complex Reservoirs of Saudi Arabia"](https://jpt.spe.org/horizontal-well-correlation-geosteering-complex-reservoirs-saudi-arabia)).
- "Constraining faults and stratigraphic zones... via 3D geocellular models" describes three concrete
  building blocks, two of which we could do: **"calculation of trend surfaces from thousands of geosteered
  3D horizontal-well position logs"** and **"residual analysis of regional and local horizontal-well trend
  surfaces to identify faults"**
  ([ScienceDirect S2949891024003610](https://www.sciencedirect.com/science/article/abs/pii/S2949891024003610)).
- The newest SPE work, **"Geosteering: Continuous Surface Model Updates Using Gamma Log" (ATCE 2025,
  SPE-227995-MS, NORCE)** is *literally GR-only continuous structural-surface updating*: "consistently
  integrating the structural model with LWD measurements... When resistivity contrasts are not observed,
  **gamma ray becomes crucial**"
  ([OnePetro 25ATCE D021S015R008](https://onepetro.org/SPEATCE/proceedings/25ATCE/25ATCE/D021S015R008/792004);
  [NORCE listing](https://nr.no/en/publication/10290186/)). This is the closest published analog to the exact
  problem and it is built from GR + trajectory + structural surface — our exact inputs.

**Buildable from our data? YES, in spirit.** ANCC is train-only, but the *structural trend model* (how ANCC
varies with X,Y, and how the well's own GR-derived TVT residuals reveal local dip) is the transferable object.
Concretely: replace plane-fit/dense-KNN ANCC imputation with a **universal kriging / kriging-with-a-dip-trend**
of ANCC (see §6.1), and add an **apparent-dip feature** per well (§6.2).

⚠️ **Adversarial check.** Does the structural lever secretly need data we lack? Two real risks:
(a) **Faults** require dense well control to resolve; with ~200 sparse hidden laterals we may not localize a
fault throw — so the *fault* part is likely not recoverable, only the smooth-dip part. (b) Our LOO/block-holdout
memory ([[gate-spatial-levers-with-block-holdout]]) shows density-coupled spatial estimators (GP/IDW/KNN)
**over-credit on single-well LOO and regress on the hidden set** — a fancier kriging is in the *same family*
that already burned us (GP: 24.50 LOO → 47.95 block-holdout → LB regression). So the dip-trend kriging MUST be
gated by **region/block holdout**, not single-well LOO, or it will look great and ship a regression. This is
the one lever I would build, but with eyes open that it lives in the family that has already failed once.

---

## 3. Bayesian / EnKF / RL geosteering — already have the estimator, maybe missing the prior

**The literature.** Kullawan, Bratvold & Bickel built the canonical **Bayesian geosteering** framework:
a Gaussian joint distribution over distances-to-boundary, updated by Bayes' rule as measurements arrive,
used for decision-making and value-of-information
(["A Decision Analytic Approach to Geosteering Operations", SPE-167433-PA](https://www.onepetro.org/journal-paper/SPE-167433-PA);
["Decision-Oriented Geosteering and the Value of Look-Ahead Information", SPE-184392-PA](https://www.onepetro.org/journal-paper/SPE-184392-PA)).
The NORCE/Alyaev line generalizes the *measurement update* to **ensemble methods**: EnKF/ensemble-smoother
over an ensemble of geomodels
(["Ensemble-Based Well Log Interpretation and Uncertainty Quantification for Geosteering", arXiv 2103.05384](https://arxiv.org/pdf/2103.05384)),
and the RL+PF and DISTINGUISH papers add decision optimization on top
(["High-Precision Geosteering via RL and Particle Filters", arXiv 2402.06377](https://arxiv.org/abs/2402.06377);
["DISTINGUISH Workflow", arXiv 2503.08509](https://arxiv.org/abs/2503.08509)).

**Adversarial read — is any of this a TVT estimator we're missing?** Mostly no:

- **Decision/VOI/RL layers are irrelevant.** We are not steering a well; we are post-hoc estimating its TVT.
  No reward, no action. RL and DDP add nothing.
- **The PF/EnKF measurement-update IS what we already do.** "Gamma-ray log data can be used as inputs to the
  particle filter... outputs become the primary criterion" (arXiv 2402.06377) — that is our 128-seed
  likelihood-weighted PF, blended at w≈0.57, our single biggest lever. We are not missing the estimator class.
- **What the literature has that we don't: a multi-well, spatially-correlated PRIOR.** The Bayesian/ensemble
  formulations carry a *geomodel ensemble with spatial covariance* (DISTINGUISH's GAN, EnKF's ensemble of
  surfaces). Our PF prior is per-well and flat; our spatial prior (KNN ANCC) is a *separate* GBM feature, not
  fused into the sequential filter. The principled upgrade is to make the PF's transition/observation model
  carry a **spatial prior on b_well/ANCC and on local dip** — i.e., couple §2 into the PF. That is a real,
  buildable idea, but it is an *integration* of two things we already have, not a new technique.

**Buildable from our data? The estimator yes (have it); the GAN/ensemble geomodel — no**, that needs a
training corpus of facies realizations we don't have, and our own memory says sequence/generative models
memorize on 773 wells ([[transductive-thread-dead]] context; RNN/CNN/transformer fail at ~14–15 ft).

**Verdict:** Don't build EnKF/RL/GAN. The *one* transferable idea is **fusing a spatial dip prior into the PF
observation model** — which is §2 wearing a Bayesian hat. Rank it with §2, not separately.

---

## 4. GR curve shape / texture beyond cross-correlation

The well-log-correlation literature offers shape descriptors our NCC/DTW don't directly use:

- **CWT/DWT + DTW correlation.** "A step toward practical stratigraphic automatic correlation of well logs
  using continuous wavelet transform and dynamic time warping"
  ([ScienceDirect S0926985118304336](https://www.sciencedirect.com/science/article/abs/pii/S0926985118304336))
  builds *correlatable spectral-trend logs* via CWT then correlates with DTW. The detail-coefficient bands
  isolate stratigraphic texture at chosen scales — a different similarity surface than raw-amplitude NCC.
- **DWT detail/approximation correlation** for boundary detection
  ([Springer s40948-016-0027-1](https://link.springer.com/article/10.1007/s40948-016-0027-1)).
- **DFA fractal scaling exponent** as a facies descriptor
  ([ScienceDirect S0378437113006833](https://www.sciencedirect.com/science/article/abs/pii/S0378437113006833)).
- **DTW Barycenter Averaging (DBA)** to build a *robust* typewell template and correlate against it
  ([The Sedimentary Record, Gotland DTW+DBA](https://thesedimentaryrecord.scholasticahq.com/article/147894);
  [tslearn dtw_barycenter_averaging](https://tslearn.readthedocs.io/en/latest/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html)).

**Buildable from our data? YES** — all are GR-only transforms. **Adversarial check:** these are *re-parameterized
similarity measures*, highly correlated with our existing multi-scale NCC/DTW. Our memory is blunt that the
"averaging axis is SPENT" and a 14-config beam ensemble added +0.015; an orthogonal output-blend hunt is done
([[output-blend-gated-by-orthogonality]], plan.md). A CWT-detail-band correlation is a *new orthogonal feature*
(scale-isolated texture vs. broadband amplitude), so it's worth one cheap shot as a **stack feature**, but the
prior that it's redundant is strong. **Expected ≤0.1 ft. Low rank.**

**Multi-well joint correlation (RGT).** Distinct from the above and more interesting: the **Wheeler–Hale
relative-geologic-time** framework (Sylvester 2023) turns *pairwise* DTW correlations into a **globally
consistent** set of depth shifts via a least-squares (conjugate-gradient) optimization, fixing the
"errors accumulate along a path, loops can't close" failure of pairwise DTW
(["Automated multi-well stratigraphic correlation and model building using relative geologic time", Basin
Research, Wiley bre.12787](https://onlinelibrary.wiley.com/doi/full/10.1111/bre.12787);
[ResearchGate 371641450](https://www.researchgate.net/publication/371641450)). For us, every horizontal well
shares the *same* typewell-defined stratigraphy, so a **global least-squares reconciliation of all per-well
GR↔typewell warps** could regularize the noisy single-well alignments toward a consistent stratigraphic frame.
**Buildable? Yes.** **Difficulty: high.** **Expected: speculative 0.1–0.3 ft.** Rank above the texture features
because it adds *cross-well consistency*, which is the recurring theme of what we lack.

---

## 5. Method vs. data — the brutal assessment

**Facts on the table:**
- Leaders at **5.99, publishing nothing**; best public ~7.78; board jumped ~9 → sub-7 quickly
  ([[lb-board-moved-sub7]], plan.md LB landscape).
- We confirmed **no resistivity / no azimuthal channel** ([[no-resistivity-channel-in-data]]) — so the
  leaders are *not* inverting a log we lack. Whatever they have, it's derivable from the same MD/X/Y/Z/GR/typewell.
- Our PF blend lever transferred *favorably* (LB beat OOF by ~0.9 ft) — the hidden set is not adversarial to
  the per-well estimator. That argues the remaining gap is **not** "our methods overfit," but "there is signal
  we are not extracting."
- There is an explicit **leakage discussion** and a **"Leakage-Aware Submission Pipeline"** notebook in the
  competition
  ([Kaggle discussion/699853](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853);
  [pilkwang leakage notebook](https://www.kaggle.com/code/pilkwang/12-049-rogii-eda-leakageriskdiscussion)).

**The case that it's a DATA exploit (not a method):** A 2 ft jump with *zero* public methodology, in a problem
where the public frontier has converged on the same NCC+PF+KNN+GBM recipe, is the classic fingerprint of a
**leak or a split exploit**, not a cleverer signal-processing chain. Candidates: typewell↔well pairing reveals
more than intended; the hidden TVT is reconstructable from a surface/ID that wasn't fully blanked
([[visible-test-wells-are-train]] shows the visible "test" wells are train wells with TVT blanked — a hint the
blanking, not the geology, is the boundary); or public/private split lets a few wells be effectively memorized.
**This is testable and cheap and I'd do it first** (§6.5).

**The case that it's a METHOD (structural):** Every geosteering source above says the same thing — single-typewell
stretch-squeeze (TSP) is the *weak* method and structural/dip-aware (TST, framework update, GR-only continuous
surface model) is the *strong* one. We are doing the weak one well. The 2025 NORCE GR-only continuous-surface
paper existing *at all* proves a GR-only structural method is a real, separate capability from what we built.
So a structural prior plausibly recovers *some* of the gap — but the literature gives no quantitative "+2 ft"
promise, and our own block-holdout history warns the spatial family over-credits.

**My honest verdict:** **~60% the leaders have a data/split exploit we can't see; ~40% a structural-prior method
edge.** The two are not mutually exclusive and the *expected value* ranking is: diagnose leakage first (cheap,
possibly large), then build the dip-trend structural surface (medium, plausibly 0.2–0.5 ft, but gate hard on
block-holdout). I would **not** bet the remaining time on a single geology technique closing 2 ft; I'd bet on
**leakage diagnosis + a gated structural-prior upgrade** together recovering *part* of it, and accept that the
last fraction may be unreachable without seeing the leaders' trick.

---

## 6. Ranked, actionable shortlist

### 6.1 — (Rank 1) Universal / dip-trend kriging of ANCC, replacing plane-fit + dense KNN
- **Method:** Model ANCC(X,Y) = regional dip trend (low-order polynomial or regression on X,Y) + spatially
  correlated residual (kriging with a fitted variogram + anisotropy). This is the *principled* version of our
  plane-fit KNN and explicitly separates **regional dip** from **local structure**
  ([universal kriging / kriging-with-trend, Matheron](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/kriging);
  [formation-top kriging patent US-12460527](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12460527)).
- **Data needed:** ANCC at train wells (have it), X,Y at all wells (have it). Trend model transfers to eval.
- **Buildable from GR+traj+typewell only:** Yes (ANCC train-only, but that's already how we use it).
- **Difficulty:** Medium. **Expected:** 0.2–0.5 ft IF it transfers.
- **⚠️ Gate:** MUST use region/block holdout, not single-well LOO ([[gate-spatial-levers-with-block-holdout]]).
  This is the same family as the GP anchor that regressed (24.50→47.95→LB loss). Build it; trust only the
  block-holdout number.

### 6.2 — (Rank 2) Apparent-dip feature + dip-corrected GR matching
- **Method:** From the per-well GR↔typewell warp, estimate the **apparent dip along the lateral** (the
  stretch/squeeze rate is a dip proxy via `tan β2 = cos α · tan β1` and the TST relation). Feed (a) apparent
  dip and its along-well gradient as **new GBM features**, and (b) optionally constrain the PF/DTW warp to be
  consistent with a slowly-varying dip ([Geosteering Using TST, URTEC-1590259](https://resdip.com/download/geosteeringUsingTST.pdf)).
- **Data needed:** GR, trajectory (X,Y,Z, azimuth α), typewell. All present.
- **Buildable:** Yes. **Difficulty:** Medium. **Expected:** 0.1–0.3 ft.
- **Why it might help:** Directly addresses the TSP→TST gap; dip gradient flags where a constant-b_well model
  breaks. **Adversarial caveat:** if ANCC+b_well already absorbs smooth dip (it largely does), the marginal
  signal is the *dip change* along the well, which is smaller.

### 6.3 — (Rank 3) RGT / global least-squares reconciliation of per-well warps
- **Method:** Wheeler–Hale RGT: turn all per-well GR↔typewell DTW alignments into one globally consistent set
  of depth shifts via least-squares/conjugate-gradient, removing path-accumulated and loop-closure errors
  ([Basin Research bre.12787](https://onlinelibrary.wiley.com/doi/full/10.1111/bre.12787)).
- **Data:** All wells' GR + shared typewell stratigraphy. Present. **Difficulty:** High. **Expected:** 0.1–0.3 ft.
- **Adversarial caveat:** benefit is cross-well *consistency*; if wells are stratigraphically heterogeneous it
  helps less. Higher build cost than payoff certainty.

### 6.4 — (Rank 4) CWT-detail-band / DBA shape features as new stack inputs
- **Method:** CWT/DWT detail-coefficient correlation vs. typewell at isolated scales; DFA exponent; DBA-robust
  template ([CWT+DTW S0926985118304336](https://www.sciencedirect.com/science/article/abs/pii/S0926985118304336);
  [DTW+DBA Gotland](https://thesedimentaryrecord.scholasticahq.com/article/147894)).
- **Data:** GR + typewell. **Difficulty:** Low–Med. **Expected:** ≤0.1 ft (likely redundant with NCC/DTW per
  our spent-averaging-axis memory). One cheap shot as a feature; don't over-invest.

### 6.5 — (Rank 5) Leakage / split diagnosis — DO THIS FIRST despite low rank
- **Method:** Read [discussion/699853](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853)
  and the [leakage notebook](https://www.kaggle.com/code/pilkwang/12-049-rogii-eda-leakageriskdiscussion).
  Audit: is any hidden TVT reconstructable from typewell↔well IDs, surface fields, or incompletely-blanked
  columns ([[visible-test-wells-are-train]])? Probe public/private split sizes.
- **Cost:** Low. **Expected payoff:** unknown, possibly the whole gap, possibly zero.
- **Rationale:** A sub-7 board with nothing published is the signature of an exploit. Cheapest high-variance bet.

---

## 7. The one question to sit with

We have invested heavily in making the *per-well* estimator excellent (PF blend, beam ensemble, NCC scales) and
our own logs say those axes are "spent." Every geosteering source says the leverage that single-typewell
correlation *leaves on the table* is **cross-well structural consistency** — yet our two spatial attempts (GP
anchor, density-coupled KNN) regressed on the hidden set. **Before building a third spatial estimator: is the
2 ft gap a structural-prior we keep failing to make transfer, or is it a data exploit that makes structure
irrelevant — and have we actually read the competition's own leakage thread to tell the difference?**

---

## Appendix — full source list

- ROGII StarSteer: [Stochastic Inversion](https://www.rogii.com/blog/stochastic-inversion) ·
  [UDAR inversion](https://www.rogii.com/blog/rogii-advances-reservoir-mapping-with-ultra-deep-azimuthal-resistivity-data-inversion) ·
  [StarSteer product](https://www.rogii.com/products/starsteer) ·
  [Geosteering solutions](https://www.rogii.com/solutions/geosteering)
- Azimuthal-GR inversion: [novel quantitative geosteering w/ azimuthal GR](https://www.sciencedirect.com/science/article/abs/pii/S096980431400400X) ·
  [fast forward algorithm](https://www.sciencedirect.com/science/article/abs/pii/S0969804317302580) ·
  [Modeling & Inversion w/ Azimuthal GR](https://www.researchgate.net/publication/346223913)
- TST/TVT/dip: [Geosteering Using TST, URTEC-1590259](https://resdip.com/download/geosteeringUsingTST.pdf) ·
  [Calculating TST](https://resdip.com/docs/calculating%20TST.pdf) ·
  [AAPG Depth & thickness conversion](https://wiki.aapg.org/Depth_and_thickness_conversion) ·
  [Holt & Schoonover SPWLA 1977](https://onepetro.org/SPWLAALS/proceedings/SPWLA-1977/SPWLA-1977/SPWLA-1977-Y/19871)
- Structural framework: [Horizontal-Well Correlation, Saudi Arabia (JPT)](https://jpt.spe.org/horizontal-well-correlation-geosteering-complex-reservoirs-saudi-arabia) ·
  [3D geocellular models / trend surfaces](https://www.sciencedirect.com/science/article/abs/pii/S2949891024003610) ·
  [Continuous Surface Model Updates Using Gamma Log, SPE-227995-MS (NORCE)](https://onepetro.org/SPEATCE/proceedings/25ATCE/25ATCE/D021S015R008/792004)
- Bayesian/EnKF/RL: [Decision Analytic Approach, SPE-167433-PA](https://www.onepetro.org/journal-paper/SPE-167433-PA) ·
  [Value of Look-Ahead, SPE-184392-PA](https://www.onepetro.org/journal-paper/SPE-184392-PA) ·
  [Ensemble well-log interpretation, arXiv 2103.05384](https://arxiv.org/pdf/2103.05384) ·
  [RL+PF geosteering, arXiv 2402.06377](https://arxiv.org/abs/2402.06377) ·
  [DISTINGUISH, arXiv 2503.08509](https://arxiv.org/abs/2503.08509) ·
  [Bayesian network geosteering](https://hughw.net/geosteering-bn/geosteering-by-bayesian-network.pdf)
- Well-log correlation / shape: [CWT+DTW correlation](https://www.sciencedirect.com/science/article/abs/pii/S0926985118304336) ·
  [DWT boundary detection](https://link.springer.com/article/10.1007/s40948-016-0027-1) ·
  [DFA facies](https://www.sciencedirect.com/science/article/abs/pii/S0378437113006833) ·
  [RGT multi-well, Basin Research bre.12787](https://onlinelibrary.wiley.com/doi/full/10.1111/bre.12787) ·
  [DTW+DBA Gotland](https://thesedimentaryrecord.scholasticahq.com/article/147894)
- Spatial interpolation: [universal kriging overview](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/kriging) ·
  [formation-top kriging patent US-12460527](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12460527)
- Competition: [ROGII competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) ·
  [leakage discussion 699853](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853) ·
  [leakage-aware notebook](https://www.kaggle.com/code/pilkwang/12-049-rogii-eda-leakageriskdiscussion) ·
  [romantamrazov better solution LB 9.956](https://www.kaggle.com/code/romantamrazov/rogii-better-solution-lb-9-956)
