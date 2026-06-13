# Where We Are — ROGII Wellbore Geology Prediction (2026-06-12)

A plain-language status report. No oil-drilling knowledge assumed. Basic machine-learning
knowledge assumed (you know what a model, a feature, training, and cross-validation are).

> This updates the 2026-06-06 report (`Summary_060626.md`). The headlines since then: **we went
> from 8.269 to 8.065 ft**, but the path there was anything but smooth — two of the six submissions
> in between were *regressions*, and the wins came from a different place than the losses. This
> document explains the full model as it stands today, and the hard-won lesson about *which kinds
> of improvements survive contact with the hidden test set* — arguably the most valuable thing we
> now know.

---

## 1. What the competition asks (unchanged — skip if you read the last report)

Companies drill **horizontal wells**: straight down, then sideways for thousands of feet through
one specific layer of rock. Picture the rock as a layer cake and the drill threading the sideways
part through one layer of frosting without wandering out of it.

**TVT** ("true vertical thickness") is "how deep into the rock layer are we, vertically, at this
point" — measured in feet. We know TVT for the early ("known") part of each well; the competition
blanks it after a cutoff point and asks us to **predict** it for the rest of the well (typically
~4,800 points per well).

For each well we get: the drill's 3D **path** (X, Y, depth at every ~1-foot step); a **GR**
(gamma-ray) sensor reading at every step — rock layers leave a characteristic GR "signature"; the
**known TVT** up to the cutoff; and a **typewell** — a nearby vertical reference well with a full
GR-vs-depth profile, i.e. the "answer key" for what GR should look like at each depth in that area.

We learn from ~773 wells and are scored on a hidden set of ~200 wells we never see — we submit
*code* that Kaggle runs on them.

**Scoring:** RMSE in feet (lower is better). Anchors: **15.9 ft** = the dumb "TVT never changes"
baseline; **~6.0 ft** = the public leaderboard top.

---

## 2. Where we stand right now

| | RMSE (ft) | Meaning |
|---|---|---|
| Dumb baseline | 15.9 | do nothing |
| Best at the last report | 8.269 | particle-filter blend, weight 0.44 |
| **Our best submitted model (NEW)** | **8.065** | as of 2026-06-12 |
| Best *published* notebook | ~7.53 | moved a lot — see §8 |
| Public leaderboard top | ~6.0 | the actual leaders (still unpublished) |

We have now closed about **79%** of the distance from "do nothing" to "the leaders." Since the last
report we made six submissions: three wins (8.158, 8.131, 8.065), one near-tie probe, and two
genuine regressions (8.218, 8.220) that taught us more than the wins did.

---

## 3. The model in full, end to end

This section is the detailed "how it works." Everything else in the report builds on it. The final
answer for every point in a hidden well is produced in four stages.

### Stage 1 — Two engines produce independent TVT estimates

**Engine 1: a gradient-boosted tree stack on 231 hand-built features.**
Six tree models (LightGBM with three different random seeds + CatBoost with three seeds) each
predict, for every point, the *drift* — how far TVT has moved from the last known value. Predicting
the *change* rather than the absolute depth matters: absolute TVT is ~11,000–12,000 ft while the
signal we care about is ±16 ft, so a model trained on absolute values drowns. The six models'
outputs are combined with a small non-negative linear blend fit on out-of-fold predictions
(weights learned once, frozen into the submission).

The 231 input features encode two big physical ideas:

- *The geometry trick.* Within a well there is a near-exact identity:
  `TVT = −(depth) + (height of a reference rock surface at this map location) + (a per-well offset)`.
  The surface height ("ANCC") is known only for training wells, so for a hidden well we **spatially
  interpolate** it at the well's (X, Y) from nearby training wells — a plane fit through the ~15
  nearest wells per rock layer, plus a denser row-level nearest-neighbor estimate. The per-well
  offset is calibrated from the well's own known prefix. This family alone takes the error from
  15.9 to ~12.
- *The GR "barcode".* The GR trace is a barcode for the rock; the typewell is the reference
  barcode. Sliding the drill's GR against the typewell's and finding where they line up reveals
  TVT. We compute this alignment several independent ways — multi-scale cross-correlation, seven
  configurations of beam search (a dynamic-programming alignment), dynamic time warping (two
  variants), and small particle filters — and feed every estimate, plus *disagreements between
  estimators* (a built-in confidence signal), to the trees.

The rest of the features are trajectory geometry (slopes, curvature), GR statistics (rolling
windows, missing-data flags — GR is missing for 28% of points on average), prefix-quality measures
(how well the known part of the well fits its typewell), and position-along-the-well measures.

Engine 1's accuracy alone: **~10.3 ft** on our cross-validation.

**Engine 2: a 128-run likelihood-weighted particle filter.**
A particle filter is a probabilistic *tracker*: a cloud of 500 guesses for the current TVT is moved
forward step by step as the drill advances, and at every step each guess is re-weighted by how well
its predicted GR (read off the typewell barcode at that guess's depth) matches the actual GR
reading. Good guesses survive, bad ones die. One run of this tracker is mediocre — the rock barcode
has repeating patterns, and a single tracker can confidently lock onto the *wrong repeat*. So we
run it **128 times from different random starts** and combine the runs weighted by each run's
*overall* GR match quality (a softmax over accumulated log-likelihoods). Wrong-lock runs match the
barcode poorly overall and get voted down.

Engine 2's accuracy alone: **~11.0 ft** — *worse* than Engine 1, and that's fine (next stage).

### Stage 2 — Blend the engines

The final answer is a weighted average: `final = (1 − w) · Engine1 + w · Engine2`. This works far
better than either engine because their errors are only ~48% correlated — they make *different
mistakes*, which partially cancel. Blending the 10.3 and the 11.0 engine yields **~9.2** on
cross-validation, and transferred to ~8.2 on the leaderboard. (The general lesson, from the last
report, still stands: judge a candidate by *the blend it produces*, never by whether it beats your
current model on its own.)

### Stage 3 — Set the blend weight by *measuring the leaderboard*, not trusting cross-validation

This is subtle and important. Our cross-validation says the best weight is w ≈ 0.44. But the
leaderboard disagrees: we submitted four different weights (0.44, 0.57, 0.60, 0.77) and the scores
traced a clean parabola — 8.269, 8.158, 8.164, 8.429 — whose minimum sits at **w ≈ 0.57**, not
0.44. The hidden wells reward the particle filter *more* than our training wells do. Two takeaways
we now treat as law:

- The fitted parabola predicted the fourth point to within **0.001 ft**, which tells us the public
  leaderboard is a *low-noise instrument* (~±0.005 ft). Differences bigger than ~0.01 ft are real.
- Where cross-validation and the leaderboard disagree about a *knob*, the leaderboard wins, and we
  pin the knob to the leaderboard-measured value (0.57).

### Stage 4 (NEW, the 8.131 → 8.065 step) — A *per-well* blend weight

Until this week, every well got the same w = 0.57. But wells differ: in some, the particle filter
is clearly trustworthy (clean GR, good typewell match, its 128 runs agree); in others it is
visibly struggling (its runs disagree with each other — a self-reported confidence signal). So we
let the weight vary per well:

`w_well = clip( 0.57 + θ · z-scored(9 per-well summary statistics), 0.45, 0.70 )`

The nine statistics are things we can compute for any well at prediction time: how long the
unknown section is, the vertical span it covers, how well the known prefix fits the typewell, how
much our alignment estimators disagree on this well, how far away the nearest training wells are,
and — most informative — two measures of the particle filter's *internal disagreement* across its
own 128 runs. The coefficient vector θ is a tiny ridge regression (nine numbers) fit on training
wells.

The design constraints are not decoration; each one blocks a failure mode we measured:

- **No intercept, statistics centered** — so the *average* weight stays pinned at the
  leaderboard-measured 0.57. An unconstrained fit drifts the average toward cross-validation's
  preferred 0.44, which we *know* scores worse (8.269). We reproduced this trap deliberately: with
  an intercept the fit "improves" cross-validation by 0.13 ft — a fake gain we declined.
- **Very heavy regularization** (the gain must survive being squeezed toward "no adjustment") and a
  **hard clip to [0.45, 0.70]** — bounding the worst case if the hidden wells' statistics are
  distributed differently from training.
- **A decomposition check before believing it**: we split the measured gain into "part explained by
  a small shift of the average weight" (≈ harmless, the parabola is flat there) and "part that is
  genuinely per-well matching." Three-quarters was genuine.

Cross-validation said this is worth −0.084 ft. The leaderboard delivered **−0.066 ft (8.131 →
8.065)** — about 80% of the promise, 13× the noise floor. It is the first per-well adjustment of
any kind that has ever survived the hidden set in this project.

One engineering rule carried through all of this: the particle filter uses randomness, and we seed
it **deterministically per well**, so the submitted code reproduces our measured numbers
byte-for-byte on Kaggle's machines (verified: the Kaggle rerun matches our local run to a mean of
0.00002 ft). Skipping this once cost us ~1.2 ft, early in the project.

---

## 4. The expensive lesson: which improvements transfer, and which don't

Between the last report and this one we also tried the *other* obvious direction: make Engine 1
smarter by adding new features. Four attempts, all of which looked good locally:

| Attempt | Local (CV) said | Leaderboard said |
|---|---|---|
| Kriging-based surface feature ("UK") | −0.02 improvement | **+0.060 regression** (8.218) |
| 12 new features jointly (UK + dip + texture) | −0.10 improvement | **+0.013 regression** (8.171) |
| Same minus UK (dip + texture only) | −0.03 improvement | **−0.027 win** (8.131) ✅ |
| 28-feature "super" union on top of that | −0.09 improvement | **+0.089 regression** (8.220) |

Three of four *inverted*: solid, properly cross-validated gains turned into losses on the hidden
set. The fourth (the lone win) looked just like the others locally. After the last failure we
closed the entire "add features to the trees" axis — not because features are bad, but because **we
possess no local measurement that can tell a transferable feature gain from a mirage**, and each
test costs a submission.

Contrast that with the things that *did* transfer, all of them: the output blend (−1.85), the
weight vertex (−0.11 cumulative), and now the per-well weights (−0.066). The pattern is stark:

> **Changes at the *output* level — how we combine finished predictions — transfer reliably.
> Changes at the *feature* level — what we feed the trees — do not.** Plausibly because output
> adjustments are few-parameter and physically interpretable, while a tree ensemble given new
> features finds subtle dataset-specific correlations that the hidden wells don't share.

We also adopted a process rule worth stating because it saved us twice this week: **every
submission gets a pre-registered verdict** — before the score arrives, we write down exactly which
score ranges mean "bank it," "tie, ignore," and "regression, revert" — so a disappointing number
can't be rationalized after the fact. Both regressions above were reverted within hours, per their
pre-registered branches, with no agonizing.

Two cheap offline ideas were also tested and killed this week without spending submissions: a
"fade-in" that damps predictions right after the cutoff (the strong public notebooks ship it, but
it only acts on the first ~85 ft after the cutoff — **1.7%** of our points, since our unknown
sections are thousands of feet long — measured effect: zero), and output smoothing (also ~zero).

---

## 5. Why still not deep learning?

Same answer as the last two reports, still well-supported: with only ~773 wells, sequence models
(RNNs / transformers) memorize each well's GR fingerprint and fail on new wells — public attempts
score ~14–15 ft, barely better than the baseline. Trees on physics-shaped features, blended with a
sequential tracker, remain the winning class of solution — including for everyone above us on the
board, as far as anything published shows.

---

## 6. The outside world moved (a lot)

Three external events this week, all confirmed from primary sources:

1. **The best published recipe is now ~7.53 ft** (it was ~8.2 at the last report). A public
   notebook ("fle3n v5") blends two engines: the best published pipeline we hadn't reproduced
   ("ridge-SP", 7.776 standalone) and — validating our architecture — a pipeline that is
   essentially *our* model (128-run particle filter + tree stack), at 7.810 standalone. Their
   0.55/0.45 blend scores 7.540. The code and even pre-trained models are public.
2. **The competition host re-issued the hidden answers.** Some hidden wells were byte-identical
   copies of training wells, which allowed a "look up the answer" exploit (we tested it earlier;
   it was worthless on the public set then). Mid-week the organizers *re-interpreted* the
   duplicated wells' hidden labels — the same prediction file scored differently before and after
   — killing the exploit even where the duplication exists. We verified our own numbers are clean
   across this change: resubmitting our banked model reproduced **8.131 exactly**, post-change.
3. **Our own earlier judgment got overturned by evidence.** We had evaluated the "ridge-SP" family
   weeks ago on cross-validation, found it redundant against our stack, and shelved it. Its 7.776
   leaderboard score says that judgment was another local-measurement false negative — consistent
   with §4's lesson, and a reminder that on this competition, *reproduce the winner wholesale, then
   measure* beats *pre-judge the parts locally*.

---

## 7. The honest picture and what's next

- We are at **8.065**, our sixth banked best, achieved almost entirely through output-level
  engineering: a blend of two diverse engines, with a leaderboard-calibrated global weight, now
  adjusted per well by a tiny, heavily-constrained regression.
- The best published recipe (~7.53) is now **0.54 ft ahead of us** — the first time since we caught
  the public frontier that it has pulled away. But its weaker half is a version of *our* pipeline,
  and its authors state the blend is capped without "a decorrelated third source."
- The leaders (~6.0) have still published nothing.

The queued plan, in order:

- **(a) Reproduce the 7.53 published blend wholesale** — fork it as-is, run it, submit it, bank its
  number. No local pre-judging of its parts (that's how we false-negatived it last time).
- **(b) Then offer our 8.065 stack as the "decorrelated third member"** of that blend. Our engine
  is provably different machinery from their ridge-SP half; if the error correlations cooperate,
  the three-way blend is the most credible path under 7.5 available from public materials.
- **(c) Keep the final-submission discipline**: two slots, currently 8.065 + 8.131, re-decided only
  on pre-registered evidence.

Where we are, stated plainly: **a well-engineered 8.07 built on one durable insight — on this
problem, gains compound at the level where finished predictions are combined, and die at the level
where features are added — now setting out to merge with, rather than chase, the recipe that just
passed us.**
