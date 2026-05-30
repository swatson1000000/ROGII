# Plan Summary — ROGII Wellbore Geology Prediction
### A from-scratch walkthrough for someone new to this kind of project

This document explains the plan in `plan.md` in plain language. It assumes you have **not** done
a Kaggle competition before and have **not** worked with oil-and-gas / geology data. Every term is
explained the first time it appears, and there's a glossary at the end. If you just want the
terse, reference version, read `plan.md`. If you want the "explain it to me like I'm new," read
this.

---

## 1. What is this competition actually about?

Oil and gas companies drill wells that start vertical and then bend to run **horizontally** for a
mile or more. They want the drill bit to stay inside one thin, valuable layer of rock (think of a
single sheet of a layer cake that's only a few feet thick). That target layer isn't flat — it
tilts and waves up and down underground. Keeping the bit inside it as you drill is called
**geosteering**.

Today a human geologist does this steering live, by watching sensor readings from the drill bit
and comparing them to a known reference. **This competition asks you to automate that judgment with
a model.** Concretely: figure out the bit's *vertical position within the rock layers* at every
point along the horizontal part of each well.

That vertical position is called **TVT** (True Vertical Thickness). Predicting TVT well = knowing
where in the layer cake the bit is = successful geosteering.

---

## 2. The data you're given

You get a folder of **wells**. Each well has two CSV files (plus a picture for training wells):

### a) The horizontal well file — the well you're steering
One row per point along the drilled path (~1 row per foot). Columns:

| Column | What it means (plain English) |
|--------|-------------------------------|
| `MD` | **Measured Depth** — how far along the borehole you are (the length of pipe in the ground). Think "distance traveled by the drill." |
| `X`, `Y`, `Z` | The 3-D location of that point (east, north, and depth). The bit's GPS coordinates. |
| `GR` | **Gamma Ray** — a radioactivity sensor reading. Different rock types glow differently (clay/shale = high, limestone = low). This is the key "what rock am I in right now" signal. |
| `TVT` | **The answer** — the bit's vertical position in the rock layers. *Only present in training wells.* |
| `TVT_input` | The TVT that's already *known* for the early part of the well, then blank. (More on this in §3.) |
| `ANCC`, `ASTNU`, … | Depths of specific geological layer boundaries ("formation tops"). *Training wells only* — you don't get these for the wells you're scored on. |

### b) The typewell file — a vertical reference well nearby
Before drilling horizontally, companies drill a straight-down "pilot" well close by, called a
**typewell**. Because it goes straight down, you know exactly which rock is at which depth. It
gives you:

| Column | Meaning |
|--------|---------|
| `TVT` | vertical position (depth in the reference well) |
| `GR` | the gamma-ray reading at that depth |
| `Geology` | the name of the rock layer at that depth (training only) |

**Why the typewell matters:** it's a "key." It tells you *"a gamma-ray reading of X means you're
at vertical position Y."* If you can match the horizontal well's live gamma-ray readings to this
key, you can read off where the bit is.

### c) The picture (`.png`)
A visual of each training well — useful for human intuition, not used by the model.

---

## 3. The "prediction start" — what exactly you predict

Each horizontal well has a known beginning. For the first stretch of the well, the geologist has
already worked out the TVT, and it's handed to you in the `TVT_input` column. At some point —
the **Prediction Start (PS)** — `TVT_input` goes blank. **Your job is to predict TVT for every row
after the PS point.**

Picture it like a hiking trail where the first part is marked with your exact altitude, then the
markers stop and you have to estimate your altitude for the rest of the trail using your compass
(trajectory) and what you see around you (gamma ray vs. the typewell key).

**The submission:** a file with two columns, `id` and `tvt`. The `id` is `{well}_{rownumber}` and
there's one row for every point you must predict.

**The score:** **RMSE** — Root Mean Squared Error. In plain terms: take how far off you are at each
point (in feet), square it, average over all points, take the square root. **Lower is better, and
it's measured in feet.** Squaring means big misses hurt much more than small ones, so you care a
lot about avoiding large errors.

**Likely a "code competition":** instead of just uploading a predictions file, you probably upload
a *notebook* that Kaggle runs on a hidden set of ~200 wells you never see. (We still need to
confirm this on the site — it's a "TODO" in the plan.)

---

## 4. The single most important idea: predict the *change*, not the absolute number

TVT values are huge — around 11,000–12,000 feet. But from where the markers stop (PS) to the end
of the well, the bit only moves up or down by a small amount — typically under ~25 feet, rarely
more than ~70.

If you ask a model to predict the raw number (11,500-ish), it wastes all its effort guessing the
big baseline and has nothing left for the small, important wiggle. In fact, teams who tried this
scored **19.5** — *worse* than doing nothing.

So the trick is: **predict the change from the last known value.** We call this the **drift**:

```
drift = TVT − (last known TVT_input before PS)
final prediction = last_known_TVT + predicted_drift
```

This one reframing is the biggest lever in the whole competition. Two reference points:
- **Do-nothing baseline** (just guess the last known value everywhere): **~16 feet** RMSE. This is
  surprisingly hard to beat because most wells don't drift far.
- **Best public solutions:** **~9.25–9.4 feet**. That's the target.

So the entire game is: explain the ~16-foot drift well enough to cut the error roughly in half.

---

## 5. Three independent ways to locate the bit (the "signals")

There's no single magic feature. The strong solutions compute several *independent estimates* of
where the bit is, then combine them. Here are the three main ideas, in plain terms.

### Signal 1 — Match the gamma-ray "barcode" (this is the core)
The gamma-ray trace as you drill is like a **barcode** of the rock layers. The typewell has the
same barcode laid out against known depths. So you slide a short window of the horizontal well's
gamma-ray pattern along the typewell's pattern and find where it matches best — that match tells
you the bit's vertical position.

The matching is done with **NCC (Normalized Cross-Correlation)**, which measures *pattern shape*
similarity while ignoring overall brightness/scale (so it still works if one well's sensor reads a
bit higher than another's). Done at a few window sizes and blended, this match correlates with the
true answer at ~0.999 — extremely tight. **This is the workhorse.**

> Caveat we found in the data: the gamma-ray reading is **missing ~28% of the time** (up to 73% in
> some wells). So this signal needs help where GR is blank — which is where Signal 2 comes in.

### Signal 2 — Use the neighbors' geology (a geometric shortcut)
We discovered (and verified on the real data) an almost *exact* relationship inside every well:

```
TVT ≈ −Z + ANCC + (a per-well constant)
```

where `ANCC` is the depth of a particular rock-layer boundary. The correlation is essentially
perfect (1.0000; error ~0.007 feet). So **if you knew ANCC at the bit's location, you'd basically
know TVT.**

The catch: `ANCC` isn't given for the wells you're scored on. The fix uses the fact that geology
changes smoothly across space — **neighboring wells share the same tilting layers.** So you take
the ~15 nearest wells, fit a gently-tilted plane through their known layer depths, and read off the
estimated `ANCC` at your bit's (X, Y) location. This is **KNN** (k-nearest-neighbors) plus a plane
fit. It's a completely different kind of evidence from the gamma-ray barcode, which is exactly why
combining the two helps.

### Signal 3 — Track the bit step by step (sequential estimators)
Instead of matching each point independently, you can *track* the bit as it moves, the way a GPS
fuses "where I probably am" with "I just moved this far." Two classic tools do this:
- **Particle filter:** keep ~500 random guesses ("particles") of the bit's position, nudge them
  forward by the motion, then up-weight the ones whose expected gamma-ray matches what's actually
  measured. The weighted average is your estimate.
- **Beam search / Viterbi:** find the smoothest path through the typewell barcode that best
  explains the whole gamma-ray sequence at once.

On their own these are noisy, but they give *directional* hints ("the bit is trending up here")
that complement the other two signals.

**The point of having three:** they make *different kinds of mistakes*. When you combine
independent estimates that fail in different ways, the combination is more accurate and more robust
than any one alone.

---

## 6. Combining the clues: the "stacker" model

Now you have, for every point you must predict, several TVT estimates plus lots of supporting
numbers (how much they disagree, gamma-ray statistics, how far past PS you are, the trajectory
direction, etc.) — on the order of 100+ **features** (input numbers).

A **GBM (Gradient-Boosted Decision Trees)** — using the libraries **LightGBM**, **XGBoost**, and
**CatBoost** — learns from the training wells how to turn all those clues into the best single
**drift** prediction. You can think of a GBM as a large committee of simple yes/no rules that, in
combination, learn things like "when the barcode match and the geometry estimate agree, trust them;
when they disagree and gamma-ray is missing, lean on the geometry." GBMs are the go-to tool for
this kind of mixed-signal tabular problem.

We train **three different GBM libraries** because they make slightly different errors, then
**blend** their outputs. Blending = taking a weighted average of several models' predictions. We
use a simple, overfitting-resistant blend (called **NNLS** or **hill-climbing**) rather than
anything fancy, because the models are very similar and fancy combining would just memorize noise.

> Note: we feed the *signals* (Section 5) into the GBM as features. The signals do the heavy
> physics; the GBM learns how much to trust each one in each situation.

---

## 7. Final polish: post-processing

After the model predicts, two cheap clean-ups squeeze out the last bit of error:
- **Shrink and fade:** scale the predicted drift slightly toward zero, and ramp it up gradually
  right after PS (we're most confident near the known section). A tuner called **Optuna**
  automatically finds the best settings.
- **Smoothing:** the true TVT path is physically smooth, so we run a gentle smoother
  (**Savitzky-Golay**) over each well's predictions to remove jitter without erasing real bends.

Worth ~0.1–0.15 feet — small, but it can move you up the leaderboard.

---

## 8. How do we know we're improving? (Validation — the make-or-break part)

You cannot just submit constantly and trust the leaderboard score — that way you end up
accidentally tuning to the specific test wells and fooling yourself. Instead you build a **local
score** that mirrors the real one, using the training data where you *know* the answers.

The crucial rule here: **split by well, never by row.** Each well has a unique gamma-ray
fingerprint. If rows from the same well end up in both the "learn from" and "test on" piles, the
model can cheat by memorizing that well's fingerprint, and your local score looks great but the
real score is bad. The fix is **GroupKFold by well**: divide the *wells* into 5 groups, and always
test on wells the model never trained on. This mimics the real task (unseen wells) honestly.

We also:
- score **only the rows after PS** (the ones that count), exactly like the real metric;
- run an **adversarial-validation** check — a quick test of whether the hidden test wells look
  statistically different from training wells, so we're not blindsided;
- **trust this local score over the public leaderboard** when deciding what to submit. (Observed
  gap: local ~10.0 tends to become ~9.4 on the real board — stable and predictable.)

---

## 9. The build order — and the score we expect at each step

We build in phases, each with a target score to beat before moving on. This keeps us honest and
makes it obvious when a new idea actually helps.

| Phase | What we add | Expected RMSE |
|-------|-------------|---------------|
| 0 | Plumbing: load data, build the local scorer, reproduce the do-nothing baseline | ~16 |
| 1 | Drift target + basic features + one GBM | ~14–15 |
| 2 | Gamma-ray barcode matching (NCC) — the core signal | ~12 |
| 3 | Neighbor-geology geometry (formation plane-fit KNN) | ~11 |
| 4 | Sequential trackers (particle filter, beam search) | ~10 |
| 5 | Full feature set + 3-library GBM ensemble + blend → **first real submission** | ~10 local / ~9.4 board |
| 6 | Post-processing (shrink/fade + smoothing) | ~9.25–9.4 |
| 7 | Stretch: extra alignment variants, more diversity | ≤ 9.25 |

Each phase is one or a few scripts; we log every run and record the score in `plan.md`'s
experiment log.

---

## 10. Things that look smart but don't work (learn from others' mistakes)

- **Predicting absolute TVT instead of drift:** scored 19.5 (worse than doing nothing). The huge
  baseline drowns the signal.
- **Neural sequence models (LSTM / TCN / 1-D CNN):** the "obvious" choice for sequential data, but
  they scored ~14.6 and didn't generalize — with only 773 training wells, they just memorize each
  well's fingerprint. *(This is notable: it's tempting to reach for a fancy neural net here, and it
  loses to the signal-matching + GBM approach.)*
- **Plain DTW (another alignment method) as a core signal:** it's sensitive to brightness/scale in
  a way that conflicts with NCC and can make things worse. Use only as optional extra diversity.
- **Elaborate multi-layer model stacking:** the base models are too similar; fancy stacking
  overfits. Keep the blend simple.

---

## 11. How the work actually runs (practical workflow)

- Heavy number-crunching to *compute the signals* (particle filter, barcode matching, neighbor
  KNN) is CPU-intensive and runs for a while over all 773 wells. We do this on the **local machine
  (skynet)** and **cache the results** to disk so we don't redo them.
- Training the GBMs is faster on a **GPU**, so we send that to the remote machine **deepthought**.
- The final submission is a self-contained Kaggle notebook that loads our pre-computed models and
  produces predictions for the hidden wells with no internet access.

(Full machine details and commands are in `CLAUDE.md`.)

---

## 12. Glossary

- **Geosteering** — steering a drill bit to stay inside a thin target rock layer as you drill.
- **TVT (True Vertical Thickness)** — the bit's vertical position within the rock layers; **the
  thing we predict**. Measured in feet.
- **MD (Measured Depth)** — distance traveled along the borehole.
- **GR (Gamma Ray)** — a rock-type sensor reading; the "barcode" signal. Often missing (~28%).
- **Typewell** — a nearby straight-down reference well that maps GR to known depth; the "key."
- **Prediction Start (PS)** — the point in each well after which TVT is unknown and must be
  predicted.
- **Drift** — `TVT − last_known_TVT`; the small change we actually model.
- **RMSE** — Root Mean Squared Error; the score (in feet), lower is better, big misses punished hard.
- **Feature** — an input number the model uses to make a prediction.
- **NCC (Normalized Cross-Correlation)** — a way to match the *shape* of two signals regardless of
  scale; used for gamma-ray barcode matching.
- **KNN (k-Nearest Neighbors)** — using the closest examples (here, nearby wells) to estimate a
  value at a new location.
- **Particle filter / Beam search** — step-by-step "tracking" methods that estimate position by
  fusing motion with observations.
- **GBM / GBDT (Gradient-Boosted Decision Trees)** — the main model type; a committee of simple
  rules. Libraries: LightGBM, XGBoost, CatBoost.
- **Ensemble / Blend** — combining several models' predictions (often a weighted average) for a
  better, steadier result.
- **NNLS (Non-Negative Least Squares) / Hill-climbing** — simple, robust ways to choose blend
  weights.
- **Optuna** — an automatic tuner that searches for the best settings.
- **Savitzky-Golay** — a smoothing filter that removes jitter while keeping real curves.
- **Cross-validation / GroupKFold** — splitting data to get an honest performance estimate;
  "GroupKFold by well" keeps all of a well's rows together so the model can't cheat.
- **OOF (Out-Of-Fold) predictions** — predictions made on data the model didn't train on; the
  basis of an honest local score.
- **Leaderboard (public/private)** — the live ranking (public, on part of the test data) vs. the
  final ranking (private, revealed at the end). Don't over-trust the public one.
- **Adversarial validation** — checking whether train and test data differ, to avoid surprises.

---

*For the precise, reference version of all this — including exact scores from each public notebook
studied, feature lists, model hyperparameters, and open questions — see `plan.md`. For the verified
competition facts (schema, metric, dates), see `SUMMARY.md`.*
