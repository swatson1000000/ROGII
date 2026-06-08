# Where We Are — ROGII Wellbore Geology Prediction (2026-06-06)

A plain-language status report. No oil-drilling knowledge assumed. Basic machine-learning
knowledge assumed (you know what a model, a feature, training, and cross-validation are).

> This updates the 2026-06-04 report (`Summary_060426.md`). The headline since then: **we jumped
> from 10.122 to 8.269 ft** — the single biggest improvement of the project. This document explains
> what we changed and why it worked.

---

## 1. What the competition asks (unchanged — skip if you read the last report)

Companies drill **horizontal wells**: straight down, then sideways for thousands of feet through one
specific layer of rock. Picture the rock as a layer cake and the drill threading the sideways part
through one layer of frosting without wandering out of it.

**TVT** is "how deep into the rock layer are we, vertically, at this point" — measured in feet. We
know TVT for the early ("known") part of each well; the competition blanks it after a cutoff and
asks us to **predict** the rest.

For each well we get: the drill's 3D **path** (X, Y, depth) at every ~1-foot point; a **GR** (gamma
ray) sensor reading at every point (rock type leaves a characteristic GR "signature"); the **known
TVT** up to the cutoff; and a **typewell** — a nearby vertical reference well with a full
GR-vs-depth profile, i.e. the "answer key" for what GR should look like at each depth in that area.

We learn from ~773 wells and are scored on a separate **hidden set of ~200 wells** we never see — we
submit code that runs on them.

**Scoring:** RMSE in feet (lower is better). Anchors: **15.9 ft** = the dumb "TVT never changes"
baseline; **~6.0 ft** = the current public leaderboard top.

---

## 2. Where we stand right now

| | RMSE (ft) | Meaning |
|---|---|---|
| Dumb baseline | 15.9 | do nothing |
| Our *previous* best | 10.122 | the model in the last report |
| **Our best submitted model (NEW)** | **8.269** | as of 2026-06-06 |
| Best *published* notebooks | ~8.2 | the strongest public recipes |
| Public leaderboard top | ~6.0 | the actual leaders (unpublished) |

We have now closed about **77%** of the distance from "do nothing" (15.9) to "the leaders" (6.0),
up from ~60% in the last report. **We have essentially caught up to the best publicly-shared
recipes (~8.2).** The leaders at ~6 ft still have not published how they did it.

Importantly, the **8.269 came in better than our own cross-validation predicted** (our internal
estimate was 9.17). A submission scoring *better* on the real hidden set than in local testing is a
strong, reassuring sign — it means the improvement is genuine and transfers, not an artifact of
tuning against our own data.

---

## 3. The one-sentence version of what changed

**Before:** we computed a "particle filter" estimate of TVT once and handed it to the tree model as
just one input column among 222, letting the trees decide how much to trust it.

**Now:** we run that particle filter **128 times**, combine the runs into one strong estimate, and
**blend it directly into the final answer as a co-equal predictor** — the final number is *56% the
tree model + 44% the particle filter*. Promoting the particle filter from "one input feature" to
"half the answer" is the entire 1.85-ft gain.

To understand why that works, you need two ideas: what the particle filter is, and why blending two
*different* estimators beats either one alone.

---

## 4. Background: the two engines (same as before)

Our model has always had two very different ways of estimating TVT. (Neither is deep learning — see
§7.)

**Engine 1 — the tree model + geometry/barcode features.** This is the "previous best" in full. A
**gradient-boosted tree ensemble** (LightGBM + CatBoost) fed ~222 hand-built features per point. The
intelligence is in the features, which come from two big ideas:

- *The geometry trick:* there's a near-exact relation `TVT = −(depth) + (a formation-surface height)
  + (a per-well offset)`. The surface height ("ANCC") is known for training wells but not hidden
  ones, so we estimate it at a hidden well's map location by **spatially interpolating** from nearby
  training wells. This alone gets ~15.9 → ~12.
- *The GR "barcode":* the GR trace is a barcode for the rock; the typewell is the reference barcode.
  Sliding the drill's GR against the typewell barcode and finding where they line up tells you TVT.
  We do this several ways (cross-correlation, beam search, dynamic time warping, and **particle
  filters**) and feed all of them to the trees.

**Engine 2 — the particle filter (the part we just upgraded).** A particle filter is a probabilistic
**tracker**: think of a GPS that keeps a *cloud of hundreds of guesses* for where TVT is, moves them
forward as the drill advances, and re-weights them at every step by how well each guess's predicted
GR matches the actual GR reading. Guesses that match well survive; bad ones die off. The weighted
average of the surviving cloud is its TVT estimate. It's a fundamentally different mechanism from
the trees — it follows the GR barcode *sequentially* down the well rather than learning rules from
features.

---

## 5. What we actually changed, in three steps

**Step 1 — Run the particle filter 128 times and keep the good runs.** A single particle-filter run
is noisy and, worse, can "lock onto" the wrong part of the barcode (the rock barcode has repeating
patterns, so a tracker can confidently follow the wrong copy). The fix the strong public notebooks
use: run the filter **128 times with different random starting points**, then combine the runs
**weighted by how well each run matched the GR overall** (runs that matched the barcode better get
more say). This "vote, weighted by quality" step is what turns a mediocre tracker into a sharp one.

**Step 2 — Recognize it as a co-equal predictor, not a feature.** This is the conceptual leap and
the part we'd gotten wrong for weeks. We tested whether to **blend** the 128-run particle filter
directly into the final answer, rather than burying it as one of 222 inputs.

**Step 3 — Pick the blend weight by measuring the blend.** We searched for the mix
`final = (1−w)·trees + w·particle_filter` that minimizes error on our cross-validation, and got
**w ≈ 0.44** — i.e. trust the particle filter for ~44% of the final number. Result: our internal
error dropped from **10.36 to 9.17 ft**, which then scored **8.269** on the real leaderboard.

---

## 6. Why this works — and why we almost missed it

Here is the counterintuitive heart of it, worth internalizing because it's a general modeling lesson:

**The 128-run particle filter is, by itself, WORSE than our tree model** — 10.99 ft standalone vs.
the trees' 10.36. So why does mixing in a worse estimator make the answer *much* better?

Because the two engines make **different mistakes.** Their errors are only ~48% correlated. When you
average two estimators of *similar* accuracy whose errors point in *different* directions, the errors
partially cancel — exactly like taking two independent measurements of the same thing and averaging
to get a more precise one. The math of this is exact: with their accuracies and their 48% error
correlation, the optimal blend *should* land at 9.17 ft, and it did. The gain comes from
**diversity (orthogonality), not from the particle filter being good on its own.**

**Why we missed it for so long — the trap:** our usual quick check was to ask "does this new
estimator beat what we already have?" (no — 10.99 > 10.36) and "does adding one cheap version as a
feature help?" (barely — +0.04 ft). *Both checks said "useless."* Both were misleading. The only
test that reveals the truth is to **actually perform the blend with the fully-ensembled version and
measure it** — which gave +1.18 ft. This is the same trap that bit us earlier in the project, when
the GR-matching features each looked useless tested one-at-a-time but were worth +1.4 ft together.

**The lesson, now a permanent rule:** judge a candidate predictor by *the blended result it
produces*, never by whether it beats the current model on its own.

---

## 7. Why this improvement transfers so well (and didn't backfire)

Earlier in the project we had a feature (a "GP/kriging anchor") that looked great in cross-validation
but **regressed badly** on the real leaderboard, because it depended on having training wells
*nearby* — and the hidden wells were often isolated, where it silently fell apart.

The particle filter has the **opposite, reassuring property.** It is computed *per well, from that
well's own GR readings and its own typewell* — it needs **no other wells at all.** So it cannot
degrade on isolated hidden wells the way the geometry trick can. That's why it transferred *even
better* than our cross-validation predicted (8.27 actual vs 9.17 expected): the hidden wells are no
harder for it than the training wells.

(One engineering note carried over from before: the particle filter uses randomness, and we seed it
**deterministically per well** so the code produces byte-for-byte identical results on Kaggle's
hidden machines as on ours. We verified the production version reproduces our measured filter
exactly. Skipping this once cost us ~1.2 ft, so it's now mandatory.)

---

## 8. Why still not deep learning?

Same answer as before, still well-supported: with only ~773 wells, sequence models (RNNs /
transformers) *memorize* each well's GR fingerprint and fail on new wells — they score ~14–15 ft,
barely better than the baseline. Tree ensembles on hand-engineered signals, now blended with the
sequential particle-filter tracker, remain the winning combination.

---

## 9. The honest picture of the gap

- We are now at **8.269**, level with the best *published* notebooks (~8.2). That's a real milestone
  — we have extracted essentially everything the public recipes offer.
- The actual leaders sit at **~6.0 ft and have published nothing.** Closing that last ~2 ft is still
  an open problem with no known public method.
- A correction to the last report's leading hypothesis: we had guessed the leaders use a
  **resistivity sensor channel** we don't exploit. On inspection, **that channel simply isn't in the
  data the host gave us** — the hidden wells expose only depth/position/GR. So whatever the leaders
  do, it must be squeezed from the *same* signals we have (GR + trajectory + spatial interpolation),
  not from extra sensors. That makes the gap harder to explain but means it's at least *theoretically*
  reachable from our data.

---

## 10. The decision in front of us

We just learned that **leaning harder on the particle filter pays.** The natural next steps, roughly
in increasing effort:

- **(a) Tune the blend weight up.** A subtle but real clue: the model scored *better* on the hidden
  set than in our own testing, which suggests the particle filter is even more reliable on the real
  test than on our training wells — so our 44% weight may be **too low.** The strongest public
  recipes weight it around 70%. Re-submitting with a higher weight (say 55–70%) is nearly free (just
  change one number) and could gain more. **Highest value-per-effort; do this first.**
- **(b) Build out the full public architecture.** The ~8.2 notebooks add a second sequential tracker
  ("14-config beam search") and a *per-well router* that picks which estimator to trust based on the
  well's characteristics. More work, but it's the proven path to the bottom of the *published* range.
- **(c) Bank and harden.** Lock in 8.269 as the protected final submission and focus on making sure
  we never accidentally select a worse model at the deadline.

Where we are, stated plainly: **a solid, well-engineered 8.27 that has caught the best public
recipes — achieved by promoting a single under-used estimator from "one feature" to "half the
answer" — now looking at a ~2 ft gap to leaders who have shared nothing.**
