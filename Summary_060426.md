# Where We Are — ROGII Wellbore Geology Prediction (2026-06-04)

A plain-language status report. No oil-drilling knowledge assumed. Basic machine-learning
knowledge assumed (you know what a model, a feature, training, and cross-validation are).

---

## 1. What the competition actually asks

Companies drill **horizontal wells**: a borehole that goes straight down for a while, then bends
and travels sideways for thousands of feet through a layer of rock they want to stay inside. Think
of the rock as a layer cake, and the drill is trying to thread the sideways part of the well
through one specific layer of frosting without wandering up into the cake or down out of it.

To know whether the drill is in the right layer, geologists track a quantity called **TVT** — you
can think of it as **"how deep into the rock layer are we, vertically, at this point along the
well"**, measured in feet. While drilling, a geologist can figure out TVT for the part of the well
already drilled (the "known" part), but the competition hides the TVT for the later part of each
well and asks us to **predict it**.

So concretely: each well is a long sequence of measurement points (roughly one per foot of
drilling). For each well we are given:

- The **path** of the drill in 3D space (X, Y, depth) at every point.
- A sensor reading called **GR** (gamma ray) at every point — a number that responds to the type
  of rock. Different rock layers have a characteristic GR "signature."
- The **known TVT** for the early part of the well only. After a cutoff point (the "prediction
  start"), TVT is blank — **that's what we predict.**
- A **"typewell"**: a nearby vertical reference well where someone has already measured the full
  GR-vs-depth profile. This is the "answer key" for what GR *should* look like at each depth in
  this area.

We predict TVT for every hidden point. There are ~773 wells we can learn from, and a separate
**hidden set of ~200 wells** we're scored on. We never see those 200; we submit a piece of code
that runs on them.

---

## 2. How we're scored

The metric is **RMSE in feet** — root-mean-square error between our predicted TVT and the true
TVT, averaged over all the hidden points. **Lower is better.** Every number in this document is in
feet of error.

Two reference points to anchor your sense of scale:

- **15.9 ft** — the "dumb" baseline: just guess that TVT never changes after the cutoff. This is
  what you get for doing essentially nothing.
- **~6.7 ft** — the current best score on the public leaderboard (more on this below).

---

## 3. Where we stand right now

| | RMSE (ft) | Meaning |
|---|---|---|
| Dumb baseline | 15.9 | do nothing |
| **Our best submitted model** | **10.122** | our current banked result |
| Best *published* notebook | ~9.0 (claimed) | the best public recipe we can see |
| Public leaderboard top | ~6.7 | the actual leaders |

So we've closed about **60% of the distance** from "do nothing" (15.9) to "the leaders" (6.7). We
are roughly **3.4 ft behind the top** of the board. The leaders at 6.7–7.5 have **not published
how they did it** — there is no public recipe below ~9 ft. That gap is an open research problem,
not something we can copy.

**Nothing is currently running.** Our last experiment (described in §6) failed its quality bar, so
the banked best is still the 10.122 model.

---

## 4. How our best model works

This is the important part. Our best model is **not a deep learning model** — it's a **gradient-
boosted tree ensemble** (LightGBM + CatBoost). If you know neural nets but not trees: think of it
as a model that learns thousands of simple "if this feature is above X, nudge the prediction up by
Y" rules and adds them together. It is the dominant tool for this kind of tabular, feature-driven
problem, and it trains in minutes on a CPU/GPU rather than needing a big network.

The intelligence is **almost entirely in the features** — the hand-built input columns we feed the
model — not in the model architecture. The whole game is: turn each measurement point into a few
hundred informative numbers, then let the trees combine them. Our model uses **222 features** per
point. They come from a few orthogonal ideas, each attacking the problem a different way:

### Building block A — the geometry trick (the single biggest lever)

There's a near-exact physical relationship: at any point in a well,

> **TVT = −(depth) + (a formation-surface height) + (a per-well offset)**

The depth and the per-well offset we can get directly. The "formation-surface height" (called
**ANCC**) is a property of the rock layer that we *know* for the training wells but *not* for the
hidden ones. So the trick is **spatial interpolation**: to predict TVT at a hidden well located at
some (X, Y) map position, we look at the *nearby* training wells, see what their formation-surface
height was, and fit a tilted plane / weighted average through them to estimate it at our location.
This alone gets you most of the way from 15.9 down to ~12.

### Building block B — the GR "barcode" matching

The GR sensor trace is like a **barcode** for the rock. The typewell gives us the reference
barcode (GR at every depth). As the drill moves through the rock sideways, we can **slide its GR
readings against the typewell's barcode** and find the depth where they line up best — that depth
tells us TVT. We do this several different ways and feed all of them in:

- **Cross-correlation** at multiple zoom levels (compare short vs. long stretches of the barcode).
- **Particle filters** — a probabilistic tracker that follows the most likely TVT path down the
  well, like a GPS that keeps a cloud of guesses and updates them as new GR readings arrive.
- **Beam / Viterbi search** — another path-finder over the barcode with different stiffness
  settings.
- **DTW (dynamic time warping)** — flexible barcode alignment that allows stretching/squeezing.

No single one of these is accurate on its own. But each makes *different* mistakes, and the tree
model learns which to trust where. **A key, hard-won lesson of this project:** when we tested these
GR-matching features *one at a time*, every one looked useless and we threw them away. They only
pay off when added **all together** — about **1.4 ft of improvement** that we missed for weeks
because we were testing them individually. Reproducing a strong public notebook *wholesale*
instead of cherry-picking features is what unlocked it.

### Building block C — uncertainty and disagreement features

For each point we also compute how much the different estimators (A, B, the various trackers)
*disagree* with each other. High disagreement = "this point is hard, be cautious." The model uses
these as confidence signals.

### The ensemble + blend

We train several copies of the tree model with different random seeds and two different algorithms
(LightGBM ×3, CatBoost ×3), then **blend their predictions** with a simple weighted average whose
weights are chosen to minimize error. Diversity across models buys a little extra accuracy. A final
light **smoothing** pass cleans up the predicted TVT trajectory along each well.

### One important engineering detail

Some of the features (the particle filters) use **randomness**. We had a painful bug where the
random numbers came out differently when the code ran on Kaggle's hidden test vs. on our training
machine — which silently corrupted predictions and cost ~1.2 ft. The fix was to **seed the
randomness per well** (deterministically, from the well's ID) so the features are byte-for-byte
reproducible everywhere. This is now a permanent rule for all our builds.

---

## 5. Why not deep learning?

A natural instinct, given the sequential GR data, is "use an RNN / transformer." The evidence
across many competitors says **don't**: with only ~773 wells, sequence models *memorize* each
well's GR fingerprint and fail to generalize to new wells — they score around 14–15 ft, barely
better than the dumb baseline. The winning approach is tree ensembles on hand-engineered signals,
not end-to-end deep learning. (A heavily-regularized neural net might earn a small spot in a final
blend, but it's not the engine.)

---

## 6. What we just tried, and why it didn't beat 10.122

The playbook that got us from 11.9 → 10.1 was "**reproduce a better public notebook end-to-end**."
We tried it a third time on the best currently-published recipe (the "super-solution," claimed
~9 ft). We rebuilt all its features (170 of them) and trained the same kind of stack.

**Result: it scored ~10.45 in our cross-validation vs. our current model's ~10.36 — slightly
worse.** Per our own rule ("don't ship something that's a wash or worse"), we did **not** submit
it. It turned out *not* to be a strict superset of our current features — it traded away some of
our DTW machinery for new families and came out marginally behind.

One genuinely useful thing came out of it, though: we ran a careful test (a "block holdout," where
we hide an entire geographic cluster of wells to simulate the hidden test) and confirmed that the
**GR-matching features are real, transferable signal — not an artifact of overfitting.** This
matters because there was a worry that our score was just the geometry trick (building block A)
carrying overfit GR junk. That worry is now refuted: the GR matching genuinely helps even on
wells far from anything we trained on.

---

## 7. The honest picture of the gap

- We are at **10.122**. The best *published* recipe targets ~9. The actual leaders are at
  **~6.7–7.5 and have published nothing.**
- Everything in the public domain — every notebook, every forum post — tops out around 9 ft. One
  forum analysis even argues that point-by-point GR matching is near its useful limit and treats
  ~9 as a soft ceiling.
- So closing the last ~2–3 ft to the leaders is **not a matter of copying** — it requires a signal
  or method nobody has shared. The leading internal hypothesis is that the leaders are using the
  **resistivity / azimuthal sensor channel** (this competition is literally the host company's
  resistivity-inversion problem) as a direct measurement signal, which no public notebook does and
  which we currently use only indirectly. That's the most promising unexplored direction, but it's
  unproven.

---

## 8. The decision in front of us

Two candidate next steps:

- **(a) Finish ruling out the super-solution.** We only trained one of three CatBoost models in
  that experiment; training the other two and re-blending is cheap. But even in the best case this
  only *ties* our current 10.122 — it has no path to the 6.7 leaders. Low risk, low upside.
- **(b) Pivot to the resistivity / inversion lever.** The one physically-motivated signal no
  public notebook touches, and the best candidate for actually closing the gap to the leaders.
  Higher risk, but the only direction with real upside.

The reason this document exists is to decide (a) vs (b) — or something else — with a clear head
about where we actually are: **a solid, well-engineered 10.1 that has squeezed most of what the
public recipes offer, now staring at a ~3 ft gap that no published method is known to close.**
