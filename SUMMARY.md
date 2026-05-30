# ROGII — Wellbore Geology Prediction

> Kaggle Featured competition. This summary mixes **verified** facts (pulled via the Kaggle
> API and confirmed against the actual data files) with a few items still **unverified**
> (the official pages are JS-rendered and couldn't be scraped). Verified vs. unverified is
> marked throughout.

## The problem (geosteering)

When drilling a **horizontal** oil/gas well, the bit must stay inside a thin productive rock
layer that undulates underground. Keeping it there in real time is **geosteering**, normally
done by hand by a geologist who correlates the live gamma-ray (GR) log against a known
**type well** (a nearby vertical reference well whose geology is already interpreted).

The task is to **automate** this: predict the wellbore's stratigraphic position —
**True Vertical Thickness (TVT)** — along the lateral section of each well.

### What "TVT" means here (verified from the data)
- Each horizontal well has a **known landing section** at its start where the true TVT is
  given (column `TVT_input`). In the training files, `TVT_input` exactly equals the true
  `TVT` over this section (difference is 0.00 everywhere).
- Beyond that section, `TVT_input` is blank and **TVT is what you must predict** — i.e. track
  the bit's vertical position relative to the geology as the well advances.

## Key facts (verified via Kaggle API)

| Item | Value |
|------|-------|
| Host | ROGII (geoscience software for oil & gas) |
| Category | Featured |
| Evaluation metric | **MSE per Kaggle config; the official deck says RMSE of `dTVT`** — see note below |
| Prize pool | **$50,000 USD** |
| Opened | 2026-05-05 |
| Deadline | **2026-08-05 23:59 UTC** |
| Max team size | 5 |
| Target column | `tvt` |
| Units | **feet** (TVT, MD, etc.) |
| Train wells | **773** |
| Test wells | **3 visible** (likely a public sample; full/private test probably larger — *unverified*) |

### Metric note (discrepancy — verified)
- Kaggle's competition config reports the metric as **Mean Squared Error**.
- The official task deck (`AI_wellbore_geology_prediction_task_en.pptx`, slide 14) states:
  *"dTVT = manualTVT − predictedTVT for each predicted point; prediction quality is measured
  as the **RMSE** of all dTVT values."*
- MSE and RMSE give **identical rankings** (MSE = RMSE²), so model selection is unaffected, but
  the **displayed score differs**: a community "LB 9.25" is ≈9.25 ft RMSE *or* ≈3.04 ft RMSE
  depending on which is actually live. Confirm against your own submission.

## Files & schema (verified by downloading and inspecting)

Per well, three files keyed by an 8-char well id (e.g. `000d7d20`):

### `train/<id>__horizontal_well.csv`
Columns: `MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input`
- `MD` — measured depth along the borehole (row order follows the drilling path; ~1-foot step)
- `X, Y, Z` — 3D trajectory coordinates of the bit
- `GR` — gamma-ray log reading (the main correlation signal). **May contain NaN.**
- `TVT` — **target**, true vertical thickness (fully filled in train)
- `TVT_input` — known TVT up to the **Prediction Start (PS)** point only (== `TVT` there),
  blank after. PS is where your prediction must begin.
- `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA` — **top depth of each geological formation** along
  the trajectory. **Train only** — not available at test time.

### `test/<id>__horizontal_well.csv`
Columns: `MD, X, Y, Z, GR, TVT_input` — no `TVT`, no marker columns. `TVT_input` filled only
for the known landing section; you predict TVT for the rest.

### `<id>__typewell.csv` (the vertical reference well)
- Train: `TVT, GR, Geology` — GR profile vs. TVT, with a `Geology` formation code per depth.
- Test: `TVT, GR` (no `Geology`). The type well's TVT is **always known** (it's the reference).
- Each horizontal well is assigned **one** type well. (In ROGII's original naming the file is
  `Well1XXXX__typewell__Typewell2XXXX.csv`, implying type wells have their own ids and can be
  shared across horizontals; the Kaggle release bundles one per well id as `<id>__typewell.csv`.)
- `Geology` codes observed: `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, LTHL, LTGT, LBHL, MNSS`
  (plus blank between picks). These match the marker columns in the train horizontal file —
  i.e. the type well tells you the GR signature of each formation, which you match against the
  horizontal well's GR to locate stratigraphic position.

### `train/<id>.png`
A rendered plot of each well (773 of them, ~0.8 MB each). **Not downloaded** (visualization,
not needed for modeling on the CSVs). Available via the same API if wanted.

### `sample_submission.csv`
Columns: `id, tvt`, where `id = "<wellid>_<rowindex>"` and `rowindex` is the **0-based row
position** in that well's `horizontal_well.csv`. All `tvt` values are `0.0` in the sample.

## Submission format (verified)
- One row per scored point: `id, tvt`.
- Only the **lateral portion** of each test well is scored — the rows *after* the known
  landing section. Examples (test): `000d7d20` rows 1442–5277, `00bbac68` rows 1545–7558,
  `00e12e8b` rows 2083–6383. The split index differs per well and equals the length of that
  well's `TVT_input`-filled region.

## Trivial baselines to beat
- `sample_submission` predicts `tvt = 0` everywhere.
- Smarter floor: carry forward the **last known TVT** (the final `TVT_input` of the landing
  section) as a constant — geosteering then improves on this by tracking GR drift.

## Modeling angle (with hints from the official deck)
Core signal: correlate each horizontal well's `GR(MD)` against its type well's `GR(TVT)`
profile to infer TVT, anchored by the known TVT up to PS and constrained by the smooth
`MD`/trajectory geometry. Deck-stated hints:
- **GR signature matching**: rising/falling GR patterns map to TVT increasing/decreasing;
  constant GR ⇒ constant TVT. This is the manual geosteerer's core move.
- **Horizontal GR before PS is higher-resolution** than the type-well GR — the deck suggests
  using the horizontal well's own pre-PS GR (with its known TVT) to correlate the lateral,
  not just the type well.
- **Geology dips, and dip depends on drilling azimuth**; **neighboring/offset wells share dip
  behavior** — so cross-well spatial features (from `X, Y, Z` and well proximity) can predict
  the current well's geology. This supports modeling TVT as a continuous path with spatial
  structure rather than independent per-point regression.

Public leaderboard scores cluster tightly (≈9.9 → 9.25 in community notebooks).

## Confirmed from the official task deck
- Units are **feet**; predictions at ~**one-foot MD steps** beyond the PS point.
- Some `GR` values are **NaN**.
- Map/3D views show all wells are spatially co-located (offset-well correlation is viable).

## Still unverified (confirm on the Kaggle site)
- Whether the scored test set is just these 3 wells or a larger hidden/private set.
- The public/private leaderboard split.
- Whether the **live metric is MSE or RMSE** (Kaggle config vs. deck disagree — see Metric note).

## Data location
Downloaded to `./data/` (`data/train/`, `data/test/`, `data/sample_submission.csv`).
Bulk CSV download (1,546 files) runs via the Kaggle single-file API; the `download-all` zip
endpoint requires accepting competition rules at
https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules
