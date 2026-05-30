# ROGII — Wellbore Geology Prediction — Copilot Instructions

**Task**: Predict True Vertical Thickness (TVT) along the lateral of horizontal wells beyond
the Prediction Start (PS) point — automating geosteering. Regression in **feet**. Metric:
Kaggle config = MSE; official deck = RMSE of `dTVT = manualTVT − predictedTVT` (same ranking).

See [plan.md](../plan.md) for the plan and [SUMMARY.md](../SUMMARY.md) for verified facts.

---

## Environment

- **Conda env is machine-dependent**: `kaggle-arch` on skynet (local, aarch64), `kaggle` on
  deepthought (remote, x86_64). See CLAUDE.md → Compute Environment for the three-machine
  inventory, dispatch (`runon`/`syncback`), and routing rules.
- **Project root**: `/home/swatson/work/kaggle/ROGII`

---

## Script Execution Policy

All Python scripts **must** run with `nohup` in the background with timestamped logs:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/kaggle/ROGII
rm -f log/train_*.log   # clean before each new training run
nohup python -u src/<script>.py [args] > log/<script>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f log/<script>_*.log
```

- **NEVER use `conda run`** for scripts that write log files — it buffers output, leaving the
  log empty while the process runs. Use `conda activate kaggle` directly before `nohup`.
- Core implementation in `src/*.py` only — **not** notebooks (notebooks are for submission).
- Shell scripts in `scripts/` must use **absolute paths**.

---

## Data Layout & Key Facts

- `data/raw/train/<id>__horizontal_well.csv` — `MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL,
  BUDA, TVT, GR, TVT_input`. **773 wells.**
- `data/raw/test/<id>__horizontal_well.csv` — `MD, X, Y, Z, GR, TVT_input` (no TVT, no markers).
- `data/raw/<split>/<id>__typewell.csv` — vertical reference: `TVT, GR[, Geology]`.
- `data/raw/sample_submission.csv` — `id, tvt`; `id = "<well_id>_<row_index>"` (0-based row in
  that well's horizontal file). **Only rows ≥ PS are scored.**

### Key pitfalls
- **`GR` can be NaN** — handle before feature building.
- **Formation markers (ANCC, …) are TRAIN-ONLY** — never feature on them at test time.
- **`Geology` column is TRAIN-ONLY** in the type well.
- **Split CV by well id, not by row** — geology is spatially autocorrelated; row-level splits
  leak. Offset/neighbouring wells share dip behaviour (deck slides 12–13).
- **Anchor on `TVT_input`**: TVT is known and equals `TVT_input` up to PS; predictions begin at PS.
- The horizontal well's own pre-PS GR is higher-resolution than the type well — use it to
  correlate the lateral (deck slide 9).

---

## Detailed Instructions

- **Dataset & features** → `.github/instructions/dataset.instructions.md`
- **Training** → `.github/instructions/training.instructions.md`
- **Inference & submission** → `.github/instructions/inference.instructions.md`
