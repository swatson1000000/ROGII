"""Central configuration for ROGII — Wellbore Geology Prediction.

Verified facts about the competition (see ../SUMMARY.md):
- Task: predict True Vertical Thickness (TVT) along the lateral of each
  horizontal well, beyond the Prediction Start (PS) point.
- Target column: `tvt`. Units: feet.
- Metric: Kaggle config reports MSE; the official deck states RMSE of
  dTVT = manualTVT - predictedTVT. Same ranking, different displayed score.
"""

import os
from functools import lru_cache
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
# ROOT resolves to the git root from this file's location, making the project
# portable across machines. Override with ROGII_ROOT if a non-standard layout
# is needed.
ROOT   = Path(os.environ.get("ROGII_ROOT") or Path(__file__).resolve().parent.parent)
RAW    = ROOT / "data" / "raw"
PROC   = ROOT / "data" / "processed"
EXT    = ROOT / "data" / "external"
MODELS = ROOT / "models"
LOG    = ROOT / "log"

# ── Competition ───────────────────────────────────────────────────────────────
COMP_SLUG = "rogii-wellbore-geology-prediction"
TARGET    = "tvt"          # submission column to predict
UNITS     = "feet"

# Columns present in the horizontal-well files.
HW_COMMON_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]   # train + test
HW_TRAIN_ONLY  = ["TVT"]                                     # target (train only)
# Geological formation-top depths — TRAIN horizontal files only.
FORMATION_MARKERS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
# Type-well columns. `Geology` (formation code per depth) is train-only.
TYPEWELL_COLS = ["TVT", "GR", "Geology"]

# ── Cross-validation ────────────────────────────────────────────────────────
N_FOLDS = 5
SEED    = 42


# ── Well discovery helpers ────────────────────────────────────────────────────
def _well_ids(split: str) -> list:
    d = RAW / split
    if not d.exists():
        return []
    ids = {p.name.split("__")[0] for p in d.glob("*__horizontal_well.csv")}
    return sorted(ids)


@lru_cache(maxsize=2)
def train_well_ids() -> list:
    """Sorted list of training well ids (e.g. '000d7d20')."""
    return _well_ids("train")


@lru_cache(maxsize=2)
def test_well_ids() -> list:
    """Sorted list of test well ids."""
    return _well_ids("test")


def horizontal_path(well_id: str, split: str) -> Path:
    return RAW / split / f"{well_id}__horizontal_well.csv"


def typewell_path(well_id: str, split: str) -> Path:
    return RAW / split / f"{well_id}__typewell.csv"
