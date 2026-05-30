"""Data loading for ROGII wellbore geology prediction.

Each well has two CSVs in data/raw/{train,test}/:
  <id>__horizontal_well.csv  — the drilled lateral (MD, X, Y, Z, GR, TVT_input,
                               plus TVT + formation markers in TRAIN only)
  <id>__typewell.csv         — the vertical reference (TVT, GR, +Geology in TRAIN)

Geosteering structure (verified):
  TVT_input is filled only up to the Prediction Start (PS) point and equals the
  true TVT there. Rows after PS are what must be predicted (the submission rows).
"""

from __future__ import annotations

import pandas as pd

from . import config as C


def load_horizontal(well_id: str, split: str) -> pd.DataFrame:
    """Load one horizontal-well file. GR may contain NaN."""
    return pd.read_csv(C.horizontal_path(well_id, split))


def load_typewell(well_id: str, split: str) -> pd.DataFrame:
    """Load one type-well (vertical reference) file."""
    return pd.read_csv(C.typewell_path(well_id, split))


def prediction_start_index(hw: pd.DataFrame) -> int:
    """0-based row index where prediction begins = first row with empty TVT_input.

    Rows [0, ps) are the known landing section; rows [ps, end] are predicted.
    The submission id for row i of well W is f"{W}_{i}".
    """
    known = hw["TVT_input"].notna()
    if known.all():
        return len(hw)
    return int(known.values.argmin())


def iter_wells(split: str):
    """Yield (well_id, horizontal_df, typewell_df) for every well in a split."""
    ids = C.train_well_ids() if split == "train" else C.test_well_ids()
    for wid in ids:
        yield wid, load_horizontal(wid, split), load_typewell(wid, split)


def submission_ids(split: str = "test") -> list[str]:
    """Reconstruct the list of scored ids = rows at/after PS for each well."""
    ids = []
    well_ids = C.train_well_ids() if split == "train" else C.test_well_ids()
    for wid in well_ids:
        hw = load_horizontal(wid, split)
        ps = prediction_start_index(hw)
        ids.extend(f"{wid}_{i}" for i in range(ps, len(hw)))
    return ids
