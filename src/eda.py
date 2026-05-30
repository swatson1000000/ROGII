"""Quick EDA over the ROGII wells. Writes a summary to data/processed/eda_summary.csv.

Run:
    conda activate kaggle && cd <project root>
    nohup python -u src/eda.py > log/eda_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .dataset import load_horizontal, load_typewell, prediction_start_index


def well_stats(well_id: str, split: str) -> dict:
    hw = load_horizontal(well_id, split)
    tw = load_typewell(well_id, split)
    ps = prediction_start_index(hw)
    row = {
        "well_id": well_id,
        "split": split,
        "hw_rows": len(hw),
        "ps_index": ps,
        "n_predict": len(hw) - ps,
        "gr_nan_frac": float(hw["GR"].isna().mean()),
        "md_min": float(hw["MD"].min()),
        "md_max": float(hw["MD"].max()),
        "tw_rows": len(tw),
    }
    if "TVT" in hw.columns:  # train only
        row["tvt_min"] = float(hw["TVT"].min())
        row["tvt_max"] = float(hw["TVT"].max())
        # sanity: TVT_input must equal TVT over the known section
        known = hw["TVT_input"].notna()
        row["tvt_input_matches"] = bool(
            np.allclose(hw.loc[known, "TVT"], hw.loc[known, "TVT_input"])
        )
    return row


def main() -> None:
    rows = []
    for split in ("train", "test"):
        ids = C.train_well_ids() if split == "train" else C.test_well_ids()
        print(f"{split}: {len(ids)} wells")
        for wid in ids:
            rows.append(well_stats(wid, split))
    df = pd.DataFrame(rows)
    out = C.PROC / "eda_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} wells)")
    print(df.describe(include="all").T[["count", "mean", "min", "max"]])


if __name__ == "__main__":
    main()
