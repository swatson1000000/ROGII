"""Phase-0 scoring harness for ROGII — Wellbore Geology Prediction.

Three pieces every later phase depends on (see ../plan.md §5 Phase 0, §4):

  1. GroupKFold-by-well split  — each well is one group; geology is a per-well
     GR fingerprint, so a random row split would leak it and inflate CV.
  2. simulate-test-on-train     — for a TRAIN well we already know its PS point
     (first NaN in TVT_input). Hide TVT for rows >= PS, predict, and score
     against the held true TVT. This mirrors exactly how the hidden test is
     scored, and is how every OOF number in the plan was produced.
  3. null baseline              — predict the last known TVT (constant) for all
     rows >= PS. The floor to beat (~15.9-16.2 ft RMSE).

Run the Phase-0 check (reproduce the null baseline + validate test ids):
    conda activate kaggle-arch && cd <project root>
    python -m src.harness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config as C
from . import dataset as D
from .evaluate import score


# ── Cross-validation ──────────────────────────────────────────────────────────
def make_well_folds(well_ids: list[str] | None = None, n_folds: int = C.N_FOLDS) -> dict[str, int]:
    """Assign each training well to one of `n_folds` CV folds.

    GroupKFold at the well level: each well is a single group, so it lands wholly
    in one fold (no row-level leakage of a well's GR fingerprint). GroupKFold is
    deterministic (it ignores any seed), so the assignment is stable across runs.

    Returns {well_id: fold_index}.
    """
    if well_ids is None:
        well_ids = C.train_well_ids()
    well_ids = sorted(well_ids)
    gkf = GroupKFold(n_splits=n_folds)
    dummy = np.zeros(len(well_ids))
    fold_of: dict[str, int] = {}
    for fold, (_, test_idx) in enumerate(gkf.split(dummy, groups=well_ids)):
        for i in test_idx:
            fold_of[well_ids[i]] = fold
    return fold_of


# ── simulate-test-on-train ──────────────────────────────────────────────────
def last_known_tvt(hw: pd.DataFrame, ps: int) -> float:
    """Final filled TVT_input value (the value at row ps-1) = the anchor TVT."""
    return float(hw["TVT_input"].iloc[ps - 1])


def build_truth(split: str = "train") -> pd.DataFrame:
    """Ground truth for the eval zone: columns [id, tvt] for rows >= PS.

    Only valid for `train` (test has no TVT). Used to score any predictions
    produced by the simulate-test-on-train loop.
    """
    if split != "train":
        raise ValueError("truth is only available for the train split")
    rows = []
    for wid, hw, _ in D.iter_wells(split):
        ps = D.prediction_start_index(hw)
        for i in range(ps, len(hw)):
            rows.append((f"{wid}_{i}", float(hw["TVT"].iloc[i])))
    return pd.DataFrame(rows, columns=["id", C.TARGET])


def null_predictions(split: str = "train") -> pd.DataFrame:
    """Null baseline: predict last_known_TVT (constant) for every eval row.

    Works for both splits — needs only TVT_input, which both have.
    Columns: [id, tvt].
    """
    rows = []
    for wid, hw, _ in D.iter_wells(split):
        ps = D.prediction_start_index(hw)
        anchor = last_known_tvt(hw, ps)
        for i in range(ps, len(hw)):
            rows.append((f"{wid}_{i}", anchor))
    return pd.DataFrame(rows, columns=["id", C.TARGET])


# ── Phase-0 verification ──────────────────────────────────────────────────────
def _check_test_ids() -> None:
    """The test eval-row ids must exactly match sample_submission.csv."""
    ss = pd.read_csv(C.RAW / "sample_submission.csv")
    expected = set(ss["id"])
    got = set(D.submission_ids("test"))
    missing, extra = expected - got, got - expected
    assert not missing, f"{len(missing)} sample_submission ids not produced, e.g. {list(missing)[:3]}"
    assert not extra, f"{len(extra)} extra ids produced, e.g. {list(extra)[:3]}"
    print(f"[ids] OK — {len(got)} test eval ids match sample_submission.csv exactly")


def main() -> None:
    folds = make_well_folds()
    sizes = pd.Series(list(folds.values())).value_counts().sort_index()
    print(f"[cv]  {C.N_FOLDS}-fold GroupKFold over {len(folds)} wells; "
          f"per-fold well counts = {sizes.to_dict()}")

    truth = build_truth("train")
    pred = null_predictions("train")
    s = score(pred, truth)
    print(f"[null] simulate-test-on-train over {len(C.train_well_ids())} wells: "
          f"RMSE={s['rmse']:.4f}  MSE={s['mse']:.4f}  MAE={s['mae']:.4f}  n={s['n']}")

    gate = abs(s["rmse"] - 15.9) <= 0.5  # plan gate: within ~0.3-0.5 of 15.9
    print(f"[gate] null RMSE near 15.9: {'PASS' if gate else 'FAIL'}")

    _check_test_ids()


if __name__ == "__main__":
    main()
