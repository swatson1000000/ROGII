"""Scoring for ROGII wellbore geology prediction.

dTVT = manualTVT - predictedTVT for each predicted point.
Kaggle config = MSE; official deck = RMSE of dTVT. Both reported here.

Usage:
    python -u src/evaluate.py --pred submission.csv --truth oof_truth.csv
where each CSV has columns: id, tvt.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def score(pred: pd.DataFrame, truth: pd.DataFrame, target: str = "tvt") -> dict:
    """Return {'mse', 'rmse', 'mae', 'n'} aligning pred and truth on `id`."""
    m = truth.merge(pred, on="id", how="left", suffixes=("_true", "_pred"))
    if m[f"{target}_pred"].isna().any():
        missing = int(m[f"{target}_pred"].isna().sum())
        raise ValueError(f"{missing} ids in truth have no prediction")
    d = m[f"{target}_true"].to_numpy() - m[f"{target}_pred"].to_numpy()
    mse = float(np.mean(d**2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(d))), "n": len(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    s = score(pd.read_csv(args.pred), pd.read_csv(args.truth))
    print(f"n={s['n']}  RMSE={s['rmse']:.4f}  MSE={s['mse']:.4f}  MAE={s['mae']:.4f}  (feet)")


if __name__ == "__main__":
    main()
