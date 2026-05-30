"""Phase-3 test inference -> submission.csv (see ../plan.md §10).

Loads the per-fold LightGBM models saved by src/train.py, predicts drift on the
test feature matrix, averages folds, adds the per-well anchor back, and writes a
submission (id, tvt) ordered to match sample_submission.csv.

End-to-end sanity check: the 3 visible test wells are byte-identical to train
wells (TVT only blanked), so we score our test predictions against the *known*
train TVT. This validates the whole pipeline produces sane values — it is NOT a
leaderboard estimate (real scoring is a hidden set).

Run:
    conda activate kaggle-arch && cd <project root>
    python -m src.predict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import dataset as D
from . import features as F
from .evaluate import score

TAG = "phase3"


def _load_models() -> list:
    import lightgbm as lgb
    models = []
    for f in range(C.N_FOLDS):
        p = C.MODELS / f"lgb_{TAG}_fold{f}.txt"
        if not p.exists():
            raise FileNotFoundError(f"missing {p} — run `python -m src.train` first")
        models.append(lgb.Booster(model_file=str(p)))
    return models


def predict_test() -> pd.DataFrame:
    """Return submission DataFrame [id, tvt] for the test split."""
    mat = F.build_feature_matrix("test", cache=True)
    feat_cols = F.feature_columns(mat)            # 67 cols, same order as train
    models = _load_models()
    drift = np.mean([m.predict(mat[feat_cols]) for m in models], axis=0)
    sub = pd.DataFrame({"id": mat["id"], C.TARGET: mat["anchor_tvt"].to_numpy() + drift})
    # order to match sample_submission exactly
    ss = pd.read_csv(C.RAW / "sample_submission.csv")
    sub = ss[["id"]].merge(sub, on="id", how="left")
    assert sub[C.TARGET].notna().all(), "some sample_submission ids got no prediction"
    return sub


def _sanity_check(sub: pd.DataFrame) -> None:
    """Score the 3 visible wells against their known (train) TVT."""
    rows = []
    for wid in C.test_well_ids():
        tr = D.load_horizontal(wid, "train")       # identical well, TVT present
        ps = D.prediction_start_index(tr)
        for i in range(ps, len(tr)):
            rows.append((f"{wid}_{i}", float(tr["TVT"].iloc[i])))
    truth = pd.DataFrame(rows, columns=["id", C.TARGET])
    s = score(sub[sub["id"].isin(truth["id"])], truth)
    print(f"[sanity] 3 visible wells vs known train TVT: RMSE={s['rmse']:.4f}  "
          f"MAE={s['mae']:.4f}  n={s['n']:,}")
    print("[sanity] NOTE: optimistic — these wells were in the training folds. "
          "This checks pipeline sanity, not generalization.")


def main() -> None:
    sub = predict_test()
    out = C.PROC / f"submission_{TAG}.csv"
    sub.to_csv(out, index=False)
    print(f"[submit] wrote {len(sub):,} rows -> {out}")
    print(f"[submit] tvt range [{sub[C.TARGET].min():.1f}, {sub[C.TARGET].max():.1f}]  "
          f"mean={sub[C.TARGET].mean():.1f}")
    _sanity_check(sub)


if __name__ == "__main__":
    main()
