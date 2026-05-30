"""Phase-3 training: LightGBM on the drift target, GroupKFold(5) by well.

Pipeline (see ../plan.md §5 Phase 3):
  1. Build/load the 67-feat base+spatial drift matrix (src/features.py).
  2. GroupKFold-by-well via src/harness.make_well_folds (no row leakage).
  3. Train one LightGBM per fold on drift = TVT - last_known_TVT; collect OOF.
  4. Reconstruct TVT_pred = anchor_tvt + drift_pred; score vs the held true TVT
     with the exact competition metric (src/evaluate.score).
  5. Save OOF predictions, per-fold models, and feature importance (TAG-named).

Gate: OOF RMSE <= 12.1 (konbu's formation-family OOF; the plan's <=11.5 was set
off konbu's LB and is the Phase-4+ target, not reachable by formation KNN alone).

Run (per CLAUDE.md execution policy; CPU is fine):
    conda activate kaggle-arch && cd <project root>
    rm -f log/train_*.log
    nohup python -u -m src.train > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    tail -f log/train_*.log
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import harness as H
from .evaluate import score
from .utils import set_seed

# Phase-3 params: the validated "push" config (probe3) over the 67-feat base+spatial
# matrix. Verified signal-limited, not capacity-limited — heavier model (160 lv) over
# the lean 63-lv gives OOF 12.762 vs 12.786, a noise-level delta — so this is near the
# formation-KNN family's ceiling (konbu OOF ~12.11). CPU on skynet is the right lane:
# ~3.8M x 67 dense floats is LightGBM's CPU sweet spot, and neither machine has a
# GPU-enabled LightGBM build anyway. The full §7 ensemble is Phase 5.
TAG = "phase3"
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    num_leaves=160,
    learning_rate=0.03,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=120,
    lambda_l2=2.0,
    n_jobs=-1,
    verbosity=-1,
)
N_EST = 4000
EARLY_STOP = 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=C.N_FOLDS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    set_seed(args.seed)
    t0 = time.time()

    mat = F.build_feature_matrix("train", cache=True)
    feat_cols = F.feature_columns(mat)
    mat["well"] = mat["id"].str.rsplit("_", n=1).str[0]
    fold_of = H.make_well_folds(n_folds=args.folds)
    mat["fold"] = mat["well"].map(fold_of)
    print(f"[data] {len(mat):,} eval rows  {len(feat_cols)} feats  "
          f"{mat['well'].nunique()} wells  folds={sorted(mat['fold'].unique())}", flush=True)

    X = mat[feat_cols]
    y = mat["drift"].to_numpy()
    oof_drift = np.zeros(len(mat))
    importances = np.zeros(len(feat_cols))
    truth = H.build_truth("train")                 # built once, reused below

    for fold in range(args.folds):
        tr = mat["fold"] != fold
        va = mat["fold"] == fold
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va])
        model = lgb.train(
            {**LGB_PARAMS, "seed": args.seed + fold},
            dtr, num_boost_round=N_EST, valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof_drift[va.to_numpy()] = model.predict(X[va], num_iteration=model.best_iteration)
        importances += model.feature_importance(importance_type="gain")
        model.save_model(str(C.MODELS / f"lgb_{TAG}_fold{fold}.txt"))
        # per-fold TVT score
        fpred = pd.DataFrame({"id": mat.loc[va, "id"], C.TARGET: mat.loc[va, "anchor_tvt"].to_numpy()
                              + oof_drift[va.to_numpy()]})
        fs = score(fpred, truth[truth["id"].isin(fpred["id"])])
        print(f"[fold {fold}] best_iter={model.best_iteration:4d}  "
              f"TVT RMSE={fs['rmse']:.4f}  n={fs['n']:,}", flush=True)

    # ── overall OOF ─────────────────────────────────────────────────────────────
    oof = pd.DataFrame({"id": mat["id"], C.TARGET: mat["anchor_tvt"].to_numpy() + oof_drift})
    s = score(oof, truth)
    null = float(np.sqrt(np.mean((truth[C.TARGET].to_numpy() - mat["anchor_tvt"].to_numpy()) ** 2)))
    print(f"\n[OOF] TVT  RMSE={s['rmse']:.4f}  MSE={s['mse']:.4f}  MAE={s['mae']:.4f}  n={s['n']:,}")
    print(f"[OOF] null RMSE={null:.4f}  ->  improvement = {null - s['rmse']:.4f} ft")
    print(f"[gate] OOF <= 12.1 (konbu OOF / Phase-3 target): {'PASS' if s['rmse'] <= 12.1 else 'FAIL'}")
    print(f"[gate] OOF <= 11.5 (plan Phase-3 gate): {'PASS' if s['rmse'] <= 11.5 else 'FAIL'}")

    C.PROC.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(C.PROC / f"oof_{TAG}.parquet", index=False)
    imp = (pd.DataFrame({"feature": feat_cols, "gain": importances / args.folds})
           .sort_values("gain", ascending=False).reset_index(drop=True))
    imp.to_csv(C.PROC / f"importance_{TAG}.csv", index=False)
    print("\n[importance] top 15 by gain:")
    print(imp.head(15).to_string(index=False))
    print(f"\n[done] {time.time() - t0:.0f}s  OOF -> {C.PROC / f'oof_{TAG}.parquet'}")
    (C.PROC / f"{TAG}_result.json").write_text(json.dumps(
        {"oof_rmse": s["rmse"], "oof_mse": s["mse"], "null_rmse": null, "n": s["n"]}, indent=2))


if __name__ == "__main__":
    main()
