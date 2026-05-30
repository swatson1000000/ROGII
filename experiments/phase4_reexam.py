"""Phase-4 re-examination: do the GR-vs-typewell MATCHING features actually help
the GBM, or was "GR conclusively dead" right?

Our Phase-4 verdict tested GR as a residual point-ESTIMATOR (beam/NCC/DTW -> a TVT
guess), found it aliases, and declared it dead — then we DELETED these features.
konbu instead feeds them as weak GBM INPUTS. This ablation isolates that group on
konbu's cached feature matrix: train identical GPU-LGB with vs without the matching
features (same shuffled group folds) and compare OOF.

Matching group (the Phase-4 "dead" signals): beam_*_delta, beam_gap, tw_diff_*,
ncc_*_shift_well, prefix_tw_rmse, prefix_tw_mae.
Local-texture GR (NOT a typewell match, kept in both arms): gr, gr_roll*, gr_grad,
gr_std*, gr_lag*, gr_lead*, gr_cumsum, gr_fft_*.
"""
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, re

ART = Path("/home/swatson/work/kaggle/ROGII/data/processed/konbu")
N_SPLITS, SPLIT_SEED = 5, 42
LGB = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89, min_child_samples=10,
           min_child_weight=0.5, n_estimators=5000, reg_alpha=2.03, reg_lambda=87.28,
           subsample=0.645, subsample_freq=1, colsample_bytree=0.821, objective="regression",
           metric="rmse", verbose=-1, device_type="cuda", max_bin=255, seed=42)

df = pd.read_parquet(ART / "train_feats.parquet")
all_feats = [c for c in df.columns if c not in {"well", "prediction_id", "target"}]
y = df["target"].to_numpy()

def is_matching(c):
    return bool(re.match(r"beam_|tw_diff_|ncc_.*shift_well|prefix_tw_(rmse|mae)", c))
matching = [c for c in all_feats if is_matching(c)]
print(f"#all feats={len(all_feats)}  #matching (Phase-4 'dead') feats={len(matching)}")
print("matching group:", matching)

rng = np.random.RandomState(SPLIT_SEED)
wells = df["well"].unique().copy(); rng.shuffle(wells)
fold_of = {w: i % N_SPLITS for i, w in enumerate(wells)}
wf = df["well"].map(fold_of).to_numpy()
splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]

def oof_rmse(cols, tag):
    oof = np.zeros(len(df), np.float32)
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(df.iloc[tr][cols], label=y[tr])
        dva = lgb.Dataset(df.iloc[va][cols], label=y[va], reference=dtr)
        m = lgb.train(LGB, dtr, valid_sets=[dva], num_boost_round=LGB["n_estimators"],
                      callbacks=[lgb.early_stopping(125, verbose=False)])
        oof[va] = m.predict(df.iloc[va][cols], num_iteration=m.best_iteration)
    r = float(np.sqrt(np.mean((oof - y) ** 2)))
    print(f"[{tag}] OOF rmse={r:.4f}  (#feats={len(cols)})", flush=True)
    return r

full = oof_rmse(all_feats, "ALL 78")
without = oof_rmse([c for c in all_feats if c not in set(matching)], "WITHOUT matching")
print(f"\n>>> Phase-4 matching features contribute: {without - full:+.4f} ft to OOF")
print(">>> (positive = removing them HURTS = they help = 'GR dead' was wrong)")
