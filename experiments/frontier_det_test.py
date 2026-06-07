"""Option B test: how much OOF do we lose dropping the 25 stochastic (PF/stoch-DTW) features?
Retrain 1 GPU-LGB on the 197 deterministic features vs the full-222 (LGB seed42 = 10.67)."""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
tr = pd.read_parquet(ROOT / "data/processed/frontier/train_feats.parquet")

# the 25 stochastic columns identified by frontier_determinism_check (differ >1e-4 across rebuilds)
STOCH = ["pf_ancc", "pf_ancc_std", "pf_ancc_delta", "pf_z", "pf_z_delta", "pf_vs_z",
         "pf_vs_spatial", "pf_vs_dense", "dtw_vs_pf", "dtw_stoch_mean_d", "dtw_stoch_std",
         "dtw_stoch_cv", "sig_std", "sig_mean_d"] + [f"tdpf{int(o)}" for o in
         (-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30)]
all_feats = [c for c in tr.columns if c not in {"well", "id", "target"}]
det = [c for c in all_feats if c not in STOCH]
print(f">> total {len(all_feats)} feats; dropping {len(set(STOCH)&set(all_feats))} stochastic -> {len(det)} deterministic", flush=True)

y = tr["target"].to_numpy(np.float32)
rng = np.random.RandomState(42)
uw = tr["well"].unique().copy(); rng.shuffle(uw)
fold_of = {w: i % 5 for i, w in enumerate(uw)}
wf = tr["well"].map(fold_of).to_numpy()
splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(5)]

P = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89, min_child_samples=10,
         min_child_weight=0.5, n_estimators=5000, n_jobs=-1, reg_alpha=2.03, reg_lambda=87.28,
         subsample=0.645, subsample_freq=1, colsample_bytree=0.821, objective="regression",
         metric="rmse", verbose=-1, device_type="cuda", max_bin=255, random_state=42)

oof = np.zeros(len(tr), np.float32)
for f, (a, v) in enumerate(splits):
    dtr = lgb.Dataset(tr.iloc[a][det], label=y[a])
    dva = lgb.Dataset(tr.iloc[v][det], label=y[v], reference=dtr)
    m = lgb.train(P, dtr, valid_sets=[dva], num_boost_round=5000,
                  callbacks=[lgb.early_stopping(125, verbose=False)])
    oof[v] = m.predict(tr.iloc[v][det], num_iteration=m.best_iteration)
rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
print(f">> LGB-197 (deterministic only) OOF = {rmse:.4f}   (full-222 LGB seed42 = 10.666)", flush=True)
print(f"   cost of dropping PF/stoch-DTW = {rmse-10.666:+.4f} ft", flush=True)
import json
json.dump({"det_feats": det, "oof_det_lgb": rmse}, open(ROOT/"models/frontier/det_feats.json", "w"))
print("DET TEST DONE", flush=True)
