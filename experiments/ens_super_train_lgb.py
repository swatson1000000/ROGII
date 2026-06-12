"""frontier_super retrain (part 1): GPU-LGB x3 on the union matrix (nouk-231 + 28 super cols).

Identical to experiments/ens_nouk_train_lgb.py except it reads data/processed/frontier_super
(UK already excluded at build time) and writes models/frontier_super/. SAME folds/params/seeds.

Run (cluster job, pinned GB10 / or local skynet):
  nohup python -u experiments/ens_super_train_lgb.py > log/ens_super_lgb_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
FR = ROOT / "data/processed/frontier_super"
MODELS = ROOT / "models/frontier_super"
MODELS.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
SPLIT_SEED = 42
LGB_SEEDS = [42, 7, 123]

LGB_PARAMS = dict(
    boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
    min_child_samples=10, min_child_weight=0.5, n_estimators=5000, n_jobs=-1,
    reg_alpha=2.03, reg_lambda=87.28, subsample=0.645, subsample_freq=1,
    colsample_bytree=0.821, objective="regression", metric="rmse", verbose=-1,
    device_type="cuda", max_bin=255)

print(">> loading frontier_super matrices...", flush=True)
train_df = pd.read_parquet(FR / "train_feats.parquet")
test_df = pd.read_parquet(FR / "test_feats.parquet")
feature_cols = [c for c in train_df.columns if c not in {"well", "id", "target"}]
print(f"   train {train_df.shape}  #feats={len(feature_cols)}", flush=True)
assert all(c in test_df.columns for c in feature_cols), "test missing feature cols"
json.dump(feature_cols, open(MODELS / "feature_cols.json", "w"))

rng = np.random.RandomState(SPLIT_SEED)
uw = train_df["well"].unique().copy(); rng.shuffle(uw)
fold_of = {w: i % N_SPLITS for i, w in enumerate(uw)}
wf = train_df["well"].map(fold_of).to_numpy()
splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]
y = train_df["target"].to_numpy(np.float32)

results = {}
for seed in LGB_SEEDS:
    print(f"\n>> LGB(gpu) seed={seed}", flush=True)
    params = dict(LGB_PARAMS); params["random_state"] = seed
    oof = np.zeros(len(train_df), np.float32)
    test_pred = np.zeros(len(test_df), np.float32)
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(train_df.iloc[tr][feature_cols], label=y[tr])
        dva = lgb.Dataset(train_df.iloc[va][feature_cols], label=y[va], reference=dtr)
        m = lgb.train(params, dtr, valid_sets=[dva], num_boost_round=params["n_estimators"],
                      callbacks=[lgb.early_stopping(125, verbose=False)])
        oof[va] = m.predict(train_df.iloc[va][feature_cols], num_iteration=m.best_iteration)
        test_pred += m.predict(test_df[feature_cols], num_iteration=m.best_iteration) / N_SPLITS
        m.save_model(str(MODELS / f"lgb_seed{seed}_fold{fold}.txt"), num_iteration=m.best_iteration)
        print(f"   fold {fold}: rmse={np.sqrt(np.mean((oof[va]-y[va])**2)):.4f} it={m.best_iteration}", flush=True)
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    print(f"   LGB seed={seed}: OOF rmse={rmse:.4f}", flush=True)
    np.save(MODELS / f"oof_lgb_{seed}.npy", oof)
    np.save(MODELS / f"test_lgb_{seed}.npy", test_pred)
    results[f"lgb_{seed}"] = rmse

from sklearn.linear_model import Ridge
oofs = np.column_stack([np.load(MODELS / f"oof_lgb_{s}.npy") for s in LGB_SEEDS])
r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(oofs, y)
blend_rmse = float(np.sqrt(np.mean((r.predict(oofs) - y) ** 2)))
print(f"\n=== LGB frontier_super summary ===", flush=True)
for k, v in results.items():
    print(f"  {k}: {v:.4f}", flush=True)
print(f"  3-seed NNLS blend: {blend_rmse:.4f}", flush=True)
json.dump({"per_seed": results, "lgb3_blend": blend_rmse}, open(MODELS / "lgb_summary.json", "w"), indent=2)
print("=== ENS SUPER LGB DONE ===", flush=True)
