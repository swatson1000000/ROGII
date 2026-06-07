"""Super-solution train (part 1): GPU-LGB x3 diverse-lr on the ~170-feat super union.

Uses the super-solution's TUNED LGB params (num_leaves=255, min_child=15, reg_lambda=3, ×3 lr
0.025/0.020/0.030 @8000it early-stop 250) -- the "wholesale" reproduction, not konbu's params.
Split = frontier's shuffled GroupKFold(seed 42) so the combined OOF gates apples-to-apples vs
frontier 10.356. Saves OOF + test preds per seed -> models/super/.

Run (skynet, CUDA-LGB): nohup python -u experiments/super_train_lgb.py > log/super_lgb_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
SP = ROOT / "data/processed/super"
MODELS = ROOT / "models/super"
MODELS.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
SPLIT_SEED = 42

# super-solution tuned LGB base (device_type cuda = skynet sm_121 build)
LGB_BASE = dict(
    boosting_type="gbdt", num_leaves=255, min_child_samples=15,
    subsample=0.75, subsample_freq=1, colsample_bytree=0.75,
    reg_lambda=3.0, reg_alpha=0.05, min_split_gain=0.01,
    objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
    device_type="cuda", max_bin=255)
# three diverse-lr configs (seed = LGB random_state)
LGB_CONFIGS = [
    dict(learning_rate=0.025, seed=42),
    dict(learning_rate=0.020, seed=7),
    dict(learning_rate=0.030, seed=123),
]
N_EST = 8000
EARLY = 250

print(">> loading super matrices...", flush=True)
train_df = pd.read_parquet(SP / "train_feats.parquet")
test_df = pd.read_parquet(SP / "test_feats.parquet")
feature_cols = [c for c in train_df.columns if c not in {"well", "id", "target"}]
print(f"   train {train_df.shape}  #feats={len(feature_cols)}  (gate vs frontier 10.356)", flush=True)
json.dump(feature_cols, open(MODELS / "feature_cols.json", "w"))

# shuffled GroupKFold-5, seed 42 (identical protocol to frontier_train_lgb)
rng = np.random.RandomState(SPLIT_SEED)
uw = train_df["well"].unique().copy(); rng.shuffle(uw)
fold_of = {w: i % N_SPLITS for i, w in enumerate(uw)}
wf = train_df["well"].map(fold_of).to_numpy()
splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]
y = train_df["target"].to_numpy(np.float32)

results = {}
for cfg in LGB_CONFIGS:
    seed = cfg["seed"]
    print(f"\n>> LGB(cuda) seed={seed} lr={cfg['learning_rate']}", flush=True)
    params = dict(LGB_BASE, learning_rate=cfg["learning_rate"], random_state=seed, seed=seed)
    oof = np.zeros(len(train_df), np.float32)
    test_pred = np.zeros(len(test_df), np.float32)
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(train_df.iloc[tr][feature_cols], label=y[tr])
        dva = lgb.Dataset(train_df.iloc[va][feature_cols], label=y[va], reference=dtr)
        m = lgb.train(params, dtr, valid_sets=[dva], num_boost_round=N_EST,
                      callbacks=[lgb.early_stopping(EARLY, verbose=False)])
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
oofs = np.column_stack([np.load(MODELS / f"oof_lgb_{c['seed']}.npy") for c in LGB_CONFIGS])
r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(oofs, y)
blend_rmse = float(np.sqrt(np.mean((r.predict(oofs) - y) ** 2)))
print(f"\n=== super LGB summary ===", flush=True)
for k, v in results.items():
    print(f"  {k}: {v:.4f}", flush=True)
print(f"  3-seed NNLS blend: {blend_rmse:.4f}  (frontier 6-model 10.356; gate vs that)", flush=True)
json.dump({"per_seed": results, "lgb3_blend": blend_rmse}, open(MODELS / "lgb_summary.json", "w"), indent=2)
print("=== SUPER LGB DONE ===", flush=True)
