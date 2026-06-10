"""Ensemble retrain (part 2): GPU-CatBoost x3 on the 234-feat frontier_ens union.

Identical to experiments/bet5_train_cat.py except FR/MODELS -> frontier_ens. Same Optuna-tuned
params, same GKF-5 seed42 as the LGB leg. Saves OOF + test preds for the 6-model Ridge blend.

Run (GPU CatBoost): nohup python -u experiments/ens_train_cat.py > log/ens_cat_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path("/home/swatson/work/kaggle/ROGII")
FR = ROOT / "data/processed/frontier_ens"
MODELS = ROOT / "models/frontier_ens"
MODELS.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
SEEDS = [42, 7, 123]

tuned = json.load(open(ROOT / "models/konbu/cat_hpo.json"))["best_params"]
print(f">> tuned base params: {tuned}", flush=True)

train_df = pd.read_parquet(FR / "train_feats.parquet")
test_df = pd.read_parquet(FR / "test_feats.parquet")
feature_cols = [c for c in train_df.columns if c not in {"well", "id", "target"}]
X = train_df[feature_cols].to_numpy(np.float32)
Xt = test_df[feature_cols].to_numpy(np.float32)
y = train_df["target"].to_numpy(np.float32)
print(f">> train {X.shape}  #feats={len(feature_cols)}", flush=True)

rng = np.random.RandomState(42)
uw = train_df["well"].unique().copy(); rng.shuffle(uw)
fold_of = {w: i % N_SPLITS for i, w in enumerate(uw)}
wf = train_df["well"].map(fold_of).to_numpy()
splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]

results = {}
for seed in SEEDS:
    print(f"\n>> CatBoost(gpu) seed={seed}", flush=True)
    params = dict(iterations=5000, task_type="GPU", devices="0",
                  loss_function="RMSE", eval_metric="RMSE",
                  depth=tuned["depth"], learning_rate=tuned["learning_rate"],
                  l2_leaf_reg=tuned["l2_leaf_reg"], random_strength=tuned["random_strength"],
                  bagging_temperature=tuned["bagging_temperature"], border_count=254,
                  od_type="Iter", od_wait=150, random_seed=seed, verbose=0)
    oof = np.zeros(len(train_df), np.float32)
    test_pred = np.zeros(len(test_df), np.float32)
    for fold, (tr, va) in enumerate(splits):
        m = CatBoostRegressor(**params)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]), use_best_model=True, verbose=0)
        oof[va] = m.predict(X[va])
        test_pred += m.predict(Xt) / N_SPLITS
        m.save_model(str(MODELS / f"cat_seed{seed}_fold{fold}.cbm"))
        print(f"   fold {fold}: rmse={np.sqrt(np.mean((oof[va]-y[va])**2)):.4f} it={m.get_best_iteration()}", flush=True)
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    print(f"   CatBoost seed={seed}: OOF rmse={rmse:.4f}", flush=True)
    np.save(MODELS / f"oof_cat_{seed}.npy", oof)
    np.save(MODELS / f"test_cat_{seed}.npy", test_pred)
    results[f"cat_{seed}"] = rmse

print(f"\n=== CatBoost-234 (frontier_ens) summary ===", flush=True)
for k, v in results.items():
    print(f"  {k}: {v:.4f}", flush=True)
json.dump(results, open(MODELS / "cat_summary.json", "w"), indent=2)
print("=== ENS CAT DONE ===", flush=True)
