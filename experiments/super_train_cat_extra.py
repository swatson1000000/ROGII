"""Super-solution train (part 2b): GPU-CatBoost seeds 7 & 123 on the ~170-feat super union.

Option (a) of the 2026-06-04 fork: rule out the CatBoost-diversity gap before declaring the
super-solution dead. The original super_train_cat.py trained CatBoost ONCE (seed 42), which now
carries 0.60 of the blend. In the *frontier* blend, cat seeds 7 & 123 carried 0.67 of the weight
(cat_7 0.355 + cat_123 0.319) vs cat_42's 0.116 -> CatBoost diversity was the strongest lever
there. This adds those two seeds so super_blend can re-gate a 6-model stack vs frontier 10.356.

Same tuned super CatBoost params (depth=7, lr=0.025, l2=2, min_data=15, subsample=0.75,
border=254, 8000it od_wait=300) and the SAME shuffled GKF-5 seed42 split as super_train_lgb /
super_train_cat. Only random_seed varies across models (mirrors frontier_train_cat).

Run (deepthought GPU): nohup python -u experiments/super_train_cat_extra.py > log/super_cat_extra_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path("/home/swatson/work/kaggle/ROGII")
SP = ROOT / "data/processed/super"
MODELS = ROOT / "models/super"
MODELS.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
SPLIT_SEED = 42
SEEDS = [7, 123]  # cat_42 already trained by super_train_cat.py


def cb_params(seed):
    return dict(
        iterations=8000, learning_rate=0.025, depth=7, l2_leaf_reg=2.0,
        min_data_in_leaf=15, subsample=0.75, bootstrap_type="Bernoulli", border_count=254,
        loss_function="RMSE", eval_metric="RMSE", task_type="GPU", devices="0",
        od_type="Iter", od_wait=300, random_seed=seed, verbose=0)


train_df = pd.read_parquet(SP / "train_feats.parquet")
test_df = pd.read_parquet(SP / "test_feats.parquet")
feature_cols = [c for c in train_df.columns if c not in {"well", "id", "target"}]
X = train_df[feature_cols].to_numpy(np.float32)
Xt = test_df[feature_cols].to_numpy(np.float32)
y = train_df["target"].to_numpy(np.float32)
print(f">> train {X.shape}  #feats={len(feature_cols)}", flush=True)

# identical shuffled GKF-5 split (seed 42) used by every super/frontier model
rng = np.random.RandomState(SPLIT_SEED)
uw = train_df["well"].unique().copy(); rng.shuffle(uw)
fold_of = {w: i % N_SPLITS for i, w in enumerate(uw)}
wf = train_df["well"].map(fold_of).to_numpy()
splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]

results = {}
for seed in SEEDS:
    print(f"\n>> CatBoost(gpu) seed={seed}", flush=True)
    oof = np.zeros(len(train_df), np.float32)
    test_pred = np.zeros(len(test_df), np.float32)
    for fold, (tr, va) in enumerate(splits):
        m = CatBoostRegressor(**cb_params(seed))
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

json.dump(results, open(MODELS / "cat_extra_summary.json", "w"), indent=2)
print(f"\n=== SUPER CAT EXTRA DONE: {results} ===", flush=True)
