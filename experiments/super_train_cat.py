"""Super-solution train (part 2): GPU-CatBoost x1 (seed 42) on the ~170-feat super union.

The super-solution trains CatBoost ONCE (single seed) -> the stack is LGB×3 + CatBoost×1 = 4
models. Uses the super-solution's TUNED CatBoost params (depth=7, lr=0.025, l2=2, min_data=15,
subsample=0.75, border=254, 8000it od_wait=300). Same shuffled GKF-5 seed42 as super_train_lgb.

Runs on a single GPU (devices="0"). Default machine = deepthought (CLAUDE.md GPU routing); the
super matrix must be present at the same repo path there. Saves OOF + test preds -> models/super/.

Run (deepthought): nohup python -u experiments/super_train_cat.py > log/super_cat_$(date +%Y%m%d_%H%M%S).log 2>&1 &
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
SEED = 42

CB_PARAMS = dict(
    iterations=8000, learning_rate=0.025, depth=7, l2_leaf_reg=2.0,
    min_data_in_leaf=15, subsample=0.75, bootstrap_type="Bernoulli", border_count=254,
    loss_function="RMSE", eval_metric="RMSE", task_type="GPU", devices="0",
    od_type="Iter", od_wait=300, random_seed=SEED, verbose=0)

train_df = pd.read_parquet(SP / "train_feats.parquet")
test_df = pd.read_parquet(SP / "test_feats.parquet")
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

print(f"\n>> CatBoost(gpu) seed={SEED}", flush=True)
oof = np.zeros(len(train_df), np.float32)
test_pred = np.zeros(len(test_df), np.float32)
for fold, (tr, va) in enumerate(splits):
    m = CatBoostRegressor(**CB_PARAMS)
    m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]), use_best_model=True, verbose=0)
    oof[va] = m.predict(X[va])
    test_pred += m.predict(Xt) / N_SPLITS
    m.save_model(str(MODELS / f"cat_seed{SEED}_fold{fold}.cbm"))
    print(f"   fold {fold}: rmse={np.sqrt(np.mean((oof[va]-y[va])**2)):.4f} it={m.get_best_iteration()}", flush=True)
rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
print(f"   CatBoost seed={SEED}: OOF rmse={rmse:.4f}", flush=True)
np.save(MODELS / f"oof_cat_{SEED}.npy", oof)
np.save(MODELS / f"test_cat_{SEED}.npy", test_pred)

json.dump({f"cat_{SEED}": rmse}, open(MODELS / "cat_summary.json", "w"), indent=2)
print("=== SUPER CAT DONE ===", flush=True)
