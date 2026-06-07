"""HPO quick-win (parallel track): Optuna-tune CatBoost on the konbu base matrix.

konbu's CatBoost params were inherited/generic, never tuned for our data, yet CatBoost
carries the largest banked-blend weight (0.386). Its current solo OOF is 12.027. Tune
depth/lr/l2/random_strength/bagging on the SAME shuffled GKF-5 (seed 42, matches konbu_prod
so OOF is comparable), minimize 5-fold OOF RMSE. If solo OOF drops well below 12.027, the
tuned params propagate into the stack (Phase B re-blends tuned-cat OOF with lgb/xgb).

Runs on deepthought (RTX 4080, GPU CatBoost). Output: models/konbu/cat_hpo.json
Run on deepthought: nohup python -u experiments/hpo_catboost.py > log/hpo_cat_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor, Pool

ROOT = Path("/home/swatson/work/kaggle/ROGII")
N_TRIALS = 50
N_SPLITS = 5

df = pd.read_parquet(ROOT / "data/processed/konbu/train_feats.parquet")
feat = [c for c in df.columns if c not in {"well", "target", "id", "prediction_id"}]
X = df[feat].to_numpy(np.float32)
y = df["target"].to_numpy(np.float32)
print(f">> {X.shape[0]:,} rows x {len(feat)} feats; current CatBoost solo OOF = 12.027", flush=True)

# shuffled GroupKFold-5, seed 42 (identical to konbu_prod.main)
rng = np.random.RandomState(42)
uw = df["well"].unique().copy()
rng.shuffle(uw)
fold_of = {w: i % N_SPLITS for i, w in enumerate(uw)}
wf = df["well"].map(fold_of).to_numpy()
splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]


def objective(trial):
    params = dict(
        iterations=3000, task_type="GPU", devices="0",
        loss_function="RMSE", eval_metric="RMSE",
        depth=trial.suggest_int("depth", 5, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 60.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        border_count=254, od_type="Iter", od_wait=100,
        random_seed=42, verbose=0)
    oof = np.zeros(len(df), np.float32)
    for tr, va in splits:
        m = CatBoostRegressor(**params)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              use_best_model=True, verbose=0)
        oof[va] = m.predict(X[va])
    score = float(np.sqrt(np.mean((oof - y) ** 2)))
    trial.set_user_attr("oof_rmse", score)
    return score


study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print(f"\n>> BEST OOF={study.best_value:.4f}  (baseline 12.027, delta {study.best_value-12.027:+.4f})", flush=True)
print(f">> params: {study.best_params}", flush=True)
json.dump({"best_value": study.best_value, "best_params": study.best_params,
           "baseline": 12.027, "n_trials": N_TRIALS},
          open(ROOT / "models/konbu/cat_hpo.json", "w"), indent=2)
print("=== CAT HPO DONE ===", flush=True)
