"""Lane B: add CatBoost as a 5th model to the konbu LGBx3+XGB Ridge stack.

Reuses the saved fold models (predict-only, deterministic) to reconstruct each
existing model's OOF on the cached feature matrix, trains CatBoost on the SAME
shuffled GroupKFold-5 (seed 42), then re-fits the non-negative Ridge stack over
{lgb_42, lgb_7, lgb_123, xgb_42, cat_42} and reports rmse_stack vs the banked
11.885. A clear improvement => productionize; ~0 / negative weight => CatBoost
adds nothing on this base (document and drop).

GATE: rmse_stack < 11.875 (i.e. > ~0.01 ft over the banked 11.8853) to bother
wiring CatBoost into the inference kernel.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge

ROOT = Path("/home/swatson/work/kaggle/ROGII")
ART = ROOT / "data/processed/konbu"
MODELS = ROOT / "models/konbu"
N_SPLITS, SPLIT_SEED = 5, 42
LGB_SEEDS = [42, 7, 123]

CAT_PARAMS = dict(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=5000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=20.0,
    random_strength=1.0,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    od_type="Iter",
    od_wait=125,
    task_type="GPU",
    devices="0",
    verbose=0,
)


def make_splits(train_df):
    rng = np.random.RandomState(SPLIT_SEED)
    wells = train_df["well"].unique().copy(); rng.shuffle(wells)
    fold_of = {w: i % N_SPLITS for i, w in enumerate(wells)}
    well_fold = train_df["well"].map(fold_of).to_numpy()
    return [(np.where(well_fold != f)[0], np.where(well_fold == f)[0]) for f in range(N_SPLITS)]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    print(">> load cached feats", flush=True)
    train_df = pd.read_parquet(ART / "train_feats.parquet")
    test_df = pd.read_parquet(ART / "test_feats.parquet")
    feat_cols = json.load(open(MODELS / "feature_cols.json"))
    print(f"   train {train_df.shape}  test {test_df.shape}  #feats {len(feat_cols)}", flush=True)
    y = train_df["target"].to_numpy()
    splits = make_splits(train_df)
    Xtr_all = train_df[feat_cols]
    Xte_all = test_df[feat_cols]

    results = {}

    # --- reconstruct existing model OOF (predict-only from saved fold models) ---
    for seed in LGB_SEEDS:
        oof = np.zeros(len(train_df), np.float32); tp = np.zeros(len(test_df), np.float32)
        for fold, (tr, va) in enumerate(splits):
            m = lgb.Booster(model_file=str(MODELS / f"lgb_seed{seed}_fold{fold}.txt"))
            oof[va] = m.predict(Xtr_all.iloc[va])
            tp += m.predict(Xte_all) / N_SPLITS
        results[f"lgb_{seed}"] = {"oof": oof, "test": tp, "rmse": rmse(oof, y)}
        print(f"   lgb_{seed} reconstructed OOF rmse={results[f'lgb_{seed}']['rmse']:.4f}", flush=True)

    # xgb
    oof = np.zeros(len(train_df), np.float32); tp = np.zeros(len(test_df), np.float32)
    dte = xgb.DMatrix(Xte_all.values)
    for fold, (tr, va) in enumerate(splits):
        m = xgb.Booster(); m.load_model(str(MODELS / f"xgb_seed42_fold{fold}.json"))
        oof[va] = m.predict(xgb.DMatrix(Xtr_all.iloc[va].values))
        tp += m.predict(dte) / N_SPLITS
    results["xgb_42"] = {"oof": oof, "test": tp, "rmse": rmse(oof, y)}
    print(f"   xgb_42 reconstructed OOF rmse={results['xgb_42']['rmse']:.4f}", flush=True)

    # sanity: do the reconstructed OOFs reproduce the banked stack?
    banked = json.load(open(MODELS / "blend.json"))
    base_keys = banked["keys"]
    Xbase = np.column_stack([results[k]["oof"] for k in base_keys])
    coef = np.array(banked["ridge_coef"])
    repro = rmse(Xbase @ coef, y)
    print(f"   reproduced banked stack rmse={repro:.4f} (banked {banked['rmse_stack']:.4f})", flush=True)

    # --- train CatBoost on same splits ---
    print(f"\n>> train CatBoost (GPU) over {N_SPLITS} folds", flush=True)
    oof = np.zeros(len(train_df), np.float32); tp = np.zeros(len(test_df), np.float32)
    Xtr_np = Xtr_all.to_numpy(np.float32)
    Xte_np = Xte_all.to_numpy(np.float32)
    for fold, (tr, va) in enumerate(splits):
        model = CatBoostRegressor(**CAT_PARAMS)
        model.fit(Pool(Xtr_np[tr], y[tr]), eval_set=Pool(Xtr_np[va], y[va]),
                  use_best_model=True)
        oof[va] = model.predict(Xtr_np[va])
        tp += model.predict(Xte_np) / N_SPLITS
        model.save_model(str(MODELS / f"cat_seed42_fold{fold}.cbm"))
        print(f"   fold {fold}: rmse={rmse(oof[va], y[va]):.4f} best_it={model.get_best_iteration()}", flush=True)
    results["cat_42"] = {"oof": oof, "test": tp, "rmse": rmse(oof, y)}
    print(f"   CatBoost OOF rmse={results['cat_42']['rmse']:.4f}", flush=True)

    # --- re-fit Ridge stack with CatBoost added ---
    all_keys = base_keys + ["cat_42"]
    Xstack = np.column_stack([results[k]["oof"] for k in all_keys])
    ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(Xstack, y)
    stack_oof = ridge.predict(Xstack)
    rmse_new = rmse(stack_oof, y)
    weights = dict(zip(all_keys, np.round(ridge.coef_, 4)))
    print(f"\n=== STACK COMPARISON ===", flush=True)
    print(f"  per-model: " + ", ".join(f"{k}={results[k]['rmse']:.4f}" for k in all_keys), flush=True)
    print(f"  banked 4-model stack : {banked['rmse_stack']:.4f}", flush=True)
    print(f"  +CatBoost 5-model    : {rmse_new:.4f}  (delta {banked['rmse_stack']-rmse_new:+.4f} ft)", flush=True)
    print(f"  ridge weights        : {weights}", flush=True)
    print(f"\n>>> GATE: improvement > +0.01 ft => productionize CatBoost; cat weight ~0 => drop", flush=True)

    json.dump({"keys": all_keys, "ridge_coef": ridge.coef_.tolist(),
               "rmse_stack": rmse_new, "banked_stack": banked["rmse_stack"],
               "cat_params": CAT_PARAMS,
               "per_model": {k: results[k]["rmse"] for k in all_keys}},
              open(MODELS / "blend_catboost.json", "w"), indent=2)
    print(">> wrote models/konbu/blend_catboost.json", flush=True)


if __name__ == "__main__":
    main()
