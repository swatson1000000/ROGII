"""Productionize the KNN-seeded GR extractor (+0.065 solo) ON TOP of CatBoost.

Step 1 of productionization = the COMBINED-gain gate. Builds the 16 extractor
features for train + test, augments the cached konbu matrices (78 -> 94 feats),
retrains the FULL 5-model stack (LGBx3 + XGB + CatBoost) on the same shuffled
GKF-5 (seed 42), Ridge-stacks, and reports OOF vs the banked 11.821.

GATE: combined OOF clearly below 11.821 (extractor adds on top of CatBoost).
If it lands ~11.82 (no additive gain), the extractor is redundant with the
existing stack diversity -> do NOT do the kernel surgery; document and stop.

Saves augmented matrices to data/processed/konbu_v2/ and models to
models/konbu_v2/ (leaves the banked konbu/ artifacts that produced LB 11.903
untouched). Runs entirely on skynet (one GPU): the build is CPU/IO-parallel and
a single 5-model stack can't usefully split across two GPUs (CLAUDE.md routing).
"""
from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from concurrent.futures import ProcessPoolExecutor

ROOT = Path("/home/swatson/work/kaggle/ROGII")
ART = ROOT / "data/processed/konbu"
ART2 = ROOT / "data/processed/konbu_v2"
MODELS = ROOT / "models/konbu"
MODELS2 = ROOT / "models/konbu_v2"
N_SPLITS, SPLIT_SEED = 5, 42
LGB_SEEDS = [42, 7, 123]
OFF_GRID = np.arange(-20.0, 20.01, 0.5, dtype=np.float32)
DIFF_OFFS = np.array([-20, -15, -10, -7, -5, -3, 0, 3, 5, 7, 10, 15, 20], dtype=np.float32)

LGB_PARAMS = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89, min_child_samples=10,
                  min_child_weight=0.5, n_estimators=5000, n_jobs=-1, reg_alpha=2.03, reg_lambda=87.28,
                  subsample=0.645, subsample_freq=1, colsample_bytree=0.821, objective="regression",
                  metric="rmse", verbose=-1, device_type="cuda", max_bin=255)
XGB_PARAMS = dict(objective="reg:squarederror", eval_metric="rmse", learning_rate=0.06, max_depth=8,
                  min_child_weight=10, subsample=0.7, colsample_bytree=0.85, reg_alpha=1.0,
                  reg_lambda=20.0, tree_method="hist", device="cuda", n_jobs=-1)
CAT_PARAMS = dict(loss_function="RMSE", eval_metric="RMSE", iterations=5000, learning_rate=0.03,
                  depth=7, l2_leaf_reg=20.0, random_strength=1.0, bootstrap_type="Bernoulli",
                  subsample=0.8, od_type="Iter", od_wait=125, task_type="GPU", devices="0", verbose=0)

# wid -> typewell path, built once in main, inherited by workers via fork
_TW = {}


def build_extractor(args):
    """Build the 16 KNN-seeded GR features for one well. args=(wid, sub_df)."""
    wid, sub = args
    tp = _TW.get(wid)
    if tp is None:
        return None
    t = pd.read_csv(tp)
    if "TVT" not in t.columns or "GR" not in t.columns:
        return None
    tw = t[["TVT", "GR"]].dropna().sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 8:
        return None
    sub = sub.sort_values("row_idx")
    gr = sub["gr"].to_numpy(np.float32)
    knn_abs = sub["knn_abs"].to_numpy(np.float32)
    gr = np.where(np.isnan(gr), np.float32(np.nanmean(tw_gr)), gr)
    out = pd.DataFrame({"prediction_id": sub["prediction_id"].to_numpy()})
    for o in DIFF_OFFS:
        samp = np.interp(knn_abs + o, tw_tvt, tw_gr).astype(np.float32)
        out[f"knn_diff_{int(o)}"] = (gr - samp).astype(np.float32)
    query = knn_abs[:, None] + OFF_GRID[None, :]
    samp = np.interp(query.ravel(), tw_tvt, tw_gr).reshape(query.shape).astype(np.float32)
    cost = np.abs(gr[:, None] - samp)
    j = cost.argmin(axis=1)
    out["knn_refined_drift"] = OFF_GRID[j].astype(np.float32)
    out["knn_refined_cost"] = cost[np.arange(len(j)), j].astype(np.float32)
    out["knn_seed_resid"] = np.abs(gr - np.interp(knn_abs, tw_tvt, tw_gr)).astype(np.float32)
    return out


def augment(df, label):
    df = df.copy()
    df["knn_abs"] = (df["last_known_tvt"].to_numpy(np.float32)
                     + df["fk_tvt_formula"].to_numpy(np.float32))
    groups = [(w, g[["prediction_id", "row_idx", "gr", "knn_abs"]].copy())
              for w, g in df.groupby("well", sort=False)]
    parts = []
    with ProcessPoolExecutor(max_workers=14) as ex:
        for i, r in enumerate(ex.map(build_extractor, groups, chunksize=4)):
            if r is not None:
                parts.append(r)
            if (i + 1) % 150 == 0:
                print(f"   {label}: {i+1}/{len(groups)}", flush=True)
    new = pd.concat(parts, ignore_index=True)
    new_feats = [c for c in new.columns if c != "prediction_id"]
    m = df.drop(columns=["knn_abs"]).merge(new, on="prediction_id", how="left")
    m[new_feats] = m[new_feats].fillna(0.0)
    print(f"   {label} augmented: {m.shape}  (+{len(new_feats)} feats, "
          f"{m[new_feats[0]].isna().sum()} missing)", flush=True)
    return m, new_feats


def make_splits(train_df):
    rng = np.random.RandomState(SPLIT_SEED)
    wells = train_df["well"].unique().copy(); rng.shuffle(wells)
    fold_of = {w: i % N_SPLITS for i, w in enumerate(wells)}
    wf = train_df["well"].map(fold_of).to_numpy()
    return [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    global _TW
    ART2.mkdir(parents=True, exist_ok=True)
    MODELS2.mkdir(parents=True, exist_ok=True)

    # typewell map (train + test dirs)
    for d in ("train", "test"):
        for p in glob.glob(str(ROOT / f"data/raw/{d}/*__typewell.csv")):
            wid = Path(p).name.replace("__typewell.csv", "")
            _TW[wid] = Path(p)
    print(f">> {len(_TW)} typewells mapped", flush=True)

    tr_cache, te_cache = ART2 / "train_feats.parquet", ART2 / "test_feats.parquet"
    if tr_cache.exists() and te_cache.exists():
        print(">> loading cached AUGMENTED matrices", flush=True)
        train_df = pd.read_parquet(tr_cache); test_df = pd.read_parquet(te_cache)
        new_feats = [c for c in train_df.columns if c.startswith("knn_diff_")
                     or c in ("knn_refined_drift", "knn_refined_cost", "knn_seed_resid")]
    else:
        print(">> load base konbu matrices", flush=True)
        base_tr = pd.read_parquet(ART / "train_feats.parquet")
        base_te = pd.read_parquet(ART / "test_feats.parquet")
        print(">> build extractor features (train)", flush=True)
        train_df, new_feats = augment(base_tr, "train")
        print(">> build extractor features (test)", flush=True)
        test_df, _ = augment(base_te, "test")
        train_df.to_parquet(tr_cache); test_df.to_parquet(te_cache)
        print(f"   cached -> {tr_cache}", flush=True)

    base_feats = json.load(open(MODELS / "feature_cols.json"))
    feat_cols = base_feats + new_feats
    json.dump(feat_cols, open(MODELS2 / "feature_cols.json", "w"))
    print(f">> {len(base_feats)} base + {len(new_feats)} extractor = {len(feat_cols)} feats", flush=True)

    y = train_df["target"].to_numpy()
    splits = make_splits(train_df)
    Xtr, Xte = train_df[feat_cols], test_df[feat_cols]
    Xtr_np, Xte_np = Xtr.to_numpy(np.float32), Xte.to_numpy(np.float32)
    results = {}

    def fit_lgb(seed):
        print(f">> LGB seed={seed}", flush=True)
        p = dict(LGB_PARAMS); p["random_state"] = seed
        oof = np.zeros(len(train_df), np.float32); tp = np.zeros(len(test_df), np.float32)
        for f, (tr, va) in enumerate(splits):
            dtr = lgb.Dataset(Xtr.iloc[tr], label=y[tr])
            dva = lgb.Dataset(Xtr.iloc[va], label=y[va], reference=dtr)
            m = lgb.train(p, dtr, valid_sets=[dva], num_boost_round=p["n_estimators"],
                          callbacks=[lgb.early_stopping(125, verbose=False)])
            oof[va] = m.predict(Xtr.iloc[va], num_iteration=m.best_iteration)
            tp += m.predict(Xte, num_iteration=m.best_iteration) / N_SPLITS
            m.save_model(str(MODELS2 / f"lgb_seed{seed}_fold{f}.txt"), num_iteration=m.best_iteration)
        results[f"lgb_{seed}"] = dict(oof=oof, test=tp, rmse=rmse(oof, y))
        print(f"   lgb_{seed} OOF={results[f'lgb_{seed}']['rmse']:.4f}", flush=True)

    def fit_xgb(seed=42):
        print(f">> XGB seed={seed}", flush=True)
        p = dict(XGB_PARAMS); p["seed"] = seed
        oof = np.zeros(len(train_df), np.float32); tp = np.zeros(len(test_df), np.float32)
        dte = xgb.DMatrix(Xte_np)
        for f, (tr, va) in enumerate(splits):
            dtr = xgb.DMatrix(Xtr_np[tr], label=y[tr]); dva = xgb.DMatrix(Xtr_np[va], label=y[va])
            m = xgb.train(p, dtr, num_boost_round=5000, evals=[(dva, "v")],
                          early_stopping_rounds=125, verbose_eval=False)
            it = (0, m.best_iteration + 1)
            oof[va] = m.predict(dva, iteration_range=it)
            tp += m.predict(dte, iteration_range=it) / N_SPLITS
            m.save_model(str(MODELS2 / f"xgb_seed{seed}_fold{f}.json"))
        results[f"xgb_{seed}"] = dict(oof=oof, test=tp, rmse=rmse(oof, y))
        print(f"   xgb_{seed} OOF={results[f'xgb_{seed}']['rmse']:.4f}", flush=True)

    def fit_cat(seed=42):
        print(f">> CatBoost seed={seed}", flush=True)
        oof = np.zeros(len(train_df), np.float32); tp = np.zeros(len(test_df), np.float32)
        for f, (tr, va) in enumerate(splits):
            m = CatBoostRegressor(**CAT_PARAMS)
            m.fit(Pool(Xtr_np[tr], y[tr]), eval_set=Pool(Xtr_np[va], y[va]), use_best_model=True)
            oof[va] = m.predict(Xtr_np[va]); tp += m.predict(Xte_np) / N_SPLITS
            m.save_model(str(MODELS2 / f"cat_seed{seed}_fold{f}.cbm"))
        results[f"cat_{seed}"] = dict(oof=oof, test=tp, rmse=rmse(oof, y))
        print(f"   cat_{seed} OOF={results[f'cat_{seed}']['rmse']:.4f}", flush=True)

    for s in LGB_SEEDS:
        fit_lgb(s)
    fit_xgb(42)
    fit_cat(42)

    keys = ["lgb_42", "lgb_7", "lgb_123", "xgb_42", "cat_42"]
    Xstack = np.column_stack([results[k]["oof"] for k in keys])
    ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(Xstack, y)
    rmse_stack = rmse(ridge.predict(Xstack), y)
    rmse_avg = rmse(Xstack.mean(1), y)
    print("\n=== RESULT ===", flush=True)
    for k in keys:
        print(f"  {k}: {results[k]['rmse']:.4f}", flush=True)
    print(f"  simple avg : {rmse_avg:.4f}", flush=True)
    print(f"  ridge stack: {rmse_stack:.4f}  weights={dict(zip(keys, np.round(ridge.coef_,4)))}", flush=True)
    print(f"\n>>> banked CatBoost-only stack = 11.8212", flush=True)
    print(f">>> +extractor combined        = {rmse_stack:.4f}  (delta {11.8212-rmse_stack:+.4f} ft)", flush=True)
    print(f">>> GATE: clearly below 11.821 => productionize kernel; ~11.82 => redundant, stop", flush=True)

    stack_test = ridge.predict(np.column_stack([results[k]["test"] for k in keys]))
    json.dump({"keys": keys, "ridge_coef": ridge.coef_.tolist(), "rmse_stack": rmse_stack,
               "rmse_avg": rmse_avg, "banked_catboost_stack": 11.8212,
               "extractor_feats": new_feats, "cat_params": CAT_PARAMS,
               "per_model": {k: results[k]["rmse"] for k in keys}},
              open(MODELS2 / "blend_catboost.json", "w"), indent=2)
    sub = test_df["last_known_tvt"].to_numpy(float) + stack_test.astype(float)
    pd.DataFrame({"id": test_df["prediction_id"], "tvt": sub}).to_csv(ART2 / "submission_local.csv", index=False)
    print(f">> saved models -> {MODELS2}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
