"""GP anchor: full combined gate (the discipline the extractor failure taught).

End-to-end, single job (because tool I/O here is heavily delayed -> minimize round trips):
 1. Load GP per-row features (gp_feats_{train,test}.parquet); build them if missing
    by importing experiments.gp_feature_build.
 2. Merge onto the cached konbu matrices by (well,row_idx); ASSERT the join reproduces
    the konbu `gr` column (fail loud if the key is wrong).
 3. Derive GP features: gp_drift, gp_std, gp_ancc, gp_vs_fk.
 4. SOLO ablation: single GPU-LGB, BASE 78 vs BASE+GP, same GKF-5 (seed 42).
    Cheap negative filter (the extractor showed solo OVER-credits, so a null here = stop).
 5. FULL 5-model stack (LGBx3+XGB+CatBoost) on BASE+GP, Ridge -> combined OOF vs banked 11.821.
    THIS is the real gate (extractor lesson: solo lift != stack lift).
 6. Write /tmp/gp_gate_result.txt with both numbers + verdict.

Artifacts isolated under data/processed/konbu_gp/ and models/konbu_gp/ (banked konbu/ untouched).

Run: nohup python -u experiments/gp_gate.py > log/gp_gate_$(date +%Y%m%d_%H%M%S).log 2>&1 &
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
ARTG = ROOT / "data/processed/konbu_gp"
MODELS = ROOT / "models/konbu"
MODELSG = ROOT / "models/konbu_gp"
RESULT = Path("/tmp/gp_gate_result.txt")
N_SPLITS, SPLIT_SEED = 5, 42
LGB_SEEDS = [42, 7, 123]
BANKED = 11.8212
GP_FEATS = ["gp_drift", "gp_std", "gp_ancc", "gp_vs_fk"]

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

_lines = []


def say(s):
    print(s, flush=True)
    _lines.append(s)
    RESULT.write_text("\n".join(_lines) + "\n")


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_splits(df):
    rng = np.random.RandomState(SPLIT_SEED)
    wells = df["well"].unique().copy(); rng.shuffle(wells)
    fold_of = {w: i % N_SPLITS for i, w in enumerate(wells)}
    wf = df["well"].map(fold_of).to_numpy()
    return [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]


def ensure_gp_feats():
    p = ART / "gp_feats_train.parquet"
    assert p.exists(), f"missing {p} -- run experiments/gp_feature_build.py first"


def merge_gp(base, gp, label):
    """Merge GP per-row feats onto base konbu matrix by (well,row_idx); validate join."""
    g = gp[["well", "row_idx", "gr", "gp_ancc", "gp_std", "gp_tvt_abs"]].rename(
        columns={"gr": "gr_gp"})
    m = base.merge(g, on=["well", "row_idx"], how="left")
    miss = m["gp_ancc"].isna().sum()
    say(f"   {label}: merged {m.shape}, {miss} rows missing GP")
    # join validation: GR must match the konbu matrix where both present
    both = m["gr"].notna() & m["gr_gp"].notna()
    if both.any():
        d = np.abs(m.loc[both, "gr"].to_numpy(np.float64) - m.loc[both, "gr_gp"].to_numpy(np.float64))
        say(f"   {label}: GR join check on {int(both.sum())} rows -> max|dgr|={np.nanmax(d):.4g} mean={np.nanmean(d):.4g}")
        assert np.nanmax(d) < 1e-2, f"{label} JOIN KEY WRONG (GR mismatch {np.nanmax(d)})"
    # derive features
    m["gp_drift"] = m["gp_tvt_abs"].to_numpy(np.float32) - m["last_known_tvt"].to_numpy(np.float32)
    m["gp_vs_fk"] = m["gp_drift"].to_numpy(np.float32) - m["fk_tvt_formula"].to_numpy(np.float32)
    for c in GP_FEATS:
        m[c] = m[c].fillna(0.0).astype(np.float32)
    return m.drop(columns=["gr_gp"])


def main():
    ARTG.mkdir(parents=True, exist_ok=True)
    MODELSG.mkdir(parents=True, exist_ok=True)
    say("=== GP COMBINED GATE ===")

    ensure_gp_feats()

    say(">> load cached konbu matrices + GP feats")
    tr = pd.read_parquet(ART / "train_feats.parquet")
    te = pd.read_parquet(ART / "test_feats.parquet")
    gtr = pd.read_parquet(ART / "gp_feats_train.parquet")
    gte = pd.read_parquet(ART / "gp_feats_test.parquet") if (ART / "gp_feats_test.parquet").exists() else None

    tr = merge_gp(tr, gtr, "train")
    if gte is not None:
        te = merge_gp(te, gte, "test")
        te.to_parquet(ARTG / "test_feats.parquet")

    base_feats = json.load(open(MODELS / "feature_cols.json"))
    feat_cols = base_feats + GP_FEATS
    json.dump(feat_cols, open(MODELSG / "feature_cols.json", "w"))
    say(f">> {len(base_feats)} base + {len(GP_FEATS)} GP = {len(feat_cols)} feats")
    say(f"   GP feats: {GP_FEATS}")

    y = tr["target"].to_numpy()
    splits = make_splits(tr)

    # ---------- SOLO ablation (cheap negative filter) ----------
    say("\n>> SOLO LGB ablation (seed 42): BASE vs BASE+GP")

    def solo(cols, tag):
        oof = np.zeros(len(tr), np.float32)
        Xs = tr[cols]
        p = dict(LGB_PARAMS); p["random_state"] = 42
        for f, (a, v) in enumerate(splits):
            dtr = lgb.Dataset(Xs.iloc[a], label=y[a])
            dva = lgb.Dataset(Xs.iloc[v], label=y[v], reference=dtr)
            m = lgb.train(p, dtr, valid_sets=[dva], num_boost_round=p["n_estimators"],
                          callbacks=[lgb.early_stopping(125, verbose=False)])
            oof[v] = m.predict(Xs.iloc[v], num_iteration=m.best_iteration)
        r = rmse(oof, y)
        say(f"   solo {tag}: OOF={r:.4f}")
        return r

    base_solo = solo(base_feats, "BASE-78")
    gp_solo = solo(feat_cols, "BASE+GP")
    say(f">>> SOLO delta (BASE - BASE+GP) = {base_solo - gp_solo:+.4f} ft")

    # ---------- FULL 5-model stack (the real gate) ----------
    say("\n>> FULL 5-model stack on BASE+GP")
    Xtr, Xte = tr[feat_cols], (te[feat_cols] if gte is not None else None)
    Xtr_np = Xtr.to_numpy(np.float32)
    Xte_np = Xte.to_numpy(np.float32) if Xte is not None else None
    results = {}

    for seed in LGB_SEEDS:
        oof = np.zeros(len(tr), np.float32); tp = np.zeros(len(te), np.float32) if Xte is not None else None
        p = dict(LGB_PARAMS); p["random_state"] = seed
        for f, (a, v) in enumerate(splits):
            dtr = lgb.Dataset(Xtr.iloc[a], label=y[a])
            dva = lgb.Dataset(Xtr.iloc[v], label=y[v], reference=dtr)
            m = lgb.train(p, dtr, valid_sets=[dva], num_boost_round=p["n_estimators"],
                          callbacks=[lgb.early_stopping(125, verbose=False)])
            oof[v] = m.predict(Xtr.iloc[v], num_iteration=m.best_iteration)
            if tp is not None:
                tp += m.predict(Xte, num_iteration=m.best_iteration) / N_SPLITS
            m.save_model(str(MODELSG / f"lgb_seed{seed}_fold{f}.txt"), num_iteration=m.best_iteration)
        results[f"lgb_{seed}"] = dict(oof=oof, test=tp, rmse=rmse(oof, y))
        say(f"   lgb_{seed} OOF={results[f'lgb_{seed}']['rmse']:.4f}")

    # xgb
    oof = np.zeros(len(tr), np.float32); tp = np.zeros(len(te), np.float32) if Xte is not None else None
    dte = xgb.DMatrix(Xte_np) if Xte_np is not None else None
    p = dict(XGB_PARAMS); p["seed"] = 42
    for f, (a, v) in enumerate(splits):
        dtr = xgb.DMatrix(Xtr_np[a], label=y[a]); dva = xgb.DMatrix(Xtr_np[v], label=y[v])
        m = xgb.train(p, dtr, num_boost_round=5000, evals=[(dva, "v")],
                      early_stopping_rounds=125, verbose_eval=False)
        it = (0, m.best_iteration + 1)
        oof[v] = m.predict(dva, iteration_range=it)
        if tp is not None:
            tp += m.predict(dte, iteration_range=it) / N_SPLITS
        m.save_model(str(MODELSG / f"xgb_seed42_fold{f}.json"))
    results["xgb_42"] = dict(oof=oof, test=tp, rmse=rmse(oof, y))
    say(f"   xgb_42 OOF={results['xgb_42']['rmse']:.4f}")

    # cat
    oof = np.zeros(len(tr), np.float32); tp = np.zeros(len(te), np.float32) if Xte is not None else None
    for f, (a, v) in enumerate(splits):
        m = CatBoostRegressor(**CAT_PARAMS)
        m.fit(Pool(Xtr_np[a], y[a]), eval_set=Pool(Xtr_np[v], y[v]), use_best_model=True)
        oof[v] = m.predict(Xtr_np[v])
        if tp is not None:
            tp += m.predict(Xte_np) / N_SPLITS
        m.save_model(str(MODELSG / f"cat_seed42_fold{f}.cbm"))
    results["cat_42"] = dict(oof=oof, test=tp, rmse=rmse(oof, y))
    say(f"   cat_42 OOF={results['cat_42']['rmse']:.4f}")

    keys = ["lgb_42", "lgb_7", "lgb_123", "xgb_42", "cat_42"]
    Xstack = np.column_stack([results[k]["oof"] for k in keys])
    ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(Xstack, y)
    rmse_stack = rmse(ridge.predict(Xstack), y)
    weights = dict(zip(keys, np.round(ridge.coef_, 4)))

    say("\n=== RESULT ===")
    for k in keys:
        say(f"  {k}: {results[k]['rmse']:.4f}")
    say(f"  ridge stack (BASE+GP): {rmse_stack:.4f}  weights={weights}")
    say(f"  banked CatBoost-only : {BANKED:.4f}")
    say(f">>> COMBINED gate delta (banked - new) = {BANKED - rmse_stack:+.4f} ft")
    say(f">>> solo delta = {base_solo - gp_solo:+.4f} ft")
    if rmse_stack < BANKED - 0.005:
        say(">>> VERDICT: GP HELPS in-stack -> productionize kernel + submit.")
    else:
        say(">>> VERDICT: GP does NOT beat banked in-stack -> shelve konbu_gp (extractor pattern).")

    json.dump({"keys": keys, "ridge_coef": ridge.coef_.tolist(), "rmse_stack": rmse_stack,
               "banked": BANKED, "solo_base": base_solo, "solo_gp": gp_solo,
               "gp_feats": GP_FEATS, "per_model": {k: results[k]["rmse"] for k in keys}},
              open(MODELSG / "blend_catboost.json", "w"), indent=2)
    if Xte is not None:
        stack_test = ridge.predict(np.column_stack([results[k]["test"] for k in keys]))
        sub = te["last_known_tvt"].to_numpy(float) + stack_test.astype(float)
        pd.DataFrame({"id": te["prediction_id"], "tvt": sub}).to_csv(ARTG / "submission_local.csv", index=False)
    say("=== DONE ===")


if __name__ == "__main__":
    main()
