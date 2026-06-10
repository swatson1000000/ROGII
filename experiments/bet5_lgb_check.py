"""BET 5 Stage B (2/4): cheap single-LGB OOF check -- do the 3 UK features REGRESS the GBM?

The OOF is GroupKFold-by-well = INTERPOLATION regime, which physically cannot see UK1's OOD-tail
benefit ([[bet5-uk1-ood-robust-but-test-interpolates]]). So the expectation is ~FLAT. This check is
NOT looking for a gain -- it's a guard: if adding uk_tvt_d/uk_vs_dense/uk_ancc HURTS the OOF (tree
overfits the new cols), stop before the 6-model retrain. Flat-or-better -> proceed.

Trains LGB seed42 (konbu params, GKF-5 seed42) on 222 feats vs 225 (+UK), same folds. Also writes the
merged matrix data/processed/frontier_uk/train_feats.parquet for the full retrain (3/4).

Run: nohup python -u experiments/bet5_lgb_check.py > log/bet5_lgbchk_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
FR = ROOT / "data/processed/frontier_seeded"
OUTDIR = ROOT / "data/processed/frontier_uk"; OUTDIR.mkdir(parents=True, exist_ok=True)
UK = ROOT / "data/processed/uk_feats.parquet"
UK_COLS = ["tvt_uk_d", "uk_ancc", "uk_vs_dense"]
N_SPLITS, SPLIT_SEED = 5, 42
LGB_PARAMS = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
    min_child_samples=10, min_child_weight=0.5, n_estimators=5000, n_jobs=-1,
    reg_alpha=2.03, reg_lambda=87.28, subsample=0.645, subsample_freq=1,
    colsample_bytree=0.821, objective="regression", metric="rmse", verbose=-1,
    device_type="cuda", max_bin=255, random_state=42)

print(">> load + merge UK feats...", flush=True)
tr = pd.read_parquet(FR / "train_feats.parquet")
uk = pd.read_parquet(UK)
tr = tr.merge(uk, on="id", how="left")
assert all(c in tr.columns for c in UK_COLS), "UK merge failed"
print(f"   merged {tr.shape}", flush=True)
tr.to_parquet(OUTDIR / "train_feats.parquet")
# also merge UK into the 3-well test matrix for consistency (Stage B 3/4 test_pred)
te = pd.read_parquet(FR / "test_feats.parquet")
te = te.merge(uk, on="id", how="left")
te.to_parquet(OUTDIR / "test_feats.parquet")
print(f"   test merged {te.shape} (UK NaN in test: {int(te['tvt_uk_d'].isna().sum())})", flush=True)

base_cols = [c for c in tr.columns if c not in {"well", "id", "target"} and c not in UK_COLS]
uk_cols = base_cols + UK_COLS
y = tr["target"].to_numpy(np.float32)
rng = np.random.RandomState(SPLIT_SEED); uw = tr["well"].unique().copy(); rng.shuffle(uw)
fold_of = {w: i % N_SPLITS for i, w in enumerate(uw)}
wf = tr["well"].map(fold_of).to_numpy()
splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]


def run(cols, tag):
    oof = np.zeros(len(tr), np.float32)
    for fold, (a, b) in enumerate(splits):
        dtr = lgb.Dataset(tr.iloc[a][cols], label=y[a])
        dva = lgb.Dataset(tr.iloc[b][cols], label=y[b], reference=dtr)
        m = lgb.train(LGB_PARAMS, dtr, valid_sets=[dva], num_boost_round=5000,
                      callbacks=[lgb.early_stopping(125, verbose=False)])
        oof[b] = m.predict(tr.iloc[b][cols], num_iteration=m.best_iteration)
        print(f"   [{tag}] fold {fold}: {np.sqrt(np.mean((oof[b]-y[b])**2)):.4f} it={m.best_iteration}", flush=True)
    r = float(np.sqrt(np.mean((oof - y) ** 2)))
    print(f"   [{tag}] OOF = {r:.4f}", flush=True)
    return r, oof


print(f">> base ({len(base_cols)} feats) ...", flush=True)
r_base, _ = run(base_cols, "base")
print(f">> +UK ({len(uk_cols)} feats) ...", flush=True)
r_uk, oof_uk = run(uk_cols, "+UK")
np.save(ROOT / "models/frontier/oof_lgb42_ukcheck.npy", oof_uk)
d = r_uk - r_base
print("\n=== CHEAP LGB CHECK VERDICT ===", flush=True)
print(f"  base LGB-222 OOF = {r_base:.4f}   +UK LGB-225 OOF = {r_uk:.4f}   delta = {d:+.4f}", flush=True)
if d <= 0.003:
    print("  >> OK (flat-or-better as expected; UK does not regress) -> proceed to full 6-model retrain.", flush=True)
else:
    print("  >> REGRESSION: UK feats HURT the interpolation OOF -> stop; the add is net-negative.", flush=True)
print("BET5 LGBCHK DONE", flush=True)
