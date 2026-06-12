"""Cheap single-LGB gate: nouk-231 + the 28 super-build columns absent from it.

The 28 cols (signal_std, signal_mean_d, gr_detr, pfx_gr_slope, tvt_d50_d, gr_vs_slp, gr_vs_tw,
knn_d, WLS tvtFw_* x6, tvtF_*_d x6, sc*_score x3, beam_vs_form, form_vs_dense, pf_ancc_d, pf_z_d)
are ALREADY COMPUTED in data/processed/super/train_feats.parquet -- join by id, zero feature-gen cost.
Some are likely renames/dupes of existing nouk cols; the gate measures that empirically.

Context: the no-UK win (LB 8.131) showed solo-dead features can transfer JOINTLY without UK, so the
feature axis is conditionally reopened for non-UK feats with pre-registered gates. None of these 28
is UK/density-coupled-spatial-NEW (tvtF/tvtFw are plane-KNN formula variants already in the family).

Same harness/folds/params as lever_ensemble_gate.py so numbers are comparable.
Ablation: base231 / +new9 (the genuinely-new families) / +ALL28.

GATE (pre-registered):
  delta(best) <= -0.05  -> real joint signal -> full 6-model retrain + stack OOF gate before any kernel work
  -0.05 < delta <= -0.02 -> marginal: only proceed if the stack OOF confirms >= -0.03 (the no-UK transfer size)
  > -0.02               -> dead, stop; do NOT rationalize.

Run: nohup python -u experiments/super28_gate.py > log/super28_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import json
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
UK = ["tvt_uk_d", "uk_ancc", "uk_vs_dense"]
NEW9 = ["signal_std", "signal_mean_d", "gr_detr", "pfx_gr_slope", "tvt_d50_d",
        "gr_vs_slp", "gr_vs_tw", "knn_d", "tvt_densew_d"]  # tvt_densew_d already in nouk -> dropped below if so
N_SPLITS, SPLIT_SEED = 5, 42
LGB_PARAMS = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
    min_child_samples=10, min_child_weight=0.5, n_estimators=5000, n_jobs=-1,
    reg_alpha=2.03, reg_lambda=87.28, subsample=0.645, subsample_freq=1,
    colsample_bytree=0.821, objective="regression", metric="rmse", verbose=-1,
    device_type="cuda", max_bin=255, random_state=42)

print(">> load nouk-231 matrix (frontier_ens minus UK) + join 28 super cols by id...", flush=True)
tr = pd.read_parquet(ROOT / "data/processed/frontier_ens/train_feats.parquet")
tr = tr.drop(columns=[c for c in UK if c in tr.columns])
nouk_cols = set(json.load(open(ROOT / "models/frontier_ens_nouk/feature_cols.json")))
sup_cols = json.load(open(ROOT / "data/processed/super/feature_cols.json"))
ADD = sorted(set(sup_cols) - nouk_cols)
NEW9 = [c for c in NEW9 if c in ADD]
print(f"   adding {len(ADD)} super cols: {ADD}", flush=True)
print(f"   'genuinely new' subset ({len(NEW9)}): {NEW9}", flush=True)

sup = pd.read_parquet(ROOT / "data/processed/super/train_feats.parquet", columns=["id"] + ADD)
tr = tr.merge(sup, on="id", how="left")
covmiss = tr[ADD].isna().mean().mean()
print(f"   mean NaN rate over added cols (pre-fill): {covmiss:.4f}", flush=True)
for c in ADD:
    tr[c] = tr[c].fillna(0.0).astype(np.float32)

base = [c for c in tr.columns if c not in {"well", "id", "target"} and c not in ADD]
assert len(base) == 231, f"base is {len(base)} cols, expected 231"
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
    r = float(np.sqrt(np.mean((oof - y) ** 2)))
    print(f"   [{tag:8s}] OOF = {r:.4f}  ({len(cols)} feats)", flush=True)
    return r, oof


print(">> ablation: base231 / +new9 / +ALL28 ...", flush=True)
r_base, oof_base = run(base, "base231")
r_new9, _ = run(base + NEW9, "+new9")
r_all, _ = run(base + ADD, "+ALL28")

resid = oof_base - y
print("\n=== added-feature diagnostics (corr target / base-OOF residual) ===", flush=True)
for c in ADD:
    v = tr[c].to_numpy(np.float32)
    ct = float(np.corrcoef(v, y)[0, 1]) if np.std(v) > 0 else 0.0
    cr = float(np.corrcoef(v, resid)[0, 1]) if np.std(v) > 0 else 0.0
    print(f"  {c:16s} corr(target)={ct:+.4f}  corr(residual)={cr:+.4f}", flush=True)

print("\n=== SUPER28 GATE VERDICT ===", flush=True)
print(f"  base231 {r_base:.4f}", flush=True)
print(f"  +new9   {r_new9:.4f}  ({r_new9 - r_base:+.4f})", flush=True)
print(f"  +ALL28  {r_all:.4f}  ({r_all - r_base:+.4f})", flush=True)
d = min(r_new9, r_all) - r_base
if d <= -0.05:
    print("  >> SIGNAL: proceed to full 6-model retrain + stack OOF gate (NOT straight to kernel).", flush=True)
elif d <= -0.02:
    print("  >> MARGINAL: only proceed if stack OOF confirms <= -0.03 (the no-UK transfer size).", flush=True)
else:
    print("  >> DEAD: the super-exclusive cols add nothing on the nouk base. STOP.", flush=True)
print("SUPER28 GATE DONE", flush=True)
