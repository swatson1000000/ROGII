"""Build #1+#2+#4 as a JOINT 12-feature ensemble + cheap LGB gate.

Tests the [[reproduce-wholesale-beats-additive-tests]] hypothesis: PF/DTW/NCC were each dead solo but
+1.4 ft JOINTLY (orthogonal weak estimators the GBM arbitrates). Could #1 (UK kriging, regressed),
#2 (dip/curvature, regressed), #4 (CWT texture, regressed) similarly hide JOINT signal a GBM unlocks via
cross-family interaction (e.g. use #4 texture-confidence to gate #1's UK estimate)?

(#3 RGT = structurally ill-posed, no buildable artifact -> not a member. #5 leak = override already tested
 as v6 = 8.158, dead on public -> not a feature.)

12 features, all already built:
    #1 UK   : tvt_uk_d, uk_ancc, uk_vs_dense                              (data/processed/uk_feats.parquet)
    #2 dip  : dogleg, cum_dogleg, tvt_dip_grad, tvt_dip_grad_z, quad_b_d  (data/processed/dip_feats.parquet)
    #4 cwt  : dwt_ncc_d, dwt_ncc_sc, gr_detail_std, dwt_vs_sc(=dwt_ncc_d-sc15_d)  (data/processed/cwt_feats.parquet)

Single-LGB GKF-5 seed42 (same harness/folds/params as the per-family gates so numbers are comparable).
Ablation: base / +UK / +dip / +cwt / +ALL12. The decisive number is +ALL12 vs base.
  delta(ALL12) <= -0.05 -> JOINT signal exists (reproduce-wholesale was right again) -> full 6-model retrain.
  ~flat / regress       -> no joint signal (redundant #1 + absorbed #2 + noise #4 don't combine). STOP.

Run: nohup python -u experiments/lever_ensemble_gate.py > log/lever_ens_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
FR = ROOT / "data/processed/frontier_seeded"
OUTDIR = ROOT / "data/processed/frontier_ens"; OUTDIR.mkdir(parents=True, exist_ok=True)
UK = ["tvt_uk_d", "uk_ancc", "uk_vs_dense"]
DIP = ["dogleg", "cum_dogleg", "tvt_dip_grad", "tvt_dip_grad_z", "quad_b_d"]
CWT = ["dwt_ncc_d", "dwt_ncc_sc", "gr_detail_std", "dwt_vs_sc"]
ALL = UK + DIP + CWT
N_SPLITS, SPLIT_SEED = 5, 42
LGB_PARAMS = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
    min_child_samples=10, min_child_weight=0.5, n_estimators=5000, n_jobs=-1,
    reg_alpha=2.03, reg_lambda=87.28, subsample=0.645, subsample_freq=1,
    colsample_bytree=0.821, objective="regression", metric="rmse", verbose=-1,
    device_type="cuda", max_bin=255, random_state=42)

print(">> load + merge UK/dip/cwt feats...", flush=True)
tr = pd.read_parquet(FR / "train_feats.parquet")
for f in ["uk_feats", "dip_feats", "cwt_feats"]:
    tr = tr.merge(pd.read_parquet(ROOT / f"data/processed/{f}.parquet"), on="id", how="left")
tr["dwt_vs_sc"] = (tr["dwt_ncc_d"] - tr["sc15_d"]).astype(np.float32)
for c in ALL:
    tr[c] = tr[c].fillna(0.0).astype(np.float32)
assert all(c in tr.columns for c in ALL), "merge failed"
print(f"   merged {tr.shape}", flush=True)
tr.to_parquet(OUTDIR / "train_feats.parquet")

base = [c for c in tr.columns if c not in {"well", "id", "target"} and c not in ALL]
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


print(">> ablation: base / +UK / +dip / +cwt / +ALL12 ...", flush=True)
r_base, oof_base = run(base, "base")
r_uk, _ = run(base + UK, "+UK")
r_dip, _ = run(base + DIP, "+dip")
r_cwt, _ = run(base + CWT, "+cwt")
r_all, _ = run(base + ALL, "+ALL12")

resid = oof_base - y
print("\n=== 12-feature diagnostics (corr target / base-OOF residual) ===", flush=True)
for c in ALL:
    v = tr[c].to_numpy(np.float32)
    ct = float(np.corrcoef(v, y)[0, 1]) if np.std(v) > 0 else 0.0
    cr = float(np.corrcoef(v, resid)[0, 1]) if np.std(v) > 0 else 0.0
    print(f"  {c:16s} corr(target)={ct:+.4f}  corr(residual)={cr:+.4f}", flush=True)

print("\n=== LEVER-ENSEMBLE GATE VERDICT ===", flush=True)
print(f"  base   {r_base:.4f}", flush=True)
print(f"  +UK    {r_uk:.4f}  ({r_uk-r_base:+.4f})", flush=True)
print(f"  +dip   {r_dip:.4f}  ({r_dip-r_base:+.4f})", flush=True)
print(f"  +cwt   {r_cwt:.4f}  ({r_cwt-r_base:+.4f})", flush=True)
print(f"  +ALL12 {r_all:.4f}  ({r_all-r_base:+.4f})  <-- the joint test", flush=True)
d = r_all - r_base
if d <= -0.05:
    print("  >> JOINT SIGNAL: the ensemble beats base -> reproduce-wholesale again -> full 6-model retrain + PF-blend + LB.", flush=True)
elif d <= 0.003:
    print("  >> ~FLAT: no joint signal -> redundant(#1)+absorbed(#2)+noise(#4) don't combine. STOP.", flush=True)
else:
    print("  >> REGRESSION: the joint union HURTS (compounds overfit/noise surface). STOP -- the ensemble is not a lever.", flush=True)
print("LEVER ENS GATE DONE", flush=True)
