"""Cheap LGB gate: the lever-ensemble WITHOUT UK (#2 dip + #4 cwt jointly, 9 feats).

The joint #1+#2+#4 (12 feats) scored LB 8.171 (regression). UK (#1) is the dominant contributor AND the
only feat with OOF signal solo (-0.034); dip (+0.007) and cwt (+0.104) each regressed solo. Hypothesis to
test: did UK drag the LB down, and do dip+cwt carry their own joint signal without it?

Same harness/folds/params as lever_ensemble_gate.py so numbers are directly comparable.
Ablation: base / +dip / +cwt / +dip+cwt(no UK). The decisive number is +dip+cwt vs base 10.6501.
  delta <= -0.05 -> real no-UK joint signal -> discuss full retrain. else STOP (axis stays closed).

Run: nohup python -u experiments/ens_no_uk_gate.py > log/ens_no_uk_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
FR = ROOT / "data/processed/frontier_seeded"
UK = ["tvt_uk_d", "uk_ancc", "uk_vs_dense"]
DIP = ["dogleg", "cum_dogleg", "tvt_dip_grad", "tvt_dip_grad_z", "quad_b_d"]
CWT = ["dwt_ncc_d", "dwt_ncc_sc", "gr_detail_std", "dwt_vs_sc"]
NOUK = DIP + CWT
ALL = UK + DIP + CWT
N_SPLITS, SPLIT_SEED = 5, 42
LGB_PARAMS = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
    min_child_samples=10, min_child_weight=0.5, n_estimators=5000, n_jobs=-1,
    reg_alpha=2.03, reg_lambda=87.28, subsample=0.645, subsample_freq=1,
    colsample_bytree=0.821, objective="regression", metric="rmse", verbose=-1,
    device_type="cuda", max_bin=255, random_state=42)

print(">> load + merge dip/cwt feats (UK loaded only to exclude from base)...", flush=True)
tr = pd.read_parquet(FR / "train_feats.parquet")
for f in ["uk_feats", "dip_feats", "cwt_feats"]:
    tr = tr.merge(pd.read_parquet(ROOT / f"data/processed/{f}.parquet"), on="id", how="left")
tr["dwt_vs_sc"] = (tr["dwt_ncc_d"] - tr["sc15_d"]).astype(np.float32)
for c in ALL:
    tr[c] = tr[c].fillna(0.0).astype(np.float32)
assert all(c in tr.columns for c in ALL), "merge failed"
print(f"   merged {tr.shape}", flush=True)

# base excludes ALL 12 (incl UK) -> identical base to the +ALL12 gate, so deltas are comparable
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
    print(f"   [{tag:10s}] OOF = {r:.4f}  ({len(cols)} feats)", flush=True)
    return r


print(">> ablation: base / +dip / +cwt / +dip+cwt(no UK) ...", flush=True)
r_base = run(base, "base")
r_dip = run(base + DIP, "+dip")
r_cwt = run(base + CWT, "+cwt")
r_nouk = run(base + NOUK, "+dip+cwt")

print("\n=== NO-UK ENSEMBLE GATE VERDICT ===", flush=True)
print(f"  base       {r_base:.4f}", flush=True)
print(f"  +dip       {r_dip:.4f}  ({r_dip-r_base:+.4f})", flush=True)
print(f"  +cwt       {r_cwt:.4f}  ({r_cwt-r_base:+.4f})", flush=True)
print(f"  +dip+cwt   {r_nouk:.4f}  ({r_nouk-r_base:+.4f})  <-- the no-UK joint test", flush=True)
d = r_nouk - r_base
if d <= -0.05:
    print("  >> NO-UK JOINT SIGNAL: dip+cwt combine without UK -> discuss full retrain.", flush=True)
elif d <= 0.003:
    print("  >> ~FLAT: no no-UK joint signal -> STOP, feature-addition axis stays closed.", flush=True)
else:
    print("  >> REGRESSION: dip+cwt without UK HURT -> STOP, the two solo-dead levers don't rescue each other.", flush=True)
print("NO-UK ENS GATE DONE", flush=True)
