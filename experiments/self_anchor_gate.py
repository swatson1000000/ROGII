"""BET 1' spinoff — cheap extractability gate for the SELF-ANCHORING feature.

Gate 1 killed the transductive (sibling-pooling) lever but left one positive: a well
using its OWN prefix-implied ANCC to anchor its eval-zone imputation (-17 ft vs base on
a weak IDW imputer). This gate asks the decisive question cheaply: does that feature
carry signal the production frontier stack (OOF 10.356) HASN'T already captured?

The self-anchoring estimator (b_well cancels out algebraically):
    self_tvt(eval) = -Z(eval) + IDW_over_OWN_PREFIX( TVT_input + Z )
i.e. extrapolate the formation surface + offset seen on the well's known prefix to the
eval (X,Y). self_drift = self_tvt - last_known_TVT.

Cheap residual-extractability probe (the project's standard first gate):
  1. Reconstruct the frontier blended OOF pred from the 6 saved oof npy + ridge_coef.
     residual = target - pred ; rmse(residual) == 10.356 (the stack's OOF).
  2. Fit a shallow GBM:  residual ~ [self_drift, self_dist]  under GroupKFold-by-well.
  3. new_rmse = rmse(residual - oof_residual_pred). If it drops materially below 10.356
     there is leftover signal -> escalate to a full base-vs-+feat retrain. If it sits at
     ~10.356 the feature is REDUNDANT with the 221 frontier feats -> dead.

CAVEAT (stated, per fail-loud): residual-prediction can miss signal that only emerges via
INTERACTION with the 221 existing feats in a joint retrain. This is the cheap first gate;
a flat null here, given the strong redundancy prior (prefix-slope/trajectory feats already
present), is treated as dead. A hint here -> escalate.

GroupKFold-by-well guards against the feature memorizing per-well residual.

Run: nohup python -u experiments/self_anchor_gate.py \
       > log/self_anchor_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import json
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
MAT = ROOT / "data/processed/frontier_seeded/train_feats.parquet"
MDL = ROOT / "models/frontier"
K_PRE = 20


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def well_feat(p):
    df = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT_input"])
    ti = df["TVT_input"].to_numpy(np.float64)
    mask = np.isnan(ti)
    if not mask.any():
        return None
    ms = int(mask.argmax())
    if ms < 5 or ms >= len(df):
        return None
    x = df["X"].to_numpy(np.float64); y = df["Y"].to_numpy(np.float64); z = df["Z"].to_numpy(np.float64)
    pre = np.arange(ms)
    pre = pre[np.isfinite(ti[pre]) & np.isfinite(x[pre]) & np.isfinite(y[pre]) & np.isfinite(z[pre])]
    if len(pre) < 5:
        return None
    a = ti[pre] + z[pre]                       # implied ANCC + b_well at prefix rows
    tree = cKDTree(np.column_stack([x[pre], y[pre]]))
    ev = np.arange(ms, len(df))
    K = min(K_PRE, len(pre))
    dd, ii = tree.query(np.column_stack([x[ev], y[ev]]), k=K)
    if K == 1:
        dd = dd[:, None]; ii = ii[:, None]
    w = 1.0 / (dd + 1e-6) ** 2
    anc = (w * a[ii]).sum(1) / w.sum(1)
    self_tvt = -z[ev] + anc
    lk = ti[ms - 1]
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    return pd.DataFrame({
        "id": [f"{wid}_{r}" for r in ev],
        "f_self_drift": self_tvt - lk,
        "f_self_dist": np.log1p(dd[:, 0]),
    })


def main():
    print(">> loading frontier matrix (id/well/target) ...", flush=True)
    m = pd.read_parquet(MAT, columns=["id", "well", "target"])
    n = len(m)
    print(f"   {n:,} rows, {m.well.nunique()} wells", flush=True)

    # ---- reconstruct frontier blended OOF -> residual ----
    blend = json.loads((MDL / "blend_frontier.json").read_text())
    keys, coef = blend["keys"], np.array(blend["ridge_coef"])
    pred = np.zeros(n, np.float64)
    for k, c in zip(keys, coef):
        oof = np.load(MDL / f"oof_{k}.npy").astype(np.float64).ravel()
        assert len(oof) == n, f"oof_{k} len {len(oof)} != {n}"
        pred += c * oof
    target = m["target"].to_numpy(np.float64)
    residual = target - pred
    print(f"   reconstructed blend OOF RMSE = {rmse(residual):.4f}  (json says {blend['oof_rmse']:.4f})", flush=True)

    # ---- build self-anchor feature ----
    paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    print(f">> building self-anchor feature over {len(paths)} wells ...", flush=True)
    feats = []
    for i, p in enumerate(paths):
        f = well_feat(p)
        if f is not None:
            feats.append(f)
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{len(paths)}", flush=True)
    F = pd.concat(feats, ignore_index=True)
    print(f"   feature rows {len(F):,}", flush=True)

    m = m.reset_index(drop=True)
    m["_res"] = residual
    m["_tgt"] = target
    m = m.merge(F, on="id", how="left")
    miss = m["f_self_drift"].isna().sum()
    print(f"   merged; {miss:,} rows missing feature ({100*miss/len(m):.2f}%) -> filled 0", flush=True)
    for c in ("f_self_drift", "f_self_dist"):
        m[c] = m[c].fillna(0.0)

    # context correlations
    cc_res = np.corrcoef(m["f_self_drift"], m["_res"])[0, 1]
    cc_tgt = np.corrcoef(m["f_self_drift"], m["_tgt"])[0, 1]
    print(f"   corr(self_drift, residual) = {cc_res:+.4f}   corr(self_drift, target) = {cc_tgt:+.4f}", flush=True)

    # ---- shallow GBM: residual ~ [self_drift, self_dist], GKF-by-well ----
    X = m[["f_self_drift", "f_self_dist"]].to_numpy(np.float32)
    yv = m["_res"].to_numpy(np.float64)
    groups = m["well"].to_numpy()
    oof_res = np.zeros(len(m), np.float64)
    gkf = GroupKFold(n_splits=5)
    params = dict(objective="regression", metric="rmse", num_leaves=31, learning_rate=0.05,
                  feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1,
                  min_child_samples=200, verbose=-1)
    for fold, (tr, va) in enumerate(gkf.split(X, yv, groups)):
        dtr = lgb.Dataset(X[tr], yv[tr])
        dva = lgb.Dataset(X[va], yv[va], reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        oof_res[va] = bst.predict(X[va], num_iteration=bst.best_iteration)
        print(f"   fold {fold}: best_iter {bst.best_iteration}", flush=True)

    new_rmse = rmse(residual - oof_res)
    base = rmse(residual)
    print("\n=== SELF-ANCHOR GATE VERDICT ===", flush=True)
    print(f"  frontier stack OOF (residual baseline) = {base:.4f}", flush=True)
    print(f"  after removing self-anchor residual-fit = {new_rmse:.4f}", flush=True)
    print(f"  potential improvement (optimistic)      = {base - new_rmse:+.4f} ft", flush=True)
    print("", flush=True)
    if base - new_rmse < 0.02:
        print("  >> NULL: self-anchor adds ~nothing the 221 frontier feats don't already have.", flush=True)
        print("     REDUNDANT (as predicted). BET 1' fully dead -- transductive AND self-anchor.", flush=True)
    elif base - new_rmse < 0.08:
        print("  >> MARGINAL: a sliver of leftover signal, but residual-fit is OPTIMISTIC (no", flush=True)
        print("     interaction cost, GKF-overfit risk). Likely washes in a joint retrain; weigh", flush=True)
        print("     vs the cost of a full base-vs-+feat gate before chasing it.", flush=True)
    else:
        print("  >> SIGNAL: meaningful leftover. ESCALATE to a full base(221)-vs-+self-anchor", flush=True)
        print("     retrain, gated via BLOCK-HOLDOUT OOF (Bet 3), before trusting it.", flush=True)
    print("SELFANCHOR DONE", flush=True)


if __name__ == "__main__":
    main()
