"""Super-solution train (part 3): blend LGB×3 + CatBoost×1 (4 models) + 3D postproc grid -> GATE.

Ridge(alpha=1, positive=True) over the 4 OOF, pick max(simple-avg, ridge) per the source.
Raw blend is the honest gate number vs frontier 10.356. Then the source's 3D postproc grid
(alpha∈[0.65,1.0] × tau∈{None,25,50,100,200} × w_pf∈{0,.05,.10}) + per-well Savitzky-Golay;
report those too but flag the grid as full-OOF-tuned (optimistic on this base).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter

ROOT = Path("/home/swatson/work/kaggle/ROGII")
SP = ROOT / "data/processed/super"
M = ROOT / "models/super"
FRONTIER = 10.356
KEYS = ["lgb_42", "lgb_7", "lgb_123", "cat_42"]

tr = pd.read_parquet(SP / "train_feats.parquet")
y = tr["target"].to_numpy(np.float32)                  # drift = TVT - last_known
base = tr["last_known_tvt"].to_numpy(np.float32)
ytrue = y + base                                        # true TVT
pf_oof = tr["pf_ancc"].to_numpy(np.float32) - base      # PF drift
md_since = np.maximum(tr["md_since"].to_numpy(np.float32), 0.0)
wells = tr["well"].to_numpy()

oofs = {k: np.load(M / f"oof_{k}.npy") for k in KEYS}
tests = {k: np.load(M / f"test_{k}.npy") for k in KEYS}
print(">> per-model OOF RMSE (drift==TVT):", flush=True)
for k in KEYS:
    print(f"   {k}: {np.sqrt(np.mean((oofs[k]-y)**2)):.4f}", flush=True)

# ---- 4-model Ridge(positive) blend; pick max(avg, ridge) per source ----
Xoof = np.column_stack([oofs[k] for k in KEYS])
r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(Xoof, y)
w = r.coef_ / max(r.coef_.sum(), 1e-9)
ridge_oof = r.predict(Xoof).astype(np.float32)
avg_oof = Xoof.mean(1).astype(np.float32)
r_ridge = float(np.sqrt(np.mean((ridge_oof - y) ** 2)))
r_avg = float(np.sqrt(np.mean((avg_oof - y) ** 2)))
use_ridge = r_ridge < r_avg
blend_oof = ridge_oof if use_ridge else avg_oof
raw_rmse = min(r_ridge, r_avg)
print(f"\n>> simple-avg OOF = {r_avg:.4f} | Ridge(pos) OOF = {r_ridge:.4f} -> using {'ridge' if use_ridge else 'avg'}", flush=True)
print(f"   weights: {dict(zip(KEYS, np.round(w,3)))}", flush=True)
print(f">> RAW blend OOF = {raw_rmse:.4f}  ({raw_rmse-FRONTIER:+.3f} vs frontier {FRONTIER})", flush=True)

# ---- source's 3D postproc grid + per-well Savitzky-Golay ----
def apply_pp(drift, alpha, tau, w_pf):
    d = drift * (1 - w_pf) + pf_oof * w_pf
    if tau:
        d = d * (1.0 - np.exp(-md_since / tau))
    return d * alpha

best = (None, None, None)
best_r = np.inf
for alpha in np.arange(0.65, 1.01, 0.05):
    for tau in [None, 25.0, 50.0, 100.0, 200.0]:
        for w_pf in [0.0, 0.05, 0.10]:
            rr = float(np.sqrt(np.mean((ytrue - (base + apply_pp(blend_oof, alpha, tau, w_pf))) ** 2)))
            if rr < best_r:
                best_r, best = rr, (float(alpha), tau, float(w_pf))
alpha, tau, w_pf = best
print(f"\n>> + postproc grid OOF = {best_r:.4f}  alpha={alpha:.2f} tau={tau} w_pf={w_pf:.2f}  [full-OOF-tuned]", flush=True)

# Savitzky-Golay per-well smoothing of the final TVT pred
pred = base + apply_pp(blend_oof, alpha, tau, w_pf)
df = pd.DataFrame({"well": wells, "pred": pred, "idx": np.arange(len(pred))})
out = pred.copy()
for _, g in df.groupby("well", sort=False):
    v = g["pred"].to_numpy(np.float64); n = len(v); wl = min(17, n)
    if wl % 2 == 0:
        wl -= 1
    if wl >= 5:
        v = savgol_filter(v, wl, 3)
    out[g["idx"].to_numpy()] = v
sg_rmse = float(np.sqrt(np.mean((ytrue - out) ** 2)))
print(f">> + Savitzky-Golay OOF = {sg_rmse:.4f}  ({sg_rmse-FRONTIER:+.3f} vs frontier {FRONTIER})", flush=True)

print(f"\n=== SUPER GATE SUMMARY (vs frontier {FRONTIER}) ===", flush=True)
print(f"   raw 4-blend : {raw_rmse:.4f}   ({raw_rmse-FRONTIER:+.3f})", flush=True)
print(f"   + postproc  : {best_r:.4f}   ({best_r-FRONTIER:+.3f})  [optimistic]", flush=True)
print(f"   + SG        : {sg_rmse:.4f}   ({sg_rmse-FRONTIER:+.3f})", flush=True)
verdict = "PASS" if raw_rmse < FRONTIER else "FAIL"
print(f"   GATE (raw vs frontier): {verdict}", flush=True)

# persist blended test prediction (drift) for the kernel/submission
Xtest = np.column_stack([tests[k] for k in KEYS])
test_drift = (r.predict(Xtest).astype(np.float32) if use_ridge else Xtest.mean(1).astype(np.float32))
np.save(M / "test_blend_drift.npy", test_drift)
json.dump({"keys": KEYS, "weights": w.tolist(), "use_ridge": bool(use_ridge),
           "raw": raw_rmse, "postproc": best_r, "pp_params": {"alpha": alpha, "tau": tau, "w_pf": w_pf},
           "sg": sg_rmse, "frontier_ref": FRONTIER,
           "per_model": {k: float(np.sqrt(np.mean((oofs[k]-y)**2))) for k in KEYS}},
          open(M / "blend_summary.json", "w"), indent=2)
print("=== SUPER BLEND DONE ===", flush=True)
