"""Phase B (part 3): blend LGB x3 + CatBoost x3 on the 222-union + 9.251 postproc -> GATE.

Raw 6-model NNLS blend is the honest gate number (vs banked 11.821). Then replicate the
9.251 postproc (Optuna alpha-shrink / tau-fade / w_pf PF-blend + Savitzky-Golay) and report
the postproc OOF too -- flagged as full-OOF-tuned (our earlier nested probe showed postproc can
overfit on this base, so treat the postproc delta with caution; the raw blend is the solid number).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter
import optuna

ROOT = Path("/home/swatson/work/kaggle/ROGII")
FR = ROOT / "data/processed/frontier"
M = ROOT / "models/frontier"
SEEDS = [42, 7, 123]
optuna.logging.set_verbosity(optuna.logging.WARNING)

tr = pd.read_parquet(FR / "train_feats.parquet")
y = tr["target"].to_numpy(np.float32)                 # drift = TVT - last_known
base = tr["last_known_tvt"].to_numpy(np.float32)
ytrue = y + base                                       # true TVT
pf_oof = tr["pf_ancc"].to_numpy(np.float32) - base     # PF drift
md_since = np.maximum(tr["md_since"].to_numpy(np.float32), 0.0)
wells = tr["well"].to_numpy()

keys = [f"lgb_{s}" for s in SEEDS] + [f"cat_{s}" for s in SEEDS]
oofs = {k: np.load(M / f"oof_{k}.npy") for k in keys}
tests = {k: np.load(M / f"test_{k}.npy") for k in keys}
print(">> per-model OOF RMSE (TVT):", flush=True)
for k in keys:
    print(f"   {k}: {np.sqrt(np.mean((oofs[k]-y)**2)):.4f}", flush=True)

# ---- 6-model NNLS blend (drift space; RMSE vs drift == TVT RMSE) ----
Xoof = np.column_stack([oofs[k] for k in keys])
r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(Xoof, y)
w = r.coef_ / max(r.coef_.sum(), 1e-9)
blend_oof = r.predict(Xoof).astype(np.float32)
raw_rmse = float(np.sqrt(np.mean((blend_oof - y) ** 2)))
print(f"\n>> 6-model NNLS blend OOF = {raw_rmse:.4f}  (banked stack 11.821)", flush=True)
print(f"   weights: {dict(zip(keys, np.round(w,3)))}", flush=True)
simple = np.column_stack([oofs[k] for k in keys]).mean(1)
print(f"   simple-avg OOF = {np.sqrt(np.mean((simple-y)**2)):.4f}", flush=True)

# ---- 9.251 postproc: alpha-shrink / tau-fade / w_pf, Optuna on OOF ----
def apply_pp(drift_md, alpha, tau, w_pf):
    d = drift_md * (1 - w_pf) + pf_oof * w_pf
    if tau:
        d = d * (1.0 - np.exp(-md_since / tau))
    return d * alpha

def objective(t):
    a = t.suggest_float("alpha", 0.5, 1.0, step=0.01)
    tau = t.suggest_int("tau", 5, 500, step=5)
    wp = t.suggest_float("w_pf", 0.0, 0.5, step=0.01)
    return float(np.sqrt(np.mean((ytrue - (base + apply_pp(blend_oof, a, tau, wp))) ** 2)))

study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50))
study.optimize(objective, n_trials=500, n_jobs=-1)
bp = study.best_params
pp_rmse = study.best_value
print(f"\n>> + postproc (alpha/tau/w_pf) OOF = {pp_rmse:.4f}  params={bp}", flush=True)

# ---- Savitzky-Golay per-well smoothing of the final TVT pred ----
pred = base + apply_pp(blend_oof, bp["alpha"], bp["tau"], bp["w_pf"])
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
print(f">> + Savitzky-Golay OOF = {sg_rmse:.4f}", flush=True)

print(f"\n=== GATE SUMMARY (vs banked 11.821) ===", flush=True)
print(f"   raw 6-blend : {raw_rmse:.4f}   ({raw_rmse-11.821:+.3f})", flush=True)
print(f"   + postproc  : {pp_rmse:.4f}   ({pp_rmse-11.821:+.3f})  [full-OOF-tuned, optimistic]", flush=True)
print(f"   + SG        : {sg_rmse:.4f}   ({sg_rmse-11.821:+.3f})", flush=True)
json.dump({"keys": keys, "weights": w.tolist(), "raw": raw_rmse,
           "postproc": pp_rmse, "pp_params": bp, "sg": sg_rmse,
           "per_model": {k: float(np.sqrt(np.mean((oofs[k]-y)**2))) for k in keys}},
          open(M / "blend_summary.json", "w"), indent=2)

# persist blended test prediction (TVT) for the kernel/submission
Xtest = np.column_stack([tests[k] for k in keys])
test_drift = r.predict(Xtest).astype(np.float32)
np.save(M / "test_blend_drift.npy", test_drift)
print("=== FRONTIER BLEND DONE ===", flush=True)
