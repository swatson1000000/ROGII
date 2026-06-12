"""Production FWLS fit (CAND B, gated PASS 2026-06-11: nested OOF 9.1811 = -0.0838).

Fits theta on ALL 773 wells at lambda=1000 (the per-lambda curve's interior optimum; nested
selection chose 300-3000) with the gate's exact protocol: centered no-intercept a_w-weighted
ridge, metas standardized by global mu/sd, target = per-well w* - 0.57.

Writes models/frontier_ens_nouk/fwls.json: {theta, mu, sd, w0, lo, hi, meta_names} for
embedding in the kernel. Meta computation at inference MUST replicate fwls_gate.py exactly
(metas 8/9 over the FULL-length PF arrays incl. prefix; 1-7 over eval rows).

Run: nohup python -u experiments/fwls_fit_production.py > log/fwls_fit_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os, json
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import Ridge

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
KEYS = ["lgb_42", "lgb_7", "lgb_123", "cat_42", "cat_7", "cat_123"]
W0, W_LO, W_HI = 0.57, 0.45, 0.70
LAM = 1000.0
rmse = lambda e: float(np.sqrt(np.mean(e * e)))

df = pd.read_parquet(ROOT / "data/processed/frontier_ens/train_feats.parquet",
                     columns=["well", "id", "last_known_tvt", "target", "md_since",
                              "z", "pfx_rmse", "sig_std", "spatial_knn_dist", "dense_dist"])
y = df["target"].to_numpy(np.float64)
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)
X = np.column_stack([np.load(ROOT / "models/frontier_ens_nouk" / f"oof_{k}.npy").astype(np.float64)
                     for k in KEYS])
r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(X, y)
g_res_all = X @ r.coef_ - y
assert abs(rmse(g_res_all) - 10.3232) < 0.01

paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells = []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue
    pf_wells.append(wid)
grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}
res = joblib.load(ROOT / "models/frontier/pf_real_results.pkl")
rg = [r for r in res if r is not None and not (isinstance(r[0], str) and r[0] == "ERR")]
assert len(rg) == len(pf_wells)

A, B, META, CC, NROW = [], [], [], [], []
SCALES = ["pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12"]
for i, wid in enumerate(pf_wells):
    truth, pf = rg[i]
    sub = grp.get(wid)
    if sub is None or len(sub) != len(truth):
        continue
    if not np.allclose(sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64),
                       truth, atol=1e-3):
        continue
    g = g_res_all[sub.index.to_numpy()]
    p = pf["pf_scale_12"] - truth
    d = p - g
    A.append(float(np.dot(d, d))); B.append(float(np.dot(g, d)))
    CC.append(float(np.dot(g, g))); NROW.append(len(sub))
    scale_mat = np.vstack([pf[s] for s in SCALES])
    zc = sub["z"].to_numpy(np.float64)
    META.append([
        np.log(len(sub)),
        np.log(max(sub["md_since"].max(), 1.0)),
        float(zc.max() - zc.min()),
        float(np.nanmean(sub["pfx_rmse"].to_numpy(np.float64))),
        float(np.nanmean(sub["sig_std"].to_numpy(np.float64))),
        float(np.log1p(np.nanmean(sub["spatial_knn_dist"].to_numpy(np.float64)))),
        float(np.log1p(np.nanmean(sub["dense_dist"].to_numpy(np.float64)))),
        float(np.mean(np.std(scale_mat, axis=0))),
        float(np.mean(np.abs(pf["pf_scale_12"] - pf["pf_mean"]))),
    ])
A = np.array(A); B = np.array(B); CC = np.array(CC); NROW = np.array(NROW)
META = np.array(META, np.float64)
META = np.where(np.isfinite(META), META, np.nan)
col_med = np.nanmedian(META, axis=0)
META = np.where(np.isnan(META), col_med, META)
n_total = int(NROW.sum())
print(f"wells {len(A)}, rows {n_total}", flush=True)

w_star = -B / A
mu, sd = META.mean(0), META.std(0)
sd[sd == 0] = 1.0
M = (META - mu) / sd
aw = A / A.mean()
MtA = M.T * aw
theta = np.linalg.solve(MtA @ M + LAM * np.eye(M.shape[1]), MtA @ (w_star - W0))
w_fit = np.clip(W0 + M @ theta, W_LO, W_HI)

base = np.sqrt((A * W0**2 + 2 * B * W0 + CC).sum() / n_total)
fit = np.sqrt((A * w_fit**2 + 2 * B * w_fit + CC).sum() / n_total)
print(f"flat 0.57: {base:.4f}  | full-fit (IN-SAMPLE, info only): {fit:.4f} ({fit - base:+.4f})", flush=True)
print(f"[honest expected number stays the NESTED OOF 9.1811 = -0.0838]", flush=True)
print(f"w distribution: p5/p50/p95 = {np.percentile(w_fit, [5, 50, 95]).round(3)}, "
      f"mean {w_fit.mean():.3f}, clipped {((w_fit <= W_LO + 1e-9) | (w_fit >= W_HI - 1e-9)).mean():.2f}",
      flush=True)
meta_names = ["log_n_eval", "log_md_span", "z_span", "pfx_rmse_mean", "sig_std_mean",
              "log1p_knn_dist", "log1p_dense_dist", "pf_scale_spread", "pf_w_vs_mean"]
print("theta by meta:", {n: round(float(t), 5) for n, t in zip(meta_names, theta)}, flush=True)

out = {"theta": theta.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
       "w0": W0, "lo": W_LO, "hi": W_HI, "lam": LAM, "meta_names": meta_names,
       "col_med": col_med.tolist()}
with open(ROOT / "models/frontier_ens_nouk/fwls.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote models/frontier_ens_nouk/fwls.json", flush=True)
print("FWLS FIT DONE", flush=True)
