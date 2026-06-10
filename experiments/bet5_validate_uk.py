"""Validate the kernel's in-build UK features match a standalone FULL-REF uk_predict (the policy the
kernel uses at inference). Confirms the build_well UK injection is bit-correct."""
from pathlib import Path
import glob, os, importlib.util
import numpy as np, pandas as pd
ROOT = Path("/home/swatson/work/kaggle/ROGII")
spec = importlib.util.spec_from_file_location("ukb", str(ROOT/"experiments/bet5_build_uk_feats.py"))
ukb = importlib.util.module_from_spec(spec); spec.loader.exec_module(ukb)  # uk_predict, DEGREE

z = np.load(ROOT/"models/frontier/uk_centroids.npz")
xy, A = z["xy"].astype(np.float64), z["ancc"].astype(np.float64)
mu, sd = z["mu"].astype(np.float64), z["sd"].astype(np.float64)
cov = (float(z["sill"]), float(z["nugget"]), float(z["ell"])); XYn = (xy - mu)/sd

def ukp(xy_raw):
    Qn = (xy_raw - mu)/sd
    return ukb.uk_predict(XYn, A, cov, Qn)   # full-ref (no exclusion), matches kernel test policy

rows = []
for p in sorted(glob.glob(str(ROOT/"data/raw/test/*__horizontal_well.csv"))):
    wid = Path(p).stem.replace("__horizontal_well","")
    h = pd.read_csv(p, usecols=lambda c: c in ("X","Y","Z","TVT_input"))
    kn = h["TVT_input"].notna().to_numpy(); ev = ~kn
    X=h["X"].to_numpy(np.float64); Y=h["Y"].to_numpy(np.float64); Z=h["Z"].to_numpy(np.float64)
    uk_kn = ukp(np.column_stack([X[kn],Y[kn]])); uk_ev = ukp(np.column_stack([X[ev],Y[ev]]))
    ktvt = h["TVT_input"].to_numpy(np.float64)[kn]
    b_uk = float(np.median(ktvt + Z[kn] - uk_kn)); last_tvt=float(ktvt[-1])
    tvt_uk = -Z[ev] + uk_ev + b_uk
    idx = np.flatnonzero(ev)
    for i,ri in enumerate(idx):
        rows.append((f"{wid}_{ri}", np.float32(tvt_uk[i]-last_tvt), np.float32(uk_ev[i])))
ref = pd.DataFrame(rows, columns=["id","ref_tvt_uk_d","ref_uk_ancc"])

k = pd.read_parquet("/tmp/kval_uk_out/kernel_uk.parquet")
m = k.merge(ref, on="id", how="inner")
print(f"matched {len(m)}/{len(k)} kernel rows")
d_ukd = np.abs(m["tvt_uk_d"].to_numpy(np.float64) - m["ref_tvt_uk_d"].to_numpy(np.float64))
d_anc = np.abs(m["uk_ancc"].to_numpy(np.float64) - m["ref_uk_ancc"].to_numpy(np.float64))
# internal consistency: uk_vs_dense == tvt_uk_d - tvt_dense_d
d_cons = np.abs(m["uk_vs_dense"].to_numpy(np.float64) - (m["tvt_uk_d"]-m["tvt_dense_d"]).to_numpy(np.float64))
print(f"tvt_uk_d   : max|Δ|={d_ukd.max():.6g}  mean|Δ|={d_ukd.mean():.6g}")
print(f"uk_ancc    : max|Δ|={d_anc.max():.6g}  mean|Δ|={d_anc.mean():.6g}")
print(f"uk_vs_dense consistency: max|Δ|={d_cons.max():.6g}")
ok = d_ukd.max()<1e-2 and d_anc.max()<1e-2 and d_cons.max()<1e-3
print("UK VALIDATION:", "PASS (kernel UK == full-ref standalone)" if ok else "FAIL")
