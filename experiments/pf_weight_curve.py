"""OOF RMSE vs PF output-blend weight w (final = (1-w)*GBM + w*PF), absolute/residual space.
Quantifies the CV cost of moving from the fitted w=0.44 toward the public ~0.70 / a 0.77 mix.
Reuses pf_output_blend.py's verified per-well alignment.
"""
from pathlib import Path
import glob, os, json
import numpy as np
import pandas as pd
import joblib

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
FR = ROOT / "data/processed/frontier_seeded"
MODELS = ROOT / "models/frontier"
PF_KEY = "pf_scale_12"

df = pd.read_parquet(FR / "train_feats.parquet", columns=["well", "id", "last_known_tvt", "target"])
blend = json.load(open(MODELS / "blend_frontier.json"))
coef = np.array(blend["ridge_coef"], np.float64)
oofs = np.column_stack([np.load(MODELS / f"oof_{k}.npy").astype(np.float64) for k in blend["keys"]])
target = df["target"].to_numpy(np.float64); lastk = df["last_known_tvt"].to_numpy(np.float64)
gbm_resid = (oofs @ coef) - target
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)

paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells = []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue
    pf_wells.append(wid)
results = joblib.load(MODELS / "pf_real_results.pkl")
def is_err(r): return r is not None and isinstance(r[0], str) and r[0] == "ERR"
good = [r for r in results if r is not None and not is_err(r)]
assert len(good) == len(pf_wells)

grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}
gkeep, pkeep = [], []
for wid, (pf_truth, preds) in zip(pf_wells, good):
    sub = grp.get(wid)
    if sub is None or len(sub) != len(pf_truth): continue
    t = sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64)
    if not np.allclose(t, pf_truth, atol=1e-3): continue
    gkeep.append(sub.index.to_numpy()); pkeep.append(preds[PF_KEY].astype(np.float64) - pf_truth)
pos = np.concatenate(gkeep)
g = gbm_resid[pos]; p = np.concatenate(pkeep)
rmse = lambda w: float(np.sqrt(np.mean(((1 - w) * g + w * p) ** 2)))
d = g - p; wstar = float(np.dot(g, d) / np.dot(d, d))

print(f"aligned rows={len(g):,}  corr(gbm,pf)={np.corrcoef(g,p)[0,1]:.3f}", flush=True)
print(f"closed-form w* = {wstar:.4f}  OOF={rmse(wstar):.4f}", flush=True)
print("\n  w      OOF RMSE   vs w*=0.44", flush=True)
for w in [0.00, 0.30, 0.44, 0.56, 0.60, 0.70, 0.77, 0.85, 1.00]:
    print(f"  {w:.2f}   {rmse(w):8.4f}   {rmse(w)-rmse(wstar):+.4f}", flush=True)
print("\nLB context: current LB(w=0.44)=8.269, OOF=9.169, LB-OOF=-0.90 (favorable).", flush=True)
print("If the SAME -0.90 transfer held at w=0.77, LB ~ OOF(0.77)-0.90 (only if PF transfers as well there).", flush=True)
print("CURVE DONE", flush=True)
