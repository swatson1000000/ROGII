"""Rebuild correct gp_feats_test (fixed b_well), regenerate the ground-truth test
prediction, and confirm the kernel output matches it.

Why: the original gp_feats_test had b_well=0 (test wells lack a TVT column -> load bug,
now fixed). The kernel computes b_well correctly from TVT_input. This script builds the
CORRECT test ground truth the gate way and compares to the kernel's /tmp/gpval/submission.csv.
"""
from pathlib import Path
import glob, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

ROOT = Path("/home/swatson/work/kaggle/ROGII")
src = (ROOT / "jupyter_konbu/rogii_konbu_inference.py").read_text()
head = src.split("# ---------------- Kaggle inference main")[0]
ns = {}; exec(head, ns)
FormationGP = ns["FormationGP"]
MODELS = ROOT / "models/konbu_gp"; N = 5

train_paths = [Path(p) for p in sorted(glob.glob(str(ROOT / "data/raw/train/*__horizontal_well.csv")))]
test_paths = [Path(p) for p in sorted(glob.glob(str(ROOT / "data/raw/test/*__horizontal_well.csv")))]
anchor = json.load(open(MODELS / "gp_anchor.json"))
gp = FormationGP(train_paths, anchor)

# rebuild correct gp_feats_test (full-ref posterior + correct b_well from TVT_input prefix)
rows = []
for p in test_paths:
    wid = p.stem.replace("__horizontal_well", "")
    h = pd.read_csv(p)
    xy = h[["X", "Y"]].to_numpy(np.float64)
    mean, std = gp.impute(xy)
    tvi = h["TVT_input"]
    ms = int(np.flatnonzero(tvi.isna().to_numpy())[0]) if tvi.isna().any() else len(h)
    z = h["Z"].to_numpy(np.float64)
    if ms > 0:
        b = float(np.median(tvi.to_numpy(np.float64)[:ms] + z[:ms] - mean[:ms].astype(np.float64)))
    else:
        b = 0.0
    tvt_abs = -z + mean + b
    rows.append(pd.DataFrame({"well": wid, "row_idx": np.arange(len(h), dtype=np.int64),
                              "gr": h["GR"].to_numpy(np.float32),
                              "gp_ancc": mean, "gp_std": std,
                              "gp_tvt_abs": tvt_abs.astype(np.float32)}))
gte = pd.concat(rows, ignore_index=True)
gte.to_parquet(ROOT / "data/processed/konbu/gp_feats_test.parquet")
print(f">> rebuilt gp_feats_test {gte.shape}; gp_tvt_abs mean={gte['gp_tvt_abs'].mean():.1f} "
      f"(should be ~true TVT ~11800, NOT ~344)", flush=True)

# gate-style test matrix + derived gp features
base = pd.read_parquet(ROOT / "data/processed/konbu/test_feats.parquet")
G = base.merge(gte[["well", "row_idx", "gp_ancc", "gp_std", "gp_tvt_abs"]], on=["well", "row_idx"], how="left")
G["gp_drift"] = G["gp_tvt_abs"].astype(np.float32) - G["last_known_tvt"].astype(np.float32)
G["gp_vs_fk"] = G["gp_drift"].astype(np.float32) - G["fk_tvt_formula"].astype(np.float32)
feat_cols = json.load(open(MODELS / "feature_cols.json"))
blend = json.load(open(MODELS / "blend_catboost.json"))
keys, coefs = blend["keys"], np.asarray(blend["ridge_coef"], np.float64)
X = G[feat_cols]; Xv = X.values

fam = {}
for k in keys:
    pr = np.zeros(len(G), np.float64); seed = k.split("_")[1]
    if k.startswith("lgb_"):
        for f in range(N):
            pr += lgb.Booster(model_file=str(MODELS / f"lgb_seed{seed}_fold{f}.txt")).predict(X) / N
    elif k.startswith("cat_"):
        for f in range(N):
            m = CatBoostRegressor(); m.load_model(str(MODELS / f"cat_seed{seed}_fold{f}.cbm")); pr += m.predict(Xv) / N
    else:
        dte = xgb.DMatrix(Xv)
        for f in range(N):
            b = xgb.Booster(); b.load_model(str(MODELS / f"xgb_seed{seed}_fold{f}.json"))
            pr += b.predict(dte, iteration_range=(0, int(b.best_iteration) + 1)) / N
    fam[k] = pr
drift = sum(c * fam[k] for c, k in zip(coefs, keys))
tvt = G["last_known_tvt"].to_numpy(np.float64) + drift
GT = pd.DataFrame({"id": G["prediction_id"], "tvt": tvt})
GT.to_csv(ROOT / "data/processed/konbu_gp/submission_local.csv", index=False)
print(f">> regenerated corrected ground-truth submission ({len(GT)} rows)", flush=True)

# compare to kernel output
K = pd.read_csv("/tmp/gpval/submission.csv")
m = K.merge(GT, on="id", suffixes=("_kernel", "_gt"))
d = np.abs(m["tvt_kernel"].to_numpy() - m["tvt_gt"].to_numpy())
print(f"\n=== KERNEL vs CORRECTED ground truth ===", flush=True)
print(f"rows={len(m)}  max|d|={d.max():.6g}  mean|d|={d.mean():.6g}  p99={np.percentile(d,99):.6g}", flush=True)
print("VERDICT:", "KERNEL MATCHES -> safe to ship" if d.max() < 0.05 else f"MISMATCH max={d.max():.4g}", flush=True)
print("=== FINALIZE DONE ===", flush=True)
