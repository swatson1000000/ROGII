"""Pinpoint which feature(s) the GP kernel recomputes differently from the gate matrix.

Execs the kernel's class/function defs (NOT its main), rebuilds the 3-test-well feature
matrix the kernel way, and compares column-by-column against the gate's matrix
(cached konbu test_feats + gp_feats_test with the 4 GP features derived as gp_gate does).
Prints only columns whose max|diff| > 1e-3.
"""
from pathlib import Path
import glob, json
import numpy as np
import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
src = (ROOT / "jupyter_konbu/rogii_konbu_inference.py").read_text()
head = src.split("# ---------------- Kaggle inference main")[0]
ns = {}
exec(head, ns)
FormationPlaneKNN = ns["FormationPlaneKNN"]; RowKNN = ns["RowKNN"]
FormationGP = ns["FormationGP"]; build_dataset = ns["build_dataset"]

train_paths = [Path(p) for p in sorted(glob.glob(str(ROOT / "data/raw/train/*__horizontal_well.csv")))]
test_paths = [Path(p) for p in sorted(glob.glob(str(ROOT / "data/raw/test/*__horizontal_well.csv")))]
anchor = json.load(open(ROOT / "models/konbu_gp/gp_anchor.json"))
print(">> rebuild imputers", flush=True)
form = FormationPlaneKNN(train_paths); row = RowKNN(train_paths); gp = FormationGP(train_paths, anchor)
print(">> build kernel test matrix", flush=True)
K = build_dataset(test_paths, form, row, is_train=False, label="diag", gp_imputer=gp)

base = pd.read_parquet(ROOT / "data/processed/konbu/test_feats.parquet")
gte = pd.read_parquet(ROOT / "data/processed/konbu/gp_feats_test.parquet")
G = base.merge(gte[["well", "row_idx", "gp_ancc", "gp_std", "gp_tvt_abs"]], on=["well", "row_idx"], how="left")
G["gp_drift"] = G["gp_tvt_abs"].astype(np.float32) - G["last_known_tvt"].astype(np.float32)
G["gp_vs_fk"] = G["gp_drift"].astype(np.float32) - G["fk_tvt_formula"].astype(np.float32)

feat_cols = json.load(open(ROOT / "models/konbu_gp/feature_cols.json"))
Ki = K.set_index("prediction_id"); Gi = G.set_index("prediction_id")
common = Ki.index.intersection(Gi.index)
print(f">> rows: kernel={len(Ki)} gate={len(Gi)} common={len(common)}", flush=True)

print("\n=== columns with max|d| > 1e-3 (kernel vs gate) ===", flush=True)
flagged = 0
for c in feat_cols:
    if c not in Ki.columns or c not in Gi.columns:
        print(f"  {c}: MISSING (kernel={c in Ki.columns} gate={c in Gi.columns})", flush=True)
        flagged += 1
        continue
    d = (Ki.loc[common, c].astype(float) - Gi.loc[common, c].astype(float)).abs()
    if d.max() > 1e-3:
        print(f"  {c}: max|d|={d.max():.4g}  mean|d|={d.mean():.4g}", flush=True)
        flagged += 1
if flagged == 0:
    print("  (none — all 82 features match to 1e-3)", flush=True)
print("=== DIAG DONE ===", flush=True)
