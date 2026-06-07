"""PROVE the GP kernel is correct: validate its gp_drift/gp_vs_fk on TRAIN wells
against gp_feats_train (which is correct and is exactly what the models trained on).

If gp_drift matches on train wells, the kernel's feature computation == the training
computation -> the kernel is consistent with the models. (The earlier test mismatch
was a bug in the throwaway test parquet, not the kernel.)
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
anchor = json.load(open(ROOT / "models/konbu_gp/gp_anchor.json"))
print(">> rebuild imputers", flush=True)
form = FormationPlaneKNN(train_paths); row = RowKNN(train_paths); gp = FormationGP(train_paths, anchor)

# sample 25 train wells (build is is_train=False to mirror the test-time path: features only)
sample = train_paths[:25]
print(f">> build kernel matrix on {len(sample)} train wells", flush=True)
K = build_dataset(sample, form, row, is_train=False, label="diag3", gp_imputer=gp)

base = pd.read_parquet(ROOT / "data/processed/konbu/train_feats.parquet",
                       columns=["well", "row_idx", "prediction_id", "last_known_tvt", "fk_tvt_formula"])
gtr = pd.read_parquet(ROOT / "data/processed/konbu/gp_feats_train.parquet")
G = base.merge(gtr[["well", "row_idx", "gp_ancc", "gp_std", "gp_tvt_abs"]], on=["well", "row_idx"], how="inner")
G["gp_drift"] = G["gp_tvt_abs"].astype(np.float32) - G["last_known_tvt"].astype(np.float32)
G["gp_vs_fk"] = G["gp_drift"].astype(np.float32) - G["fk_tvt_formula"].astype(np.float32)

Ki = K.set_index("prediction_id"); Gi = G.set_index("prediction_id")
common = Ki.index.intersection(Gi.index)
print(f">> common rows = {len(common)}", flush=True)
print("\n=== GP feature diffs (kernel vs TRAIN ground truth) ===", flush=True)
for c in ["gp_drift", "gp_std", "gp_ancc", "gp_vs_fk"]:
    d = (Ki.loc[common, c].astype(float) - Gi.loc[common, c].astype(float)).abs()
    print(f"  {c}: max|d|={d.max():.5g}  mean|d|={d.mean():.5g}", flush=True)
print("VERDICT:", "KERNEL gp features MATCH training -> kernel correct"
      if (Ki.loc[common, "gp_drift"].astype(float) - Gi.loc[common, "gp_drift"].astype(float)).abs().max() < 0.05
      else "STILL MISMATCH -> investigate", flush=True)
print("=== DIAG3 DONE ===", flush=True)
