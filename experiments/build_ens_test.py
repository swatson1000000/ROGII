"""Build frontier_ens/test_feats.parquet = base-222 test + the 12 lever feats.

Mirrors experiments/lever_ensemble_gate.py's TRAIN merge EXACTLY so train/test
columns match. Deterministic transform, no model. The 12 feats live in
data/processed/{uk,dip,cwt}_feats.parquet keyed on id (cover train+test).
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
FR = ROOT / "data/processed/frontier_seeded"
OUTDIR = ROOT / "data/processed/frontier_ens"
UK = ["tvt_uk_d", "uk_ancc", "uk_vs_dense"]
DIP = ["dogleg", "cum_dogleg", "tvt_dip_grad", "tvt_dip_grad_z", "quad_b_d"]
CWT = ["dwt_ncc_d", "dwt_ncc_sc", "gr_detail_std", "dwt_vs_sc"]
ALL = UK + DIP + CWT

print(">> load base-222 test + merge UK/dip/cwt feats...", flush=True)
te = pd.read_parquet(FR / "test_feats.parquet")
for f in ["uk_feats", "dip_feats", "cwt_feats"]:
    te = te.merge(pd.read_parquet(ROOT / f"data/processed/{f}.parquet"), on="id", how="left")
te["dwt_vs_sc"] = (te["dwt_ncc_d"] - te["sc15_d"]).astype(np.float32)
for c in ALL:
    te[c] = te[c].fillna(0.0).astype(np.float32)
assert all(c in te.columns for c in ALL), "merge failed"

# verify exact FEATURE parity vs the gate's train matrix (test has no target/well)
tr_feats = {c for c in pd.read_parquet(OUTDIR / "train_feats.parquet", columns=None).columns
            if c not in {"well", "id", "target"}}
miss = tr_feats - set(te.columns)
assert not miss, f"test missing train feature cols: {miss}"
print(f"   test {te.shape}; feature parity OK ({len(tr_feats)} feats all present)", flush=True)
te.to_parquet(OUTDIR / "test_feats.parquet")
print("=== ENS TEST FEATS DONE ===", flush=True)
