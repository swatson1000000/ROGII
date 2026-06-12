"""Build the frontier_super union matrices: nouk-231 + the 28 super-build cols (+ selfcorr5 if passed).

Joins by id; train from data/processed/{frontier_ens,super}/train_feats.parquet, test likewise.
The super28 gate passed -0.1666 (cheap LGB); this is the matrix for the pre-registered full
6-model retrain + stack OOF gate. Set INCLUDE_SELFCORR=1 in the environment to also merge
data/processed/selfcorr_feats.parquet (only if its gate passed).

Run: python -u experiments/ens_super_build.py
"""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
OUT = ROOT / "data/processed/frontier_super"; OUT.mkdir(parents=True, exist_ok=True)
UK = {"tvt_uk_d", "uk_ancc", "uk_vs_dense"}
INCLUDE_SELFCORR = os.environ.get("INCLUDE_SELFCORR", "0") == "1"

nouk_cols = set(json.load(open(ROOT / "models/frontier_ens_nouk/feature_cols.json")))
sup_cols = json.load(open(ROOT / "data/processed/super/feature_cols.json"))
ADD = sorted(set(sup_cols) - nouk_cols)
print(f">> union = nouk-231 + {len(ADD)} super cols; selfcorr={'YES' if INCLUDE_SELFCORR else 'no'}", flush=True)

for split in ["train", "test"]:
    df = pd.read_parquet(ROOT / f"data/processed/frontier_ens/{split}_feats.parquet")
    df = df.drop(columns=[c for c in UK if c in df.columns])
    sup = pd.read_parquet(ROOT / f"data/processed/super/{split}_feats.parquet", columns=["id"] + ADD)
    n0 = len(df)
    df = df.merge(sup, on="id", how="left")
    assert len(df) == n0, f"{split}: join changed row count"
    miss = df[ADD].isna().mean().mean()
    print(f"   [{split}] {df.shape}  mean NaN over added cols: {miss:.4f}", flush=True)
    if INCLUDE_SELFCORR and split == "train":
        sc = pd.read_parquet(ROOT / "data/processed/selfcorr_feats.parquet")
        df = df.merge(sc, on="id", how="left")
        df["selfcorr_vs_sc"] = (df["selfcorr_d"] - df["sc15_d"]).astype(np.float32)
        assert len(df) == n0
    for c in df.columns:
        if c in {"well", "id", "target"}:
            continue
        df[c] = df[c].fillna(0.0).astype(np.float32)
    df.to_parquet(OUT / f"{split}_feats.parquet")
    print(f"   [{split}] -> {OUT / f'{split}_feats.parquet'}", flush=True)

fc = [c for c in pd.read_parquet(OUT / "train_feats.parquet").columns if c not in {"well", "id", "target"}]
print(f">> final feature count: {len(fc)}", flush=True)
print("ENS SUPER BUILD DONE", flush=True)
