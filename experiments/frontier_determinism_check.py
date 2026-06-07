"""Rebuild test features and diff per-column vs the cached test_feats.parquet, to confirm
the kernel mismatch is PF/DTW non-determinism (and isolate which columns are stochastic)."""
from pathlib import Path
import os
import numpy as np
import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
SRC = Path("/tmp/nihilisticneuralnet_9-251-rogii-wellbore-geology-prediction-dwt-based.code.py")
os.environ["ROGII_COMP"] = str(ROOT / "data/raw")
prefix = SRC.read_text().split("# ===== CODE CELL 5 =====")[0]
for bad in ("from hill_climbing import Climber\n", "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n", "import optuna\n"):
    prefix = prefix.replace(bad, "")
prefix = prefix.replace(
    'dataset_path = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")',
    'dataset_path = Path(os.environ["ROGII_COMP"])')
prefix = prefix.replace(
    'artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")',
    'artifacts_path = Path("/tmp/_nonexistent_artifacts")')
prefix = prefix.replace("cache=True", "cache=False")
ns = {"os": os}
exec(compile(prefix, str(SRC), "exec"), ns)
build_dataset = ns["build_dataset"]
CFG = ns["CFG"]

test_paths = sorted((CFG.dataset_path / "test").glob("*__horizontal_well.csv"))
fresh = build_dataset(test_paths, is_train=False, label="test").reset_index(drop=True)
cached = pd.read_parquet(ROOT / "data/processed/frontier/test_feats.parquet").reset_index(drop=True)
fresh = fresh.set_index("id").reindex(cached["id"]).reset_index()

feats = [c for c in cached.columns if c not in {"well", "id", "target"}]
diffs = []
for c in feats:
    if pd.api.types.is_numeric_dtype(cached[c]):
        d = float(np.nanmean(np.abs(fresh[c].to_numpy(float) - cached[c].to_numpy(float))))
        diffs.append((c, d))
diffs.sort(key=lambda x: -x[1])
print(">> columns that DIFFER most between two builds (mean|Δ|):", flush=True)
for c, d in diffs[:25]:
    print(f"   {c:24s} {d:.5f}", flush=True)
nz = [c for c, d in diffs if d > 1e-4]
print(f"\n>> {len(nz)}/{len(feats)} columns differ >1e-4 (the stochastic ones)", flush=True)
print(f">> {len(feats)-len(nz)} columns are deterministic (identical)", flush=True)
print("DETERMINISM CHECK DONE", flush=True)
