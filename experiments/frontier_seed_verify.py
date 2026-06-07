"""Verify deterministic per-well numba seeding makes PF/stochastic-DTW reproducible.
Build the 3 test wells TWICE with seeding; confirm the previously-stochastic columns now match.
If mean|Δ|==0, seeding works -> safe to do the full seeded rebuild + retrain.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
SRC = Path("/tmp/nihilisticneuralnet_9-251-rogii-wellbore-geology-prediction-dwt-based.code.py")
os.environ["ROGII_COMP"] = str(ROOT / "data/raw")


def patched_prefix():
    p = SRC.read_text().split("# ===== CODE CELL 5 =====")[0]
    for bad in ("from hill_climbing import Climber\n", "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n", "import optuna\n"):
        p = p.replace(bad, "")
    p = p.replace(
        'dataset_path = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")',
        'dataset_path = Path(os.environ["ROGII_COMP"])')
    p = p.replace(
        'artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")',
        'artifacts_path = Path("/tmp/_nonexistent_artifacts")')
    p = p.replace("cache=True", "cache=False")
    # --- deterministic per-well numba seeding ---
    # add an njit helper that seeds numba's internal RNG, + stdlib zlib for a stable hash
    p = p.replace("from numba import njit, prange\n",
                  "from numba import njit, prange\nimport zlib\n")
    # define the seed helper right before build_well
    p = p.replace("def build_well(hw_path, tw_path, is_train):\n",
                  "@njit(cache=False)\n"
                  "def _seed_numba(s):\n    np.random.seed(s)\n\n\n"
                  "def build_well(hw_path, tw_path, is_train):\n", 1)
    # seed per-well (stable crc32 of wid) right after wid is computed, before any PF/DTW
    p = p.replace(
        "    wid = Path(hw_path).stem.replace('__horizontal_well', '')\n",
        "    wid = Path(hw_path).stem.replace('__horizontal_well', '')\n"
        "    _seed_numba(int(zlib.crc32(wid.encode()) & 0x7fffffff))\n", 1)
    return p


ns = {"os": os}
exec(compile(patched_prefix(), str(SRC), "exec"), ns)
build_dataset = ns["build_dataset"]; CFG = ns["CFG"]
test_paths = sorted((CFG.dataset_path / "test").glob("*__horizontal_well.csv"))

print(">> build #1 (seeded)...", flush=True)
a = build_dataset(test_paths, is_train=False, label="t").set_index("id")
print(">> build #2 (seeded)...", flush=True)
b = build_dataset(test_paths, is_train=False, label="t").set_index("id").reindex(a.index)

STOCH = ["pf_ancc", "pf_ancc_std", "pf_ancc_delta", "pf_z", "pf_z_delta", "pf_vs_z",
         "pf_vs_spatial", "pf_vs_dense", "dtw_vs_pf", "dtw_stoch_mean_d", "dtw_stoch_std",
         "dtw_stoch_cv", "sig_std", "sig_mean_d"] + [f"tdpf{int(o)}" for o in
         (-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30)]
worst = 0.0
for c in STOCH:
    d = float(np.nanmean(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))))
    worst = max(worst, d)
print(f">> max mean|Δ| over the 25 previously-stochastic cols = {worst:.6e}", flush=True)
print(">> SEEDING WORKS (reproducible)" if worst < 1e-5 else ">> STILL NON-DETERMINISTIC", flush=True)
print("SEED VERIFY DONE", flush=True)
