"""Phase A of the 9.251 frontier reproduction: build its FULL feature union on OUR data.

Strategy (per user, 2026-05-31): the project's one big jump (12.6->11.9) came from
reproducing a better public notebook wholesale (konbu). To reach <10 we reproduce the next
rung: nihilisticneuralnet's LB-9.251 recipe, which adds PF(ANCC), PF(Z), multi-scale +
stochastic DTW, 7 beam configs, and multi-scale NCC on top of konbu's plane-KNN/row-KNN/
target-distance base -> ~150 features -> LGB x3 + CatBoost x3 -> hill-climb -> Optuna+SG.

We sourced their notebook code VERBATIM (only patching the data path + throttle + dropping
modeling/plot imports) so this is a faithful reproduction, not a reinterpretation. Our prior
ablations found PF/DTW/NCC individually null/dead, but always as SOLO gates; the bet is that
the tuned simultaneous UNION carries joint signal (the GP/extractor episodes proved solo gates
mislead in both directions). This script builds + caches the features; Phase B trains/blends.

Output: data/processed/frontier/{train,test}_feats.parquet
Run: nohup python -u experiments/frontier_repro_build.py > log/frontier_build_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import re
import time
import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
SRC = Path("/tmp/nihilisticneuralnet_9-251-rogii-wellbore-geology-prediction-dwt-based.code.py")
OUT = ROOT / "data/processed/frontier"
OUT.mkdir(parents=True, exist_ok=True)

# --- source their feature-build code (cells 1-4: njit estimators, imputers, build_well,
#     build_dataset, plus FI/DI/hw_paths instantiation). Stop before CODE CELL 5 (modeling). ---
text = SRC.read_text()
prefix = text.split("# ===== CODE CELL 5 =====")[0]

# drop imports we don't need for the build (and that aren't installed / are Kaggle-only)
for bad in ("from hill_climbing import Climber\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import optuna\n"):
    prefix = prefix.replace(bad, "")

# point at our data (has train/, test/, sample_submission.csv)
prefix = prefix.replace(
    'dataset_path = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")',
    f'dataset_path = Path("{ROOT / "data/raw"}")')
# their artifacts cache won't exist -> harmless, but neutralize the path anyway
prefix = prefix.replace(
    'artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")',
    'artifacts_path = Path("/tmp/_nonexistent_artifacts")')
# un-throttle: their NCPU = min(4, cpu_count()); use more cores on skynet
prefix = re.sub(r"NCPU = min\(4, multiprocessing\.cpu_count\(\)\)",
                "NCPU = min(16, multiprocessing.cpu_count())", prefix)
# numba on-disk cache fails to pickle on this aarch64 build ('cannot pickle PyCapsule');
# compile in-memory instead (only affects startup time, not results).
prefix = prefix.replace("cache=True", "cache=False")

ns = {}
print(">> compiling + instantiating imputers (sources 9.251 feature code)...", flush=True)
t0 = time.time()
exec(compile(prefix, str(SRC), "exec"), ns)
print(f"   ready in {time.time()-t0:.0f}s; NCPU={ns.get('NCPU')}, "
      f"train wells={len(ns['hw_paths'])}", flush=True)

build_dataset = ns["build_dataset"]
hw_paths = ns["hw_paths"]
CFG = ns["CFG"]
test_paths = sorted((CFG.dataset_path / "test").glob("*__horizontal_well.csv"))
print(f"   test wells={len(test_paths)}", flush=True)

print(">> building TRAIN feature union (PF + DTW + beams + NCC + KNN over 773 wells)...", flush=True)
t0 = time.time()
train_df = build_dataset(hw_paths, is_train=True, label="train")
print(f"   train_df {train_df.shape} in {time.time()-t0:.0f}s "
      f"({(train_df.shape[1])} cols)", flush=True)
train_df.to_parquet(OUT / "train_feats.parquet")

print(">> building TEST feature union...", flush=True)
t0 = time.time()
test_df = build_dataset(test_paths, is_train=False, label="test")
print(f"   test_df {test_df.shape} in {time.time()-t0:.0f}s", flush=True)
test_df.to_parquet(OUT / "test_feats.parquet")

feats = [c for c in train_df.columns if c not in {"well", "id", "target"}]
print(f">> #features={len(feats)}; cached -> {OUT}", flush=True)
print(f"   sample feats: {feats[:20]}", flush=True)
print("=== FRONTIER BUILD DONE ===", flush=True)
