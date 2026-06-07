"""Process-parallel rebuild of the 9.251 feature union (replaces the GIL-bound threaded build).

The notebook's build_dataset uses joblib Parallel(prefer='threads'); on this many-core box
40 threads spin on the GIL and waste most cores (the original run burned 180+ CPU-min with
no telemetry and didn't finish). This driver instead:
  1. execs the patched 9.251 feature code into THIS module's globals (sets up FI, DI,
     build_well, all njit funcs) ONCE in the parent — reads CSVs + compiles numba once.
  2. forks a ProcessPoolExecutor: each worker inherits FI/DI + compiled njit via copy-on-write
     (read-only -> shared pages, low memory), no pickling of big objects, no recompile.
  3. processes wells in parallel with REAL multi-core scaling + per-chunk progress logging.

Same features/results as experiments/frontier_repro_build.py (sources the same code verbatim);
only the parallelism backend differs. Output: data/processed/frontier/{train,test}_feats.parquet
Run: nohup python -u experiments/frontier_build_mp.py > log/frontier_mp_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
import os
import time
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
SRC = Path("/tmp/nihilisticneuralnet_9-251-rogii-wellbore-geology-prediction-dwt-based.code.py")
OUT = ROOT / "data/processed/frontier_seeded"
OUT.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = min(20, (os.cpu_count() or 8))

# ---- source the 9.251 feature code into THIS module's globals (parent only) ----
_text = SRC.read_text()
_prefix = _text.split("# ===== CODE CELL 5 =====")[0]
for _bad in ("from hill_climbing import Climber\n", "import matplotlib.pyplot as plt\n",
             "import seaborn as sns\n", "import optuna\n"):
    _prefix = _prefix.replace(_bad, "")
_prefix = _prefix.replace(
    'dataset_path = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")',
    f'dataset_path = Path("{ROOT / "data/raw"}")')
_prefix = _prefix.replace(
    'artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")',
    'artifacts_path = Path("/tmp/_nonexistent_artifacts")')
_prefix = _prefix.replace("cache=True", "cache=False")  # aarch64 numba pickle bug
# --- deterministic per-well numba seeding (PF/stochastic-DTW reproducibility, verified max|d|=0) ---
_prefix = _prefix.replace("from numba import njit, prange\n",
                          "from numba import njit, prange\nimport zlib\n")
_prefix = _prefix.replace("def build_well(hw_path, tw_path, is_train):\n",
                          "@njit(cache=False)\ndef _seed_numba(s):\n    np.random.seed(s)\n\n\n"
                          "def build_well(hw_path, tw_path, is_train):\n", 1)
_prefix = _prefix.replace(
    "    wid = Path(hw_path).stem.replace('__horizontal_well', '')\n",
    "    wid = Path(hw_path).stem.replace('__horizontal_well', '')\n"
    "    _seed_numba(int(zlib.crc32(wid.encode()) & 0x7fffffff))\n", 1)
exec(compile(_prefix, str(SRC), "exec"), globals())
# now in globals(): FI, DI, build_well, run_pf_ancc/_z, run_dtw_*, beam_search, CFG, hw_paths, ...


def _worker(arg):
    """Top-level (picklable) per-well worker; uses inherited FI/DI/build_well via fork."""
    hp, is_train = arg
    wid = Path(hp).stem.replace("__horizontal_well", "")
    tp = Path(hp).parent / f"{wid}__typewell.csv"
    if not tp.exists():
        return None
    try:
        return build_well(str(hp), str(tp), is_train)  # noqa: F821 (from exec)
    except Exception as e:
        print(f"   [well {wid}] FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
        return None


def _build(paths, is_train, label):
    args = [(str(p), is_train) for p in paths]
    parts = []
    done = 0
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
        for r in ex.map(_worker, args, chunksize=2):
            done += 1
            if r is not None:
                parts.append(r)
            if done % 50 == 0 or done == len(args):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(args) - done) / max(rate, 1e-9)
                print(f"   {label}: {done}/{len(args)} wells  "
                      f"({rate:.1f}/s, ETA {eta/60:.1f} min)", flush=True)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    print(f"   {label}: built {df.shape} in {(time.time()-t0)/60:.1f} min", flush=True)
    return df


if __name__ == "__main__":
    print(f">> imputers ready (parent); MAX_WORKERS={MAX_WORKERS}; "
          f"train wells={len(hw_paths)}", flush=True)  # noqa: F821
    test_paths = sorted((CFG.dataset_path / "test").glob("*__horizontal_well.csv"))  # noqa: F821

    print(">> building TRAIN feature union (process-parallel)...", flush=True)
    train_df = _build(hw_paths, True, "train")  # noqa: F821
    train_df.to_parquet(OUT / "train_feats.parquet")

    print(">> building TEST feature union...", flush=True)
    test_df = _build(test_paths, False, "test")
    test_df.to_parquet(OUT / "test_feats.parquet")

    feats = [c for c in train_df.columns if c not in {"well", "id", "target"}]
    print(f">> #features={len(feats)}; cached -> {OUT}", flush=True)
    print(f"   sample feats: {feats[:20]}", flush=True)
    print("=== FRONTIER MP BUILD DONE ===", flush=True)
