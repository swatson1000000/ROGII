"""Process-parallel build of the romantamrazov "SUPER SOLUTION (top-3)" feature union.

Ports the build_well from /tmp/rogii_top3_code.py wholesale (the proven reproduce-wholesale
playbook: konbu 12.6->11.9, frontier 11.9->10.1). It is our frontier-222 skeleton
(PF/beam/NCC/plane+dense-KNN) PLUS ~10 new feature families: WLS b_well, per-formation
known-zone RMSE, formation-consensus std/range, inter-signal std, GR envelope/energy/detrend,
prefix GR slope, a 4th tw_diff family anchored at PF-ANCC, and multi-scale NCC hw=8/15/25.

Mirrors experiments/frontier_build_mp.py (same fork-based ProcessPoolExecutor harness):
  1. exec the source's feature code (everything before training) into THIS module's globals
     ONCE in the parent -> builds FI/DI imputers + compiles numba once.
  2. fork a ProcessPoolExecutor: workers inherit FI/DI + compiled njit via copy-on-write.
  3. process wells in parallel with per-chunk progress logging.

DETERMINISM: the super-solution's only randomness is np.random.* inside the pure-Python PFs
(run_pf_ancc/run_pf_z) -- there is NO DTW and the beam @njit uses no RNG. The source uses
joblib prefer='threads' which shares ONE global RNG across wells (non-reproducible). We instead
(a) run wells in separate processes and (b) seed np.random per well at the top of build_well
(crc32(wid)). Without this the train feats won't reproduce at inference (cost us 1.24 ft last time).

Output: data/processed/super/{train,test}_feats.parquet + feature_cols.json
Run: nohup python -u experiments/super_build.py > log/super_build_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
import os
import json
import time
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
SRC = Path("/tmp/rogii_top3_code.py")
# NOTE: do NOT name this OUT -- the exec'd source defines its own global OUT
# (=/kaggle/working/submission.csv) which would clobber ours. Use OUTDIR.
OUTDIR = ROOT / "data/processed/super"
OUTDIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = min(20, (os.cpu_count() or 8))

# ---- source the super-solution feature code into THIS module's globals (parent only) ----
# Take everything before the training section ("Building train..."); keep helpers, PFs,
# imputers, build_well, build_dataset. The exec builds FI/DI in the parent.
_text = SRC.read_text()
_prefix = _text.split('print("Building train...")')[0]

# Patch 0: drop `from __future__ import annotations` (not first stmt once exec'd; runtime no-op).
_prefix = _prefix.replace("from __future__ import annotations\n", "")
# Patch 1: numba cache dir off the (nonexistent) /kaggle path.
_prefix = _prefix.replace("/kaggle/working/.numba", "/tmp/super_numba")
# Patch 2: aarch64 numba pickle bug with cache=True -> disable njit disk cache.
_prefix = _prefix.replace("cache=True", "cache=False")
# Patch 3: point the data resolver at our local raw data.
_prefix = _prefix.replace("DATA=_find()", f'DATA=Path("{ROOT / "data/raw"}")')
# Patch 4: deterministic per-well np.random seeding (PF reproducibility, verified max|d|=0).
_prefix = _prefix.replace(
    "import gc, time, multiprocessing, warnings",
    "import gc, time, multiprocessing, warnings, zlib")
_prefix = _prefix.replace(
    "    wid=Path(hw_path).stem.replace('__horizontal_well','')\n",
    "    wid=Path(hw_path).stem.replace('__horizontal_well','')\n"
    "    np.random.seed(int(zlib.crc32(wid.encode()) & 0x7fffffff))\n", 1)

exec(compile(_prefix, str(SRC), "exec"), globals())
# now in globals(): FI, DI, build_well, run_pf_ancc/_z, beam_search, TRAIN_DIR, TEST_DIR, hw_paths, ...


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
    test_paths = sorted((TEST_DIR).glob("*__horizontal_well.csv"))  # noqa: F821

    print(">> building TRAIN feature union (process-parallel)...", flush=True)
    train_df = _build(hw_paths, True, "train")  # noqa: F821
    train_df.to_parquet(OUTDIR / "train_feats.parquet")

    print(">> building TEST feature union...", flush=True)
    test_df = _build(test_paths, False, "test")
    test_df.to_parquet(OUTDIR / "test_feats.parquet")

    feats = [c for c in train_df.columns if c not in {"well", "id", "target"}]
    (OUTDIR / "feature_cols.json").write_text(json.dumps(feats))
    print(f">> #features={len(feats)}; cached -> {OUTDIR}", flush=True)
    print(f"   sample feats: {feats[:20]}", flush=True)
    print("=== SUPER BUILD DONE ===", flush=True)
