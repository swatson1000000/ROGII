"""Measure per-well cost of the 9.251 feature build to give a real ETA.
Sources the same build code, builds N wells SERIALLY (n_jobs=1), times it.
"""
from pathlib import Path
import re, time
ROOT = Path("/home/swatson/work/kaggle/ROGII")
SRC = Path("/tmp/nihilisticneuralnet_9-251-rogii-wellbore-geology-prediction-dwt-based.code.py")
text = SRC.read_text()
prefix = text.split("# ===== CODE CELL 5 =====")[0]
for bad in ("from hill_climbing import Climber\n", "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n", "import optuna\n"):
    prefix = prefix.replace(bad, "")
prefix = prefix.replace(
    'dataset_path = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")',
    f'dataset_path = Path("{ROOT / "data/raw"}")')
prefix = prefix.replace(
    'artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")',
    'artifacts_path = Path("/tmp/_nonexistent_artifacts")')
prefix = prefix.replace("cache=True", "cache=False")
ns = {}
t0 = time.time()
exec(compile(prefix, str(SRC), "exec"), ns)
print(f"setup (imputers+compile): {time.time()-t0:.0f}s", flush=True)
build_well = ns["build_well"]; hw_paths = ns["hw_paths"]
N = 6
t0 = time.time()
for p in hw_paths[:N]:
    wid = p.stem.replace("__horizontal_well", "")
    tp = p.parent / f"{wid}__typewell.csv"
    build_well(str(p), str(tp), True)
dt = time.time() - t0
print(f"SERIAL: {N} wells in {dt:.1f}s -> {dt/N:.2f}s/well", flush=True)
print(f"ETA for 773 wells @ ~4.7x parallel: {773*(dt/N)/4.7/60:.0f} min "
      f"(serial-equiv {773*(dt/N)/60:.0f} min)", flush=True)
print("PROBE DONE", flush=True)
