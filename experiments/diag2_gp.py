"""Instrument gp_tvt_abs / b_well for one well: kernel recompute vs build (gp_feats_test)."""
from pathlib import Path
import glob, json
import numpy as np
import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
src = (ROOT / "jupyter_konbu/rogii_konbu_inference.py").read_text()
head = src.split("# ---------------- Kaggle inference main")[0]
ns = {}
exec(head, ns)
FormationGP = ns["FormationGP"]

wid = "000d7d20"
train_paths = [Path(p) for p in sorted(glob.glob(str(ROOT / "data/raw/train/*__horizontal_well.csv")))]
anchor = json.load(open(ROOT / "models/konbu_gp/gp_anchor.json"))
gp = FormationGP(train_paths, anchor)

h = pd.read_csv(ROOT / f"data/raw/test/{wid}__horizontal_well.csv")
mask = h["TVT_input"].isna().to_numpy()
ms = int(np.flatnonzero(mask)[0])
xy_full = h[["X", "Y"]].to_numpy(np.float64)
gp_ancc_full, gp_std_full = gp.impute(xy_full)
known = h.iloc[:ms]
known_tvt = known["TVT_input"].to_numpy(np.float32)
known_z = known["Z"].to_numpy(np.float32)
z_post = h.iloc[ms:]["Z"].to_numpy(np.float32)
b_kernel = float(np.median(known_tvt + known_z - gp_ancc_full[:ms]))
gp_tvt_abs_kernel = -z_post + gp_ancc_full[ms:] + np.float32(b_kernel)

# build side
gte = pd.read_parquet(ROOT / "data/processed/konbu/gp_feats_test.parquet")
g = gte[gte["well"] == wid].sort_values("row_idx")
g_ancc = g["gp_ancc"].to_numpy(np.float64)
g_tvt_abs = g["gp_tvt_abs"].to_numpy(np.float64)
g_z = h["Z"].to_numpy(np.float64)  # full Z aligned to row order
# implied build b: gp_tvt_abs = -z + ancc + b  -> b = gp_tvt_abs + z - ancc
b_build_implied = g_tvt_abs + g_z - g_ancc

print(f"well {wid}  ms={ms}  n={len(h)}", flush=True)
print(f"len(g)={len(g)}  len(h)={len(h)}  match={len(g)==len(h)}", flush=True)
print(f"b_kernel (median prefix) = {b_kernel:.4f}", flush=True)
print(f"b_build_implied: median={np.median(b_build_implied):.4f}  std={np.std(b_build_implied):.4f}  "
      f"first5={np.round(b_build_implied[:5],3)}", flush=True)
print(f"gp_ancc kernel[:3]={np.round(gp_ancc_full[:3],4)}  build[:3]={np.round(g_ancc[:3],4)}", flush=True)
print(f"Z kernel-from-h[:3]={np.round(h['Z'].to_numpy()[:3],3)}", flush=True)
print(f"gp_tvt_abs kernel post[:3]={np.round(gp_tvt_abs_kernel[:3],3)}", flush=True)
print(f"gp_tvt_abs build  post[:3]={np.round(g_tvt_abs[ms:ms+3],3)}", flush=True)
print(f"diff gp_tvt_abs post mean = {np.mean(gp_tvt_abs_kernel - g_tvt_abs[ms:]):.4f}", flush=True)
print(f"last_known_tvt = {float(known.iloc[-1]['TVT_input']):.3f}", flush=True)
# is g_tvt_abs already a DRIFT (small) rather than absolute?
print(f"g_tvt_abs stats: min={g_tvt_abs.min():.2f} max={g_tvt_abs.max():.2f} mean={g_tvt_abs.mean():.2f}", flush=True)
print("=== DIAG2 DONE ===", flush=True)
