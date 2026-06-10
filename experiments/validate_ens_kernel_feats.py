"""Validate the dip+cwt feature port in the regenerated kernel against the train-side ground-truth
parquets (dip_feats.parquet / cwt_feats.parquet). Extracts the EXACT helper functions from the
generated kernel, reproduces build_well's per-well indexing on a sample of train wells, and diffs.
Pass = max|delta| ~ 0 on all 8 computed cols (dwt_vs_sc derived from sc15_d is checked separately)."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
KERN = ROOT / "jupyter_frontier/rogii_frontier_inference.py"

# --- extract the 3 helper fns from the kernel (def well_dip_feats ... up to def build_well) ---
src = KERN.read_text()
seg = src[src.index("def well_dip_feats"):src.index("def build_well")]
ns = {"np": np, "pd": pd}
exec(seg, ns)
well_dip_feats = ns["well_dip_feats"]; _cwt_detail = ns["_cwt_detail"]; detail_ncc = ns["detail_ncc"]
print(">> extracted kernel helpers: well_dip_feats, _cwt_detail, detail_ncc", flush=True)

dip_gt = pd.read_parquet(ROOT / "data/processed/dip_feats.parquet").set_index("id")
cwt_gt = pd.read_parquet(ROOT / "data/processed/cwt_feats.parquet").set_index("id")
DIP_COLS = ["dogleg", "cum_dogleg", "tvt_dip_grad", "tvt_dip_grad_z", "quad_b_d"]
CWT_COLS = ["dwt_ncc_d", "dwt_ncc_sc", "gr_detail_std"]

# sample wells that exist in the gt + on disk
wells = [p.stem.replace("__horizontal_well", "") for p in sorted(RAW.glob("*__horizontal_well.csv"))]
rng = np.random.RandomState(0); sample = list(rng.choice(wells, size=12, replace=False))

maxd = {c: 0.0 for c in DIP_COLS + CWT_COLS}; n_checked = 0
for wid in sample:
    hw = pd.read_csv(RAW / f"{wid}__horizontal_well.csv")
    if "TVT_input" not in hw.columns:
        continue
    ev_mask = hw["TVT_input"].isna().to_numpy()
    if ev_mask.sum() == 0:
        continue
    _eidx = np.where(ev_mask)[0]
    ids = [f"{wid}_{i}" for i in _eidx]
    keep = [i for i, x in enumerate(ids) if x in dip_gt.index]
    if not keep:
        continue
    # --- dip (mirror build_well wiring) ---
    dip = well_dip_feats(hw["MD"].to_numpy(np.float64), hw["X"].to_numpy(np.float64),
                         hw["Y"].to_numpy(np.float64), hw["Z"].to_numpy(np.float64),
                         hw["TVT_input"].to_numpy(np.float64))
    # --- cwt (mirror build_well wiring) ---
    kn_mask = hw["TVT_input"].notna().to_numpy()
    if int(kn_mask.sum()) >= 40:
        gr_det = _cwt_detail(hw["GR"].to_numpy(np.float64))
        ktvt_c = hw["TVT_input"].to_numpy(np.float64)[kn_mask]; last_c = float(ktvt_c[-1])
        dwt_tvt, dwt_sc = detail_ncc(gr_det[kn_mask], ktvt_c.astype(np.float32), gr_det[_eidx])
        dwt_d = (dwt_tvt - np.float32(last_c)).astype(np.float32)
        det_std = pd.Series(gr_det).rolling(15, center=True, min_periods=1).std().fillna(0.).values[_eidx].astype(np.float32)
    else:
        dwt_d = np.zeros(len(_eidx), np.float32); dwt_sc = np.zeros(len(_eidx), np.float32); det_std = np.zeros(len(_eidx), np.float32)
    got = {"dogleg": dip["dogleg"][_eidx], "cum_dogleg": dip["cum_dogleg"][_eidx],
           "tvt_dip_grad": dip["tvt_dip_grad"][_eidx], "tvt_dip_grad_z": dip["tvt_dip_grad_z"][_eidx],
           "quad_b_d": dip["quad_b_d"][_eidx], "dwt_ncc_d": dwt_d, "dwt_ncc_sc": dwt_sc, "gr_detail_std": det_std}
    ids_k = [ids[i] for i in keep]
    for c in DIP_COLS:
        d = np.abs(np.asarray(got[c])[keep] - dip_gt.loc[ids_k, c].to_numpy()); maxd[c] = max(maxd[c], float(d.max()))
    for c in CWT_COLS:
        d = np.abs(np.asarray(got[c])[keep] - cwt_gt.loc[ids_k, c].to_numpy()); maxd[c] = max(maxd[c], float(d.max()))
    n_checked += len(keep)

print(f">> checked {n_checked} eval rows across {len(sample)} sampled wells", flush=True)
print("=== max|kernel - ground_truth| per feature ===", flush=True)
ok = True
for c in DIP_COLS + CWT_COLS:
    flag = "OK" if maxd[c] < 1e-3 else "MISMATCH"
    if maxd[c] >= 1e-3: ok = False
    print(f"  {c:16s} max|d|={maxd[c]:.3e}  {flag}", flush=True)
print("=== KERNEL FEAT PORT VALIDATION", "PASS ===" if ok else "FAIL ===", flush=True)
