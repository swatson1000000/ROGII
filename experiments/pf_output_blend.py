"""DECISIVE: does output-blending the 128-seed likelihood-weighted PF (standalone 10.993)
with the frontier GBM stack (10.356) actually buy anything?

The standalone gate (pf_real_gate.py) landed 'PARTIAL/marginal' (10.993, above the 10.5 pass
bar). The pre-registered follow-up: measure the real output-blend gain. The single-seed PF (the
one the GBM already eats as a feature, 14.37) output-blended at only w=0.08 / +0.042 ft. This
PF is far better standalone -> measure its actual blend gain + optimal weight on OUR OOF.

Works entirely in RESIDUAL space (both residuals are vs the same true TVT), so no last_known
arithmetic is needed for the blend; true_tvt = target + last_known_tvt is used only to align.

Alignment: train_feats rows (OOF order) <-> PF PKL (sorted-glob order). PF PKL stores no ids,
so we replay sorted(glob)+skip logic to map each PF result to a well, then verify per-well by
matching the reconstructed true-TVT sequence. Refuses to blend any well that fails the check.
"""
from pathlib import Path
import glob, os, json
import numpy as np
import pandas as pd
import joblib

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
FR = ROOT / "data/processed/frontier_seeded"
MODELS = ROOT / "models/frontier"
PF_KEY = "pf_scale_12"  # best likelihood-weighted scale from the gate

print(">> loading frontier OOF + meta...", flush=True)
df = pd.read_parquet(FR / "train_feats.parquet", columns=["well", "id", "last_known_tvt", "target"])
blend = json.load(open(MODELS / "blend_frontier.json"))
keys, coef = blend["keys"], np.array(blend["ridge_coef"], np.float64)
oofs = np.column_stack([np.load(MODELS / f"oof_{k}.npy").astype(np.float64) for k in keys])
blended_drift = oofs @ coef
target = df["target"].to_numpy(np.float64)
lastk = df["last_known_tvt"].to_numpy(np.float64)
gbm_resid = blended_drift - target          # = GBM_abs - true_tvt
true_tvt = target + lastk
print(f"   GBM blended OOF RMSE (drift) = {np.sqrt(np.mean(gbm_resid**2)):.4f}  (expect 10.3556)", flush=True)

# row index parsed from id ('<well>_<rowidx>') to order within well
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)

# ---- replay sorted(glob) + the gate's skip logic to recover the well id for each PF result ----
print(">> replaying PF well order...", flush=True)
paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells = []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue  # process() returned None -> not in `good`
    pf_wells.append(wid)

results = joblib.load(MODELS / "pf_real_results.pkl")
def is_err(r): return r is not None and isinstance(r[0], str) and r[0] == "ERR"
good = [r for r in results if r is not None and not is_err(r)]
print(f"   PF good wells = {len(good)}  replay wells = {len(pf_wells)}", flush=True)
assert len(good) == len(pf_wells), "replay count mismatch -> skip logic drifted; aborting"

# ---- per-well align + collect residuals ----
gbm_keep, pf_keep = [], []
n_ok = n_bad = 0
grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}
for wid, (pf_truth, preds) in zip(pf_wells, good):
    sub = grp.get(wid)
    if sub is None or len(sub) != len(pf_truth):
        n_bad += 1; continue
    t_tvt = sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64)
    if not np.allclose(t_tvt, pf_truth, atol=1e-3, rtol=0):
        n_bad += 1; continue
    n_ok += 1
    gbm_keep.append(sub.index.to_numpy())            # row positions in df
    pf_keep.append(preds[PF_KEY].astype(np.float64) - pf_truth)  # PF_abs - true_tvt
print(f"   aligned wells: ok={n_ok} bad={n_bad}", flush=True)

pos = np.concatenate(gbm_keep)
g_res = gbm_resid[pos]
p_res = np.concatenate(pf_keep)
print(f"   blendable eval rows = {len(g_res):,}  ({len(g_res)/len(gbm_resid)*100:.1f}% of OOF)", flush=True)
print(f"   corr(gbm_resid, pf_resid) = {np.corrcoef(g_res, p_res)[0,1]:.3f}", flush=True)

base = np.sqrt(np.mean(g_res**2))
print(f"\n=== OUTPUT-BLEND SCAN (RMSE on the {n_ok} aligned wells) ===", flush=True)
print(f"  w=0.00 (GBM only)  RMSE={base:.4f}", flush=True)
best_w, best_r = 0.0, base
for w in np.arange(0.02, 0.71, 0.02):
    r = np.sqrt(np.mean(((1 - w) * g_res + w * p_res) ** 2))
    if r < best_r:
        best_r, best_w = r, w
# closed-form optimum for a 2-vector blend
d = g_res - p_res
w_star = float(np.dot(g_res, d) / np.dot(d, d))
r_star = np.sqrt(np.mean(((1 - w_star) * g_res + w_star * p_res) ** 2))
print(f"  grid-best  w={best_w:.2f}  RMSE={best_r:.4f}  gain={base-best_r:+.4f}", flush=True)
print(f"  closed-form w*={w_star:.3f}  RMSE={r_star:.4f}  gain={base-r_star:+.4f}", flush=True)

# per-fold robustness of w* (GKF-5 seed42 as in training) -- is the weight stable / does the
# gain hold when w is fit out-of-fold?
rng = np.random.RandomState(42)
uw = df["well"].unique().copy(); rng.shuffle(uw)
fold_of = {w: i % 5 for i, w in enumerate(uw)}
well_of_pos = df["well"].to_numpy()[pos]
fold = np.array([fold_of[w] for w in well_of_pos])
oof_blend = g_res.copy()
ws = []
for f in range(5):
    tr, va = fold != f, fold == f
    dd = g_res[tr] - p_res[tr]
    wf = float(np.dot(g_res[tr], dd) / np.dot(dd, dd))
    ws.append(wf)
    oof_blend[va] = (1 - wf) * g_res[va] + wf * p_res[va]
r_oofw = np.sqrt(np.mean(oof_blend**2))
print(f"  out-of-fold w (per fold {[round(x,3) for x in ws]}) -> RMSE={r_oofw:.4f}  gain={base-r_oofw:+.4f}", flush=True)

print("\n=== VERDICT ===", flush=True)
gain = base - r_oofw
if gain >= 0.10:
    print(f"  BANKABLE: out-of-fold output-blend gain {gain:+.4f} ft -> productionize (add PF to kernel,", flush=True)
    print("  blend at w* in absolute space), gate, submit. PF isn't density-coupled -> lower transfer risk.", flush=True)
elif gain >= 0.03:
    print(f"  MARGINAL: {gain:+.4f} ft -> a free fold-in at best (below the ~0.23 OOF<->LB resolution).", flush=True)
    print("  Not worth a standalone submission; fold into any future kernel rebuild, else DROP.", flush=True)
else:
    print(f"  NULL: {gain:+.4f} ft -> the better PF adds nothing the GBM doesn't already have. DROP the lever.", flush=True)
print("PFBLEND DONE", flush=True)
