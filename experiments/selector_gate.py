"""GATE: does a PER-WELL PF-vs-GBM blend weight (routed by n_eval/z_span) beat the global w=0.57?

The last untested piece of the public ravaghi/pilkwang architecture: instead of a global output
weight, pick w per well from well characteristics. Gated OFFLINE (0 submissions) on cached
residuals. Discipline: the weight function is fit OUT-OF-FOLD (GKF-5 seed42) — an in-sample
per-well optimum is trivially overfit and meaningless.

Reports:
  global     : single w fit out-of-fold (reproduces ~0.57 / OOF ~9.30) -- baseline
  oracle     : best per-well w fit IN-SAMPLE -- the CEILING the selector could ever reach
  diagnostic : corr(per-well w*, log n_eval) and corr(w*, z_span) -- do the routing features
               even predict the optimal weight? If ~0, a selector on them CANNOT work.
  binned     : 6 bins (n_eval x z_span quantiles), per-bin w fit OUT-OF-FOLD -- the honest selector

Gate: binned out-of-fold gain over global >= ~0.05 ft -> productionize; else NULL.
Reuses pf_weight_curve.py's verified alignment.
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
PF_KEY = "pf_scale_12"

print(">> loading + aligning (GBM oof, PF pkl, well features)...", flush=True)
df = pd.read_parquet(FR / "train_feats.parquet", columns=["well", "id", "last_known_tvt", "target", "z"])
blend = json.load(open(MODELS / "blend_frontier.json"))
coef = np.array(blend["ridge_coef"], np.float64)
oofs = np.column_stack([np.load(MODELS / f"oof_{k}.npy").astype(np.float64) for k in blend["keys"]])
gbm_resid = (oofs @ coef) - df["target"].to_numpy(np.float64)
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)

paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells = []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue
    pf_wells.append(wid)
results = joblib.load(MODELS / "pf_real_results.pkl")
def is_err(r): return r is not None and isinstance(r[0], str) and r[0] == "ERR"
good = [r for r in results if r is not None and not is_err(r)]
assert len(good) == len(pf_wells)

grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}
# per-row arrays + per-well bookkeeping, all in PF-aligned order
g_list, p_list, wellid_list = [], [], []
well_order = []
for wid, (pf_truth, preds) in zip(pf_wells, good):
    sub = grp.get(wid)
    if sub is None or len(sub) != len(pf_truth): continue
    t = sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64)
    if not np.allclose(t, pf_truth, atol=1e-3): continue
    idx = sub.index.to_numpy()
    g_list.append(gbm_resid[idx]); p_list.append(preds[PF_KEY].astype(np.float64) - pf_truth)
    wellid_list.append(np.full(len(idx), len(well_order)))
    well_order.append((wid, sub["z"].to_numpy(np.float64)))
g = np.concatenate(g_list); p = np.concatenate(p_list); wi = np.concatenate(wellid_list)
nW = len(well_order)
print(f"   aligned {nW} wells, {len(g):,} rows", flush=True)

# fold map (GKF-5 seed42) at WELL level
rng = np.random.RandomState(42)
uw = df["well"].unique().copy(); rng.shuffle(uw)
fold_of_name = {w: i % 5 for i, w in enumerate(uw)}
well_fold = np.array([fold_of_name[wid] for wid, _ in well_order])
row_fold = well_fold[wi]

# per-well features: n_eval (rows), z_span (range of z within well)
n_eval = np.array([np.sum(wi == k) for k in range(nW)], float)
z_span = np.array([(zz.max() - zz.min()) if len(zz) else 0.0 for _, zz in well_order], float)

rmse = lambda r: float(np.sqrt(np.mean(r * r)))
def blend_rmse(wrow):  # wrow: per-row weight
    return rmse((1 - wrow) * g + wrow * p)

# ---- baseline: GLOBAL w, out-of-fold ----
gw = np.empty(len(g))
for f in range(5):
    tr, va = row_fold != f, row_fold == f
    d = g[tr] - p[tr]; w = float(np.dot(g[tr], d) / np.dot(d, d))
    gw[va] = w
global_rmse = blend_rmse(gw)
# single global on all data (for reference)
d_all = g - p; w_all = float(np.dot(g, d_all) / np.dot(d_all, d_all))
print(f"\n  global (out-of-fold w)   RMSE={global_rmse:.4f}   (full-data w*={w_all:.3f})", flush=True)

# ---- per-well ORACLE (in-sample ceiling) ----
def well_wstar(k):
    m = wi == k; gg = g[m]; pp = p[m]; d = gg - pp; den = np.dot(d, d)
    return float(np.dot(gg, d) / den) if den > 1e-9 else w_all
wstar = np.array([well_wstar(k) for k in range(nW)])
wstar_cl = np.clip(wstar, -0.5, 1.5)  # tame degenerate tiny-|g-p| wells
oracle_row = wstar_cl[wi]
oracle_rmse = blend_rmse(oracle_row)
print(f"  per-well ORACLE (in-sample CEILING)  RMSE={oracle_rmse:.4f}   gain_vs_global={global_rmse-oracle_rmse:+.4f}", flush=True)
print(f"     per-well w*: median {np.median(wstar_cl):.2f}  IQR [{np.percentile(wstar_cl,25):.2f},{np.percentile(wstar_cl,75):.2f}]  (global {w_all:.2f})", flush=True)

# ---- DIAGNOSTIC: do routing features predict the optimal per-well weight? ----
# weight per well by n_eval so the corr reflects what matters for the row-RMSE
def wcorr(x, w, wt):
    xm = np.average(x, weights=wt); wm = np.average(w, weights=wt)
    cov = np.average((x - xm) * (w - wm), weights=wt)
    sx = np.sqrt(np.average((x - xm) ** 2, weights=wt)); sw = np.sqrt(np.average((w - wm) ** 2, weights=wt))
    return float(cov / (sx * sw)) if sx > 0 and sw > 0 else 0.0
c_ne = wcorr(np.log(n_eval), wstar_cl, n_eval)
c_zs = wcorr(z_span, wstar_cl, n_eval)
print(f"  diagnostic: corr(w*, log n_eval)={c_ne:+.3f}   corr(w*, z_span)={c_zs:+.3f}"
      f"   (|corr|~0 -> features can't route)", flush=True)

# ---- BINNED selector: 3 (n_eval) x 2 (z_span) quantile bins, per-bin w OUT-OF-FOLD ----
ne_b = np.digitize(n_eval, np.quantile(n_eval, [1/3, 2/3]))      # 0,1,2
zs_b = np.digitize(z_span, np.quantile(z_span, [0.5]))           # 0,1
well_bin = ne_b * 2 + zs_b                                       # 0..5
row_bin = well_bin[wi]
bw = np.empty(len(g))
for f in range(5):
    for b in range(6):
        tr = (row_fold != f) & (row_bin == b)
        va = (row_fold == f) & (row_bin == b)
        if va.sum() == 0: continue
        if tr.sum() < 500:
            d = g[row_fold != f] - p[row_fold != f]; w = float(np.dot(g[row_fold != f], d) / np.dot(d, d))
        else:
            d = g[tr] - p[tr]; w = float(np.dot(g[tr], d) / np.dot(d, d))
        bw[va] = w
binned_rmse = blend_rmse(bw)
print(f"\n  BINNED selector (6 bins, out-of-fold w)  RMSE={binned_rmse:.4f}   gain_vs_global={global_rmse-binned_rmse:+.4f}", flush=True)
print("     per-bin full-data w* (n_eval x z_span):", flush=True)
for b in range(6):
    m = row_bin == b
    if m.sum() == 0: continue
    d = g[m] - p[m]; w = float(np.dot(g[m], d) / np.dot(d, d))
    print(f"       bin {b} (ne={b//2} zs={b%2}): w*={w:.3f}  wells={int((well_bin==b).sum())}  rows={int(m.sum()):,}", flush=True)

print("\n=== VERDICT ===", flush=True)
gain = global_rmse - binned_rmse
ceil = global_rmse - oracle_rmse
print(f"  ceiling (per-well oracle)      = {ceil:+.4f} ft", flush=True)
print(f"  honest binned out-of-fold gain = {gain:+.4f} ft", flush=True)
if gain >= 0.05:
    print("  PASS: per-well routing beats the global weight -> productionize the selector (per-bin w by", flush=True)
    print("  n_eval/z_span in the kernel), gate, submit.", flush=True)
elif gain >= 0.02:
    print("  MARGINAL: below the ~0.02 LB resolution we measured -> not worth a submission / public-overfit risk.", flush=True)
else:
    print("  NULL: routing features don't separate the PF/GBM tradeoff (see diagnostic + flat ceiling).", flush=True)
    print("  The selector is the last public-architecture piece -> it's now closed. Bank 8.158 + harden.", flush=True)
print("SELECTOR DONE", flush=True)
