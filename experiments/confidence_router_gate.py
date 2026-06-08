"""GATE: can a CONFIDENCE/AGREEMENT router beat the global PF-vs-GBM weight where geometry couldn't?

selector_gate.py killed geometry routing (corr(w*, n_eval/z_span)~0) but found a per-well oracle
ceiling: the optimal weight varies hugely well-to-well. This tests whether INFERENCE-AVAILABLE
confidence features predict that per-well optimum: PF uncertainty (pf_ancc_std), PF-GBM disagreement
(|PF-GBM| = |p-g| in resid space, no label), prefix-fit quality (pfx_rmse), GR calibration (cal_a/b),
signal spread (sig_std, dense_std/rmse), known/eval length.

Discipline: predict per-well w* OUT-OF-FOLD (fit the router on 4 folds, apply to the 5th), weight
the fit by n_eval, clip predicted w to [0,1]. Pre-registered bar: OOF gain over global >= 0.10 ft
(high, because a meta-model that picks 'which model is right' is exactly the GP-style OOF->LB
transfer-risk regime). Also reports out-of-fold R^2 of the w* prediction (if ~0, null regardless).

Reuses selector_gate.py's alignment. 0 submissions.
"""
from pathlib import Path
import glob, os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
FR = ROOT / "data/processed/frontier_seeded"
MODELS = ROOT / "models/frontier"
PF_KEY = "pf_scale_12"
FEAT_COLS = ["pf_ancc_std", "pfx_rmse", "cal_a", "cal_b", "sig_std",
             "dense_std", "dense_rmse", "dense_nb_std", "known_len", "eval_len", "frm_rmse_ANCC"]

print(">> loading + aligning...", flush=True)
df = pd.read_parquet(FR / "train_feats.parquet",
                     columns=["well", "id", "last_known_tvt", "target", "z"] + FEAT_COLS)
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
g_list, p_list, wi_list, wrows = [], [], [], []
for wid, (pf_truth, preds) in zip(pf_wells, good):
    sub = grp.get(wid)
    if sub is None or len(sub) != len(pf_truth): continue
    t = sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64)
    if not np.allclose(t, pf_truth, atol=1e-3): continue
    idx = sub.index.to_numpy()
    gg = gbm_resid[idx]; pp = preds[PF_KEY].astype(np.float64) - pf_truth
    g_list.append(gg); p_list.append(pp); wi_list.append(np.full(len(idx), len(wrows)))
    feat = {c: float(np.nanmean(sub[c].to_numpy(np.float64))) for c in FEAT_COLS}
    feat["disagree"] = float(np.mean(np.abs(pp - gg)))      # |PF - GBM|, inference-available
    feat["disagree_sd"] = float(np.std(pp - gg))
    feat["z_span"] = float(sub["z"].max() - sub["z"].min())
    feat["log_neval"] = float(np.log(len(idx)))
    feat["well"] = wid
    wrows.append(feat)
g = np.concatenate(g_list); p = np.concatenate(p_list); wi = np.concatenate(wi_list)
W = pd.DataFrame(wrows); nW = len(W)
n_eval = np.array([np.sum(wi == k) for k in range(nW)], float)
print(f"   aligned {nW} wells, {len(g):,} rows", flush=True)

rng = np.random.RandomState(42)
uw = df["well"].unique().copy(); rng.shuffle(uw)
fold_of = {w: i % 5 for i, w in enumerate(uw)}
well_fold = np.array([fold_of[w] for w in W["well"]])
row_fold = well_fold[wi]

rmse = lambda r: float(np.sqrt(np.mean(r * r)))
def blend_rmse(wrow): return rmse((1 - wrow) * g + wrow * p)

# global out-of-fold baseline
gw = np.empty(len(g))
for f in range(5):
    tr, va = row_fold != f, row_fold == f
    d = g[tr] - p[tr]; gw[va] = float(np.dot(g[tr], d) / np.dot(d, d))
global_rmse = blend_rmse(gw)

# per-well w* target (clipped)
def wstar(k):
    m = wi == k; d = g[m] - p[m]; den = np.dot(d, d)
    return float(np.dot(g[m], d) / den) if den > 1e-9 else 0.44
ws = np.clip(np.array([wstar(k) for k in range(nW)]), -0.5, 1.5)

FEATS = FEAT_COLS + ["disagree", "disagree_sd", "z_span", "log_neval"]
X = W[FEATS].to_numpy(np.float64)
X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

print(f"\n  global (out-of-fold w)  RMSE={global_rmse:.4f}", flush=True)

def run_router(kind):
    wpred_well = np.empty(nW)
    r2s = []
    for f in range(5):
        tr, va = well_fold != f, well_fold == f
        sw = n_eval[tr]
        if kind == "ridge":
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=10.0).fit(sc.transform(X[tr]), ws[tr], sample_weight=sw)
            pr = m.predict(sc.transform(X[va]))
        else:
            m = lgb.LGBMRegressor(n_estimators=200, num_leaves=7, min_child_samples=40,
                                  learning_rate=0.03, reg_lambda=5.0, subsample=0.8,
                                  colsample_bytree=0.8, verbose=-1)
            m.fit(X[tr], ws[tr], sample_weight=sw)
            pr = m.predict(X[va])
        wpred_well[va] = pr
        # out-of-fold R^2 of predicting w*, n_eval-weighted
        yt = ws[va]; wv = n_eval[va]
        ybar = np.average(yt, weights=wv)
        ss_res = np.sum(wv * (yt - pr) ** 2); ss_tot = np.sum(wv * (yt - ybar) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    wpred_well = np.clip(wpred_well, 0.0, 1.0)
    r = blend_rmse(wpred_well[wi])
    return r, float(np.mean(r2s)), wpred_well

for kind in ["ridge", "gbm"]:
    r, r2, wp = run_router(kind)
    print(f"  router[{kind:5s}] RMSE={r:.4f}  gain_vs_global={global_rmse-r:+.4f}  "
          f"oof_R2(w*)={r2:+.3f}  wpred IQR [{np.percentile(wp,25):.2f},{np.percentile(wp,75):.2f}]", flush=True)

# best of the two
best_gain = max(global_rmse - run_router("ridge")[0], global_rmse - run_router("gbm")[0])
print("\n=== VERDICT ===", flush=True)
print(f"  best confidence-router out-of-fold gain = {best_gain:+.4f} ft  (bar: >= +0.10)", flush=True)
if best_gain >= 0.10:
    print("  PASS (offline): confidence features route the per-well weight -> productionize the router", flush=True)
    print("  in the kernel (predict w per well from the same feats), gate, submit. ⚠️ WATCH OOF<->LB:", flush=True)
    print("  this is a meta-model -> discount for GP-style transfer risk; a small LB probe, not a bank.", flush=True)
else:
    print("  NULL: confidence features don't predict which model wins per well any better than geometry.", flush=True)
    print("  The per-well oracle ceiling is real but UNREACHABLE from inference-available signals.", flush=True)
    print("  ROUTING AXIS FULLY CLOSED. Bank LB 8.158 + harden (final-selection discipline).", flush=True)
print("CONFROUTER DONE", flush=True)
