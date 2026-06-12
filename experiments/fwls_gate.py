"""CAND B gate: per-well SOFT blend weights via heavily-regularized FWLS.

w_well = clip(0.57 + g(meta), 0.45, 0.70), g = ridge over per-well meta-features,
fit OUT-OF-FOLD with NO intercept on train-fold-centered metas (the centering constraint:
the mean weight stays at the LB-mapped 0.57 vertex by construction; an intercept would
recover the OOF-opt 0.44 = LB-known-worse 8.269).

Per-well objective is exactly quadratic in w:
    J_w(w) = sum_rows ((1-w) g + w p)^2 = a_w w^2 + 2 b_w w + c_w,
    a_w = sum (p-g)^2,  b_w = sum g (p-g),  w*_w = -b_w / a_w
so the FWLS fit is a_w-weighted ridge of delta*_w = (w*_w - 0.57) on metas; row-level
blended RMSE is recovered exactly from the quadratics (no approximation).

Nested protocol: outer = production RandomState(42) 5-fold well split; inner = 4-fold over
the outer-train wells to choose lambda from {30, 100, 300, 1000, 3000, 10000} by row-level
RMSE; refit on all outer-train wells; apply to outer-val wells.

PRE-REGISTERED BAR (set before running): nested OOF improvement >= 0.02 ft vs flat-0.57
baseline AND clipped weights within [0.45, 0.70] (enforced) with a sane unclipped spread.
NULL closes the whole per-well-weighting axis (hard selector + confidence router already
gated null, commit 5d4d34e).

Diagnostics (NOT the gate): in-band per-well ORACLE w* (headroom upper bound); the
WITH-intercept fit (expected to drift toward OOF-opt ~0.44 — the documented trap).

Run: nohup python -u experiments/fwls_gate.py > log/fwls_gate_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import Ridge

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
KEYS = ["lgb_42", "lgb_7", "lgb_123", "cat_42", "cat_7", "cat_123"]
W0, W_LO, W_HI = 0.57, 0.45, 0.70
# lambda is applied with a_w NORMALIZED to mean 1, so the data term is O(n_wells) and
# lambda ~ 1000 is genuinely heavy (the LSHTC4 ridge~1000 regime); 1e5 ~= frozen at 0.57
LAMBDAS = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
rmse = lambda e: float(np.sqrt(np.mean(e * e)))

print(">> load frontier_ens train matrix + meta columns...", flush=True)
df = pd.read_parquet(ROOT / "data/processed/frontier_ens/train_feats.parquet",
                     columns=["well", "id", "last_known_tvt", "target", "md_since",
                              "z", "pfx_rmse", "sig_std", "spatial_knn_dist", "dense_dist"])
y = df["target"].to_numpy(np.float64)
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)

X = np.column_stack([np.load(ROOT / "models/frontier_ens_nouk" / f"oof_{k}.npy").astype(np.float64)
                     for k in KEYS])
assert len(X) == len(y)
r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(X, y)
g_res_all = X @ r.coef_ - y
print(f"   nouk stack OOF = {rmse(g_res_all):.4f}  [self-check 10.3232]", flush=True)
assert abs(rmse(g_res_all) - 10.3232) < 0.01

print(">> align PF residuals + per-well quadratics + metas...", flush=True)
paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells = []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue
    pf_wells.append(wid)
grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}


def good(r):
    return r is not None and not (isinstance(r[0], str) and r[0] == "ERR")


res = joblib.load(ROOT / "models/frontier/pf_real_results.pkl")
rg = [r for r in res if good(r)]
assert len(rg) == len(pf_wells)

wells, A, B, NROW = [], [], [], []
META = []  # per-well meta vector
g_l, p_l, wrow_l = [], [], []
SCALES = ["pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12"]
for i, wid in enumerate(pf_wells):
    truth, pf = rg[i]
    sub = grp.get(wid)
    if sub is None or len(sub) != len(truth):
        continue
    if not np.allclose(sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64),
                       truth, atol=1e-3):
        continue
    g = g_res_all[sub.index.to_numpy()]
    p = pf["pf_scale_12"] - truth
    d = p - g
    a, b = float(np.dot(d, d)), float(np.dot(g, d))
    scale_mat = np.vstack([pf[s] for s in SCALES])
    zc = sub["z"].to_numpy(np.float64)
    meta = [
        np.log(len(sub)),                                            # n_eval
        np.log(max(sub["md_since"].max(), 1.0)),                     # lateral eval length
        float(zc.max() - zc.min()),                                  # z span
        float(np.nanmean(sub["pfx_rmse"].to_numpy(np.float64))),     # prefix tw-fit RMSE
        float(np.nanmean(sub["sig_std"].to_numpy(np.float64))),      # inter-signal disagreement
        float(np.log1p(np.nanmean(sub["spatial_knn_dist"].to_numpy(np.float64)))),
        float(np.log1p(np.nanmean(sub["dense_dist"].to_numpy(np.float64)))),
        float(np.mean(np.std(scale_mat, axis=0))),                   # PF multi-scale spread
        float(np.mean(np.abs(pf["pf_scale_12"] - pf["pf_mean"]))),   # weighted-vs-mean PF gap
    ]
    wells.append(wid); A.append(a); B.append(b); NROW.append(len(sub)); META.append(meta)
    g_l.append(g); p_l.append(p)

wells = np.array(wells); A = np.array(A); B = np.array(B); NROW = np.array(NROW)
META = np.array(META, np.float64)
META = np.where(np.isfinite(META), META, np.nan)
col_med = np.nanmedian(META, axis=0)
META = np.where(np.isnan(META), col_med, META)
n_total = int(NROW.sum())
print(f"   wells: {len(wells)}, rows: {n_total}", flush=True)

w_star = -B / A
base_res = np.concatenate([(1 - W0) * g + W0 * p for g, p in zip(g_l, p_l)])
base = rmse(base_res)
print(f"\n   BASELINE flat w=0.57 OOF = {base:.4f}", flush=True)


def score_weights(w_by_well):
    """exact row-level RMSE from per-well quadratics"""
    sse = float(np.sum(A * w_by_well**2 + 2 * B * w_by_well
                       + np.array([np.dot(g, g) for g in g_l])))
    return np.sqrt(sse / n_total)


CC = np.array([np.dot(g, g) for g in g_l])


def sse_of(w):
    return A * w**2 + 2 * B * w + CC


print(f"   [cross-check] quadratic recon of flat 0.57: {np.sqrt(sse_of(np.full(len(wells), W0)).sum()/n_total):.4f}",
      flush=True)

# oracle headroom (in-sample, in-band)
w_or = np.clip(w_star, W_LO, W_HI)
o = np.sqrt(sse_of(w_or).sum() / n_total)
print(f"   ORACLE in-band per-well w*: {o:.4f} ({o - base:+.4f})  "
      f"[absolute headroom of this axis; in-sample]", flush=True)
print(f"   w* distribution: p10/p50/p90 = {np.percentile(w_star, [10, 50, 90]).round(3)}, "
      f"in-band fraction = {((w_star >= W_LO) & (w_star <= W_HI)).mean():.2f}", flush=True)

# production outer folds
rng = np.random.RandomState(42)
uw = df["well"].unique().copy()
rng.shuffle(uw)
fo = {w: i % 5 for i, w in enumerate(uw)}
fold_w = np.array([fo[w] for w in wells])


def fit_predict(tr_idx, va_idx, lam, intercept=False):
    M = META[tr_idx]
    mu, sd = M.mean(0), M.std(0)
    sd[sd == 0] = 1.0
    Mtr = (M - mu) / sd
    Mva = (META[va_idx] - mu) / sd
    if intercept:
        Mtr = np.hstack([Mtr, np.ones((len(Mtr), 1))])
        Mva = np.hstack([Mva, np.ones((len(Mva), 1))])
    tgt = w_star[tr_idx] - W0
    aw = A[tr_idx] / A[tr_idx].mean()  # normalize so lambda has a stable, interpretable scale
    # weighted ridge: theta = (M' diag(a) M + lam I)^-1 M' diag(a) tgt
    MtA = Mtr.T * aw
    theta = np.linalg.solve(MtA @ Mtr + lam * np.eye(Mtr.shape[1]), MtA @ tgt)
    return np.clip(W0 + Mva @ theta, W_LO, W_HI), theta


print("\n=== nested out-of-fold FWLS (centered, no intercept) ===", flush=True)
w_oof = np.full(len(wells), W0)
lam_chosen = []
for f in range(5):
    tr, va = np.where(fold_w != f)[0], np.where(fold_w == f)[0]
    # inner 4-fold over outer-train wells
    inner = fold_w[tr].copy()
    inner_ids = sorted(set(inner))
    best_lam, best_sse = None, np.inf
    for lam in LAMBDAS:
        sse = 0.0
        for fi in inner_ids:
            itr, iva = tr[inner != fi], tr[inner == fi]
            wv, _ = fit_predict(itr, iva, lam)
            sse += float(np.sum(A[iva] * wv**2 + 2 * B[iva] * wv + CC[iva]))
        if sse < best_sse:
            best_sse, best_lam = sse, lam
    wv, theta = fit_predict(tr, va, best_lam)
    w_oof[va] = wv
    lam_chosen.append(best_lam)
oo = np.sqrt(sse_of(w_oof).sum() / n_total)
print(f"   per-fold lambda: {lam_chosen}", flush=True)
print(f"   OOF weights: p5/p50/p95 = {np.percentile(w_oof, [5, 50, 95]).round(3)}, "
      f"mean {w_oof.mean():.3f}, clipped frac = {((w_oof <= W_LO + 1e-9) | (w_oof >= W_HI - 1e-9)).mean():.2f}",
      flush=True)
print(f"   >>> NESTED OOF = {oo:.4f}  ({oo - base:+.4f} vs base {base:.4f})", flush=True)
print(f"   GATE (>= -0.02 AND weights in [0.45,0.70]): "
      f"{'PASS' if oo - base <= -0.02 else 'NULL'}", flush=True)

# DECISIVE decomposition (the alpha=1.05 lesson): how much of the gain is just an
# EFFECTIVE flat-weight shift toward the OOF-opt (the known LB-inverted direction),
# vs genuine per-well matching on top of it?
w_eff = float(np.sum(A * w_oof) / A.sum())  # curvature-weighted mean = the shift that moves OOF
flat_eff = np.sqrt(sse_of(np.full(len(wells), w_eff)).sum() / n_total)
print(f"\n   DECOMPOSITION: a-weighted effective mean w = {w_eff:.3f} (unweighted {w_oof.mean():.3f})",
      flush=True)
print(f"   flat blend at w={w_eff:.3f}: {flat_eff:.4f} ({flat_eff - base:+.4f}) "
      f"<- the mirage component (flat-shift toward OOF-opt 0.44 = LB-known-WORSE)", flush=True)
print(f"   residual per-well matching gain = {oo - flat_eff:+.4f} "
      f"<- the part that is genuinely per-well", flush=True)
rk = np.corrcoef(np.argsort(np.argsort(A)), w_oof)[0, 1]
print(f"   rank-corr(a_w, w_oof) = {rk:+.3f} (negative = shades PF down where PF/GBM disagree most)",
      flush=True)

# per-lambda OOF curve (no inner selection; diagnostic only)
print("\n=== diagnostic: per-lambda OOF curve (centered) ===", flush=True)
for lam in LAMBDAS:
    wv = np.full(len(wells), W0)
    for f in range(5):
        tr, va = np.where(fold_w != f)[0], np.where(fold_w == f)[0]
        wv[va], _ = fit_predict(tr, va, lam)
    rr = np.sqrt(sse_of(wv).sum() / n_total)
    print(f"   lam={lam:>7.0f}: OOF {rr:.4f} ({rr - base:+.4f})  "
          f"w p5/p95 = {np.percentile(wv, [5, 95]).round(3)}", flush=True)

# the documented trap, shown once: WITH intercept
print("\n=== diagnostic: WITH-intercept fit (expected to drift toward OOF-opt ~0.44) ===", flush=True)
for lam in (100.0, 1000.0):
    wv = np.full(len(wells), W0)
    for f in range(5):
        tr, va = np.where(fold_w != f)[0], np.where(fold_w == f)[0]
        wv[va], _ = fit_predict(tr, va, lam, intercept=True)
    rr = np.sqrt(sse_of(wv).sum() / n_total)
    print(f"   lam={lam:>6.0f}: OOF {rr:.4f} ({rr - base:+.4f})  mean w = {wv.mean():.3f}  "
          f"[OOF gain here is the KNOWN mirage — LB vertex is 0.57]", flush=True)

print("\nFWLS GATE DONE", flush=True)
