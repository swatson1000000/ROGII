"""BET 5 — Stage A gate: does DIP-TREND UNIVERSAL KRIGING of ANCC beat plane-fit KNN
UNDER BLOCK-HOLDOUT (not single-well LOO)?

THE HYPOTHESIS (research_geosteering_6ft.md §2, §6.1). Everything we do is single-typewell
stretch-squeeze (TSP). TVT is dip-distorted. Our ANCC imputer (plane-fit KNN / dense KNN) bakes
in a single LOCAL plane per neighborhood -> a constant local dip; it cannot extrapolate the REGIONAL
structural dip into a region with no nearby training well. GP failed there because a GP posterior
MEAN-REVERTS to the GLOBAL MEAN far from data (24.50 LOO -> 47.95 block -> LB 12.631 regression,
[[gate-spatial-levers-with-block-holdout]]). UNIVERSAL KRIGING with a polynomial dip TREND instead
mean-reverts to the TREND SURFACE -> it extrapolates the regional dip into the OOD region. That is a
precise, falsifiable fix for the exact pathology that killed GP.

WHY THE HARNESS IS RIGHT. score_well re-levels each well by b_well = median(prefix tvt_input+Z-ANCC),
so a CONSTANT ANCC offset is FREE; only the ALONG-WELL ANCC GRADIENT (= the dip) survives into the
scored residual. So this gate measures exactly the dip-placement quality the hypothesis is about.
Absolute RMSE levels here are imputer-reimpl artifacts (plane arm ~ konbu's family but a clean
reimpl) -> ONLY the DELTAS between estimators and the LOO->block DEGRADATION pattern are decisive,
exactly as in gp_block_holdout.py.

==================  PRE-REGISTERED GATE (set BEFORE seeing results; do not move)  ==================
Decision is on BLOCK-HOLDOUT only (LOO is shown only to expose the GP-style degradation tell).
Let  P = plane-KNN block-holdout RMSE   U = best UK variant block-holdout RMSE.
Let  deg(e) = e_block - e_loo  (density-coupling tell; GP's was +23).

  CLEAN PASS  (-> proceed to Stage B: rebuild ANCC features w/ UK, retrain GBM, OOF-gate vs 9.30):
      U <= 0.95 * P            (UK beats plane-KNN by >=5% under block-holdout)
      AND deg(UK) <= deg(plane-KNN) + 0.5*|deg(plane-KNN)|   (UK is NOT more density-coupled)
  KILL  (record as another spatial-family null; do NOT touch the GBM):
      U >= P                   (no transfer -- GP redux: a fancier spatial estimator that doesn't hold)
  AMBIGUOUS  (0.95*P < U < P, or degradation worse):
      Treat as NO. Per the sit-with question, "ambiguous in the density-coupled family" = "no"
      after the GP lesson. Record null; do not spend the GBM rebuild on a sub-noise imputer gain.

Stage A is NECESSARY, NOT SUFFICIENT: a clean pass only earns the expensive Stage B (the imputer
gain dilutes through b_well-releveling and the GBM, where rk_tvt_formula is already the #1 feature).
A regression/wash here kills BET 5 outright.
===================================================================================================

Run: nohup python -u experiments/bet5_kriging_block_holdout.py > log/bet5_kriging_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
K_BLOCKS = 6
K_NN = 10            # plane-KNN neighbors (konbu uses 10) -- identical to gp_block_holdout
ELL_GRID = (0.2, 0.4, 0.7, 1.0, 1.5)   # kriging length scale (standardized XY units)


# ----------------------------- data (identical to gp_block_holdout) -----------------------------
def load_well(p):
    h = pd.read_csv(p)
    if "TVT_input" not in h.columns or "TVT" not in h.columns:
        return None
    if "ANCC" not in h.columns or not h["ANCC"].notna().any():
        return None
    mask = h["TVT_input"].isna().to_numpy()
    if not mask.any():
        return None
    ms = int(np.flatnonzero(mask)[0])
    if ms == 0:
        return None
    return dict(
        wid=Path(p).name.replace("__horizontal_well.csv", ""),
        x=h["X"].to_numpy(np.float64), y=h["Y"].to_numpy(np.float64),
        z=h["Z"].to_numpy(np.float64), tvt=h["TVT"].to_numpy(np.float64),
        tvt_input=h["TVT_input"].to_numpy(np.float64), ms=ms,
        x_med=float(h["X"].median()), y_med=float(h["Y"].median()),
        ancc_med=float(h["ANCC"].median()),
    )


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def report(tag, err):
    ae = np.abs(err)
    print(f"  {tag:26s} RMSE={rmse(err):7.2f}  median={np.median(ae):6.2f}  "
          f"p90={np.percentile(ae,90):6.2f}  p99={np.percentile(ae,99):7.2f}  "
          f"max={ae.max():8.1f}  n={len(err):,}", flush=True)


def score_well(d, ancc_pred):
    """b_well-calibrated tvt_formula residual on eval rows. b absorbs the constant ANCC
    offset, so this isolates the along-well ANCC gradient (the dip)."""
    ms = d["ms"]
    b = np.median(d["tvt_input"][:ms] + d["z"][:ms] - ancc_pred[:ms])
    tvt_pred = -d["z"] + ancc_pred + b
    return tvt_pred[ms:] - d["tvt"][ms:]


# ------------------------------------- plane-KNN baseline ---------------------------------------
def plane_impute_well(d, XYref_raw, Aref):
    c = np.array([d["x_med"], d["y_med"]])
    dist = np.sqrt(((XYref_raw - c) ** 2).sum(1))
    idx = np.argpartition(dist, min(K_NN, len(dist) - 1))[:K_NN]
    Xn, yn, dn = XYref_raw[idx], Aref[idx], dist[idx]
    w = 1.0 / (dn + 1e-6)
    A = np.column_stack([np.ones(len(Xn)), Xn[:, 0], Xn[:, 1]])
    W = np.diag(w)
    try:
        coef = np.linalg.lstsq(W @ A, W @ yn, rcond=None)[0]
    except np.linalg.LinAlgError:
        coef = np.array([np.average(yn, weights=w), 0.0, 0.0])
    return coef[0] + coef[1] * d["x"] + coef[2] * d["y"]


# --------------------------------------- GP reference -------------------------------------------
def fit_gp_kernel(XYn, A):
    k = (ConstantKernel(1.0, (1e-2, 1e4))
         * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=1.5)
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e4)))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(XYn, A)
    return gp.kernel_


def gp_impute_well(d, XYr, Ar, kf, ymean, mu, sd):
    K = kf(XYr)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(XYr)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Ar - ymean))
    qxy = np.column_stack([(d["x"] - mu[0]) / sd[0], (d["y"] - mu[1]) / sd[1]])
    Ks = kf(qxy, XYr)
    return ymean + Ks @ alpha


# ----------------------- universal kriging (poly dip-trend + kriged residual) -------------------
def trend_basis(XY, degree):
    x, y = XY[:, 0], XY[:, 1]
    cols = [np.ones(len(x))]
    if degree >= 1:
        cols += [x, y]
    if degree >= 2:
        cols += [x * x, x * y, y * y]
    return np.column_stack(cols)


def _pdist2(A, B):
    """squared euclidean between rows of A (m x 2) and B (n x 2) -> m x n"""
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(2)


def fit_uk_cov(XYn, A, degree):
    """Fit residual-process covariance (sill, nugget, ell) once on a reference set.
    Detrend with OLS poly(degree); pick ell by 5-fold reference RMSE of ordinary kriging
    of the residual. Returns (sill, nugget, ell)."""
    F = trend_basis(XYn, degree)
    beta, *_ = np.linalg.lstsq(F, A, rcond=None)
    r = A - F @ beta
    sill = float(np.var(r))
    nugget = max(1e-6, 0.10 * sill)
    n = len(XYn)
    rng = np.random.RandomState(0)
    fold = rng.randint(0, 5, n)
    D = np.sqrt(_pdist2(XYn, XYn))
    best_ell, best_err = ELL_GRID[0], np.inf
    for ell in ELL_GRID:
        C = sill * np.exp(-D / ell)
        C[np.diag_indices_from(C)] += nugget
        err = []
        for f in range(5):
            tr = fold != f
            va = ~tr
            Ctr = C[np.ix_(tr, tr)]
            try:
                w = np.linalg.solve(Ctr, r[tr])
            except np.linalg.LinAlgError:
                continue
            Cva = sill * np.exp(-D[np.ix_(va, tr)] / ell)
            pred = Cva @ w
            err.append(r[va] - pred)
        if err:
            e = rmse(np.concatenate(err))
            if e < best_err:
                best_err, best_ell = e, ell
    return sill, nugget, best_ell


def uk_impute_well(d, XYr, Ar, degree, cov, mu, sd):
    """Universal kriging (dual/Lagrange form) of ANCC along well d, ref=(XYr std, Ar)."""
    sill, nugget, ell = cov
    n = len(XYr)
    F = trend_basis(XYr, degree)
    p = F.shape[1]
    Drr = np.sqrt(_pdist2(XYr, XYr))
    C = sill * np.exp(-Drr / ell)
    C[np.diag_indices_from(C)] += nugget
    M = np.zeros((n + p, n + p))
    M[:n, :n] = C
    M[:n, n:] = F
    M[n:, :n] = F.T
    Q = np.column_stack([(d["x"] - mu[0]) / sd[0], (d["y"] - mu[1]) / sd[1]])
    Dq = np.sqrt(_pdist2(XYr, Q))            # n x q  (ref-query; no nugget on cross)
    cq = sill * np.exp(-Dq / ell)
    Fq = trend_basis(Q, degree)             # q x p
    RHS = np.vstack([cq, Fq.T])             # (n+p) x q
    try:
        sol = np.linalg.solve(M, RHS)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(M, RHS, rcond=None)[0]
    lam = sol[:n, :]                        # n x q
    return lam.T @ Ar


# --------------------------------------------- main ---------------------------------------------
def main():
    paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    print(f">> loading {len(paths)} wells...", flush=True)
    wells = [load_well(p) for p in paths]
    wells = [w for w in wells if w is not None]
    print(f">> {len(wells)} usable wells", flush=True)

    XY = np.array([[w["x_med"], w["y_med"]] for w in wells])
    A = np.array([w["ancc_med"] for w in wells])
    mu, sd = XY.mean(0), XY.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    XYn = (XY - mu) / sd

    DEGREES = [1, 2]

    # ---------------- (a) single-well LOO ----------------
    print("\n>> (a) SINGLE-WELL LOO ...", flush=True)
    kf_all = fit_gp_kernel(XYn, A)
    ymean_all = A.mean()
    cov_all = {dg: fit_uk_cov(XYn, A, dg) for dg in DEGREES}
    for dg in DEGREES:
        s, ng, el = cov_all[dg]
        print(f"   UK deg{dg} cov: sill={s:.3g} nugget={ng:.3g} ell={el}", flush=True)
    acc = {"plane": [], "GP": [], **{f"UK{dg}": [] for dg in DEGREES}}
    for j, d in enumerate(wells):
        ref = np.delete(np.arange(len(wells)), j)
        acc["plane"].append(score_well(d, plane_impute_well(d, XY[ref], A[ref])))
        acc["GP"].append(score_well(d, gp_impute_well(d, XYn[ref], A[ref], kf_all, ymean_all, mu, sd)))
        for dg in DEGREES:
            acc[f"UK{dg}"].append(score_well(d, uk_impute_well(d, XYn[ref], A[ref], dg, cov_all[dg], mu, sd)))
        if (j + 1) % 200 == 0:
            print(f"   loo {j+1}/{len(wells)}", flush=True)
    loo = {k: np.concatenate(v) for k, v in acc.items()}
    print("  --- LOO (held well keeps near neighbors) ---", flush=True)
    for k in acc:
        report(f"{k} LOO", loo[k])

    # ---------------- (b) spatial block-holdout ----------------
    print(f"\n>> (b) SPATIAL BLOCK-HOLDOUT (K={K_BLOCKS} KMeans blocks) ...", flush=True)
    labels = KMeans(n_clusters=K_BLOCKS, n_init=10, random_state=0).fit_predict(XYn)
    for b in range(K_BLOCKS):
        print(f"   block {b}: {(labels==b).sum()} wells", flush=True)
    bacc = {"plane": [], "GP": [], **{f"UK{dg}": [] for dg in DEGREES}}
    for b in range(K_BLOCKS):
        held = np.flatnonzero(labels == b)
        ref = np.flatnonzero(labels != b)
        kf_b = fit_gp_kernel(XYn[ref], A[ref])           # refit GP hyperparams on reduced ref (honest OOD)
        ymean_b = A[ref].mean()
        cov_b = {dg: fit_uk_cov(XYn[ref], A[ref], dg) for dg in DEGREES}   # refit UK cov on reduced ref
        pe, ge, ue = [], [], {dg: [] for dg in DEGREES}
        for j in held:
            d = wells[j]
            pe.append(score_well(d, plane_impute_well(d, XY[ref], A[ref])))
            ge.append(score_well(d, gp_impute_well(d, XYn[ref], A[ref], kf_b, ymean_b, mu, sd)))
            for dg in DEGREES:
                ue[dg].append(score_well(d, uk_impute_well(d, XYn[ref], A[ref], dg, cov_b[dg], mu, sd)))
        bacc["plane"].append(np.concatenate(pe))
        bacc["GP"].append(np.concatenate(ge))
        for dg in DEGREES:
            bacc[f"UK{dg}"].append(np.concatenate(ue[dg]))
        msg = f"   block {b}: plane={rmse(np.concatenate(pe)):.2f}  GP={rmse(np.concatenate(ge)):.2f}"
        for dg in DEGREES:
            msg += f"  UK{dg}={rmse(np.concatenate(ue[dg])):.2f}"
        print(msg, flush=True)
    blk = {k: np.concatenate(v) for k, v in bacc.items()}
    print("  --- BLOCK-HOLDOUT (held block has NO in-block neighbor) ---", flush=True)
    for k in bacc:
        report(f"{k} BLOCK", blk[k])

    # ---------------- pre-registered gate ----------------
    print("\n=== DEGRADATION (LOO -> BLOCK; GP-style density-coupling tell) ===", flush=True)
    deg = {}
    for k in loo:
        deg[k] = rmse(blk[k]) - rmse(loo[k])
        print(f"  {k:6s}: LOO {rmse(loo[k]):7.2f} -> BLOCK {rmse(blk[k]):7.2f}  (deg {deg[k]:+.2f})", flush=True)

    P = rmse(blk["plane"])
    uk_keys = [f"UK{dg}" for dg in DEGREES]
    best_uk = min(uk_keys, key=lambda k: rmse(blk[k]))
    U = rmse(blk[best_uk])
    degP = deg["plane"]
    print("\n=== PRE-REGISTERED GATE VERDICT ===", flush=True)
    print(f"  plane-KNN block = {P:.2f}   best UK = {best_uk} block = {U:.2f}   "
          f"(U/P = {U/P:.3f})", flush=True)
    print(f"  degradation: plane {degP:+.2f}   {best_uk} {deg[best_uk]:+.2f}", flush=True)
    cond_win = U <= 0.95 * P
    cond_deg = deg[best_uk] <= degP + 0.5 * abs(degP)
    if U >= P:
        print(f"  >> KILL: UK does NOT beat plane-KNN under block-holdout (U/P={U/P:.3f} >= 1).", flush=True)
        print("     GP redux -- a fancier spatial estimator that doesn't transfer. BET 5 dies here.", flush=True)
    elif cond_win and cond_deg:
        print(f"  >> CLEAN PASS: UK beats plane-KNN by {(1-U/P)*100:.1f}% under block-holdout "
              f"AND degradation not worse.", flush=True)
        print("     -> proceed to Stage B (rebuild ANCC features w/ UK, retrain GBM, OOF-gate vs 9.30).", flush=True)
    else:
        why = []
        if not cond_win:
            why.append(f"margin only {(1-U/P)*100:.1f}% (<5%)")
        if not cond_deg:
            why.append(f"degradation {deg[best_uk]:+.2f} worse than plane {degP:+.2f}")
        print(f"  >> AMBIGUOUS -> treat as NO ({'; '.join(why)}).", flush=True)
        print("     Per the GP lesson, ambiguous in the density-coupled family = no. Record null; "
              "do NOT spend the GBM rebuild.", flush=True)
    print("BET5 KRIGING BLOCK DONE", flush=True)


if __name__ == "__main__":
    main()
