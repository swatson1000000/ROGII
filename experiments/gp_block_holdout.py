"""Does GP's anchor advantage SURVIVE out-of-distribution wells?

The gate ('GP 24.50 vs plane-KNN 47.25') was measured under SINGLE-WELL LOO: a held
well still keeps all its near neighbors in the 766-centroid reference. The hidden
~200-well test set may sit in (X,Y) regions with NO nearby training well. A GP
posterior MEAN-REVERTS to the global mean far from data; plane-KNN extrapolates
LOCALLY. Hypothesis C: under a spatially-shifted hidden set GP loses its advantage
(and its tail blows up) -> this, not a zero-fill guard, is the LB 12.631 regression.

Decisive test: SPATIAL BLOCK-HOLDOUT. KMeans the well centroids into K blocks; hold
out an entire block from the reference, impute ANCC for the held block's wells, score
tvt_formula = -Z + ANCC + b_well on their hidden rows. Compare GP vs plane-KNN under
(a) single-well LOO and (b) block-holdout. If GP wins LOO but loses block-holdout,
hypothesis C is confirmed and GP is correctly dead (doesn't transfer), not mis-dead.

Run: nohup python -u experiments/gp_block_holdout.py > log/gp_block_$(date +%Y%m%d_%H%M%S).log 2>&1 &
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
K_NN = 10  # plane-KNN neighbors (konbu uses 10)


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
    print(f"  {tag:24s} RMSE={rmse(err):7.2f}  median={np.median(ae):6.2f}  "
          f"p90={np.percentile(ae,90):6.2f}  p99={np.percentile(ae,99):7.2f}  "
          f"max={ae.max():8.1f}  n={len(err):,}", flush=True)


def fit_gp_kernel(XYn, A):
    k = (ConstantKernel(1.0, (1e-2, 1e4))
         * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=1.5)
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e4)))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(XYn, A)
    return gp.kernel_


def gp_impute_well(d, XYr, Ar, kf, ymean, mu, sd):
    """GP posterior mean ANCC along well d's trajectory, ref = (XYr, Ar)."""
    K = kf(XYr)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(XYr)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Ar - ymean))
    qxy = np.column_stack([(d["x"] - mu[0]) / sd[0], (d["y"] - mu[1]) / sd[1]])
    Ks = kf(qxy, XYr)
    return ymean + Ks @ alpha


def plane_impute_well(d, XYref_raw, Aref):
    """Local weighted-plane ANCC: K nearest centroids to the well centroid, fit
    ANCC ~ a + b*X + c*Y (distance-weighted), evaluate at each row's (X,Y)."""
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


def score_well(d, ancc_pred):
    ms = d["ms"]
    b = np.median(d["tvt_input"][:ms] + d["z"][:ms] - ancc_pred[:ms])
    tvt_pred = -d["z"] + ancc_pred + b
    return tvt_pred[ms:] - d["tvt"][ms:]


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

    # ---- (a) single-well LOO (reproduces the gate condition) ----
    print("\n>> (a) SINGLE-WELL LOO ...", flush=True)
    kf_all = fit_gp_kernel(XYn, A)
    print(f"   GP kernel (full ref): {kf_all}", flush=True)
    ymean_all = A.mean()
    gp_loo, pk_loo = [], []
    for j, d in enumerate(wells):
        ref = np.delete(np.arange(len(wells)), j)
        gp_loo.append(score_well(d, gp_impute_well(d, XYn[ref], A[ref], kf_all, ymean_all, mu, sd)))
        pk_loo.append(score_well(d, plane_impute_well(d, XY[ref], A[ref])))
        if (j + 1) % 200 == 0:
            print(f"   loo {j+1}/{len(wells)}", flush=True)
    gp_loo = np.concatenate(gp_loo); pk_loo = np.concatenate(pk_loo)
    print("  --- LOO (held well keeps near neighbors) ---", flush=True)
    report("GP  LOO", gp_loo)
    report("plane-KNN LOO", pk_loo)

    # ---- (b) spatial block-holdout (simulates a shifted hidden region) ----
    print(f"\n>> (b) SPATIAL BLOCK-HOLDOUT (K={K_BLOCKS} KMeans blocks) ...", flush=True)
    labels = KMeans(n_clusters=K_BLOCKS, n_init=10, random_state=0).fit_predict(XYn)
    for b in range(K_BLOCKS):
        print(f"   block {b}: {(labels==b).sum()} wells", flush=True)
    gp_blk, pk_blk = [], []
    block_rmse = []
    for b in range(K_BLOCKS):
        held = np.flatnonzero(labels == b)
        ref = np.flatnonzero(labels != b)
        # refit GP hyperparams on the REDUCED reference (honest OOD)
        kf_b = fit_gp_kernel(XYn[ref], A[ref])
        ymean_b = A[ref].mean()
        gp_b, pk_b = [], []
        for j in held:
            d = wells[j]
            gp_b.append(score_well(d, gp_impute_well(d, XYn[ref], A[ref], kf_b, ymean_b, mu, sd)))
            pk_b.append(score_well(d, plane_impute_well(d, XY[ref], A[ref])))
        gp_b = np.concatenate(gp_b); pk_b = np.concatenate(pk_b)
        block_rmse.append((b, len(held), rmse(gp_b), rmse(pk_b)))
        gp_blk.append(gp_b); pk_blk.append(pk_b)
        print(f"   block {b}: GP RMSE={rmse(gp_b):.2f}  plane-KNN RMSE={rmse(pk_b):.2f}", flush=True)
    gp_blk = np.concatenate(gp_blk); pk_blk = np.concatenate(pk_blk)
    print("  --- BLOCK-HOLDOUT (held block has NO in-block neighbor) ---", flush=True)
    report("GP  BLOCK", gp_blk)
    report("plane-KNN BLOCK", pk_blk)

    print("\n=== VERDICT ===", flush=True)
    print(f"  GP:        LOO {rmse(gp_loo):.2f} -> BLOCK {rmse(gp_blk):.2f}  "
          f"(degradation {rmse(gp_blk)-rmse(gp_loo):+.2f})", flush=True)
    print(f"  plane-KNN: LOO {rmse(pk_loo):.2f} -> BLOCK {rmse(pk_blk):.2f}  "
          f"(degradation {rmse(pk_blk)-rmse(pk_loo):+.2f})", flush=True)
    print("  If GP degrades MUCH more than plane-KNN (esp. tail) -> hypothesis C confirmed:", flush=True)
    print("  GP mean-reverts OOD, the +0.233 OOF does NOT transfer, regression is REAL not a guard bug.", flush=True)
    print("BLOCK HOLDOUT DONE", flush=True)


if __name__ == "__main__":
    main()
