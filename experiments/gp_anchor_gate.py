"""GP/kriging anchor GATE: can GP impute the formation surface better than plane-KNN?

The binding constraint on our 11.9 score is the spatial anchor: plane-KNN imputes
ANCC(X,Y) with ~47 ft downstream TVT RMSE, with a heavy tail (isolated wells where
the local plane EXTRAPOLATES wildly -> +-700 ft outliers). Hypothesis: a Gaussian
Process regularizes extrapolation and kills that tail.

Decisive, cheap gate (no GBM): recompute the dominant feature tvt_formula
  TVT_pred = -Z + ANCC_imputed(X,Y) + b_well        (b_well from the known prefix)
with the ANCC imputer swapped plane-KNN -> GP, LOO over the 773 train wells, and
compare RMSE vs true TVT on the SAME hidden rows. The plane-KNN baseline is the
cached fk_tvt_formula (RMSE 47.25). Also report the error tail (median/p90/p99/max)
to see whether GP kills the outliers.

GATE: GP tvt_formula RMSE materially below 47.25 (esp. p99/max collapse) => the
anchor is improvable, push GP imputation into the feature build + retrain. If GP
is ~47 or worse, the anchor is not the lever (local dip beats global smoothness),
and we stop.

Method: well-centroid reference (x_med, y_med, ANCC_med per well) -- same reference
set plane-KNN uses. Fit GP hyperparams ONCE on all 773 centroids (anisotropic
Matern; geology dips directionally), then LOO: exclude each well, GP-posterior on
the other 772 (fixed hyperparams), predict ANCC along the held well's full (X,Y)
trajectory, calibrate b_well on its prefix, score tvt_formula on its hidden rows.
773 Cholesky solves of ~772x772 ~ seconds.
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
ART = ROOT / "data/processed/konbu"
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def load_well(wid):
    """Return dict with trajectory + prefix/hidden split, or None."""
    p = RAW / f"{wid}__horizontal_well.csv"
    h = pd.read_csv(p)
    if "TVT_input" not in h.columns or "TVT" not in h.columns:
        return None
    mask = h["TVT_input"].isna().to_numpy()
    if not mask.any():
        return None
    ms = int(np.flatnonzero(mask)[0])
    if ms == 0:
        return None
    return dict(
        wid=wid,
        x=h["X"].to_numpy(np.float64), y=h["Y"].to_numpy(np.float64),
        z=h["Z"].to_numpy(np.float64), tvt=h["TVT"].to_numpy(np.float64),
        tvt_input=h["TVT_input"].to_numpy(np.float64), ms=ms,
        x_med=float(h["X"].median()), y_med=float(h["Y"].median()),
        ancc_med=float(h["ANCC"].median()) if h["ANCC"].notna().any() else np.nan,
    )


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def stats(ae, tag):
    return (f"{tag}: RMSE={rmse(ae):.2f}  median={np.median(np.abs(ae)):.2f}  "
            f"p90={np.percentile(np.abs(ae),90):.2f}  p99={np.percentile(np.abs(ae),99):.2f}  "
            f"max={np.abs(ae).max():.1f}")


def main():
    wids = sorted({Path(p).name.replace("__horizontal_well", "").replace(".csv", "")
                   for p in glob.glob(str(RAW / "*__horizontal_well.csv"))})
    print(f">> {len(wids)} candidate wells; loading trajectories...", flush=True)
    wells = []
    for i, w in enumerate(wids):
        d = load_well(w)
        if d is not None and np.isfinite(d["ancc_med"]):
            wells.append(d)
        if (i + 1) % 150 == 0:
            print(f"   loaded {i+1}/{len(wids)}", flush=True)
    print(f">> {len(wells)} usable wells", flush=True)

    XY = np.array([[d["x_med"], d["y_med"]] for d in wells])
    ANCC = np.array([d["ancc_med"] for d in wells])

    # normalize coords (anisotropy handled by per-dim length scale on normalized space)
    mu, sd = XY.mean(0), XY.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    XYn = (XY - mu) / sd

    # fit GP hyperparameters ONCE on all centroids (anisotropic Matern + noise)
    print(">> fitting GP hyperparameters on 773 centroids (one-time)...", flush=True)
    k = (ConstantKernel(1.0, (1e-2, 1e4))
         * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=1.5)
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e4)))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(XYn, ANCC)
    print(f"   learned kernel: {gp.kernel_}", flush=True)

    # extract fitted kernel for fast manual LOO posteriors (fixed hyperparams)
    kf = gp.kernel_
    ymean = ANCC.mean()

    print(">> LOO: GP tvt_formula on hidden rows, per well...", flush=True)
    gp_err, pk_err_check = [], []
    rows_total = 0
    for j, d in enumerate(wells):
        # reference = all other wells
        ref = np.delete(np.arange(len(wells)), j)
        Xr, yr = XYn[ref], ANCC[ref]
        # GP posterior mean with fixed hyperparams: m(x*) = ymean + k*^T (K+sI)^-1 (y-ymean)
        K = kf(Xr)                                   # includes WhiteKernel noise on diag
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(Xr)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yr - ymean))
        # query at the held well's full trajectory
        qxy = np.column_stack([(d["x"] - mu[0]) / sd[0], (d["y"] - mu[1]) / sd[1]])
        Ks = kf(qxy, Xr)                             # (nrows, nref)
        ancc_pred = ymean + Ks @ alpha              # (nrows,)
        # b_well from prefix (known) rows: TVT_input + Z - ANCC_pred
        ms = d["ms"]
        b = np.median(d["tvt_input"][:ms] + d["z"][:ms] - ancc_pred[:ms])
        tvt_pred = -d["z"] + ancc_pred + b
        e = tvt_pred[ms:] - d["tvt"][ms:]           # hidden rows only
        gp_err.append(e)
        rows_total += len(e)
        if (j + 1) % 150 == 0:
            print(f"   {j+1}/{len(wells)} wells  ({rows_total:,} hidden rows)", flush=True)

    gp_err = np.concatenate(gp_err)
    print("\n=== GATE RESULT (hidden-row TVT error) ===", flush=True)
    print("  PLANE-KNN baseline (cached): RMSE=47.25  (the binding constraint)", flush=True)
    print("  " + stats(gp_err, "GP centroid    "), flush=True)
    print(f"\n  rows scored: {len(gp_err):,}", flush=True)
    print("\n  GATE: GP RMSE << 47.25 (esp. p99/max collapse) => anchor improvable, "
          "push GP into the feature build. GP ~47 or worse => stop.", flush=True)


if __name__ == "__main__":
    main()
