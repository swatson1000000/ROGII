"""Build GP/kriging ANCC features for the konbu matrices (train LOO + test full-ref).

The GP anchor gate PASSED at the imputation layer (LOO TVT RMSE 24.50 vs row-KNN
27.37 vs plane-KNN 47.25; tail max 693->173). This script productionizes the GP
imputer into joinable per-row features so we can (1) cheap solo-LGB ablation and
(2) the full 5-model stack combined gate vs the banked 11.821.

GP = centroid GP (766 well centroids), the SAME estimator that won the gate
(anisotropic Matern 1.5 + WhiteKernel, hyperparams fit once on all centroids).
Per row we emit:
  gp_ancc      imputed ANCC at the row's (X,Y)               [GP posterior mean]
  gp_std       GP posterior std at (X,Y)  (uncertainty)      [the new signal vs KNN]
  gp_tvt_abs   -Z + gp_ancc + b_well   (b_well from prefix)  [absolute TVT estimate]
Drift / disagreement features are derived downstream from the cached matrix
columns (last_known_tvt, fk_tvt_formula) to avoid a units mismatch here.

TRAIN: leave-one-well-out posterior (matches how fk_/rk_ are built -> no leak).
TEST : full train-centroid reference (matches the inference kernel).

Self-validating: re-reads raw GR per row and the script that consumes this asserts
the (well,row_idx) join reproduces the konbu matrix `gr` column.

Run:  nohup python -u experiments/gp_feature_build.py > log/gp_build_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.linalg import cholesky, solve_triangular

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW_TR = ROOT / "data/raw/train"
RAW_TE = ROOT / "data/raw/test"
ART = ROOT / "data/processed/konbu"
OUT_TR = ART / "gp_feats_train.parquet"
OUT_TE = ART / "gp_feats_test.parquet"

# globals shared with workers via fork
_G = {}


def load_traj(p):
    """Load a horizontal_well.csv -> dict of arrays, or None if unusable."""
    h = pd.read_csv(p)
    for c in ("X", "Y", "Z", "GR"):
        if c not in h.columns:
            return None
    wid = Path(p).name.replace("__horizontal_well.csv", "")
    # prefix/PS split keys off TVT_input (present on BOTH train and test wells);
    # do NOT gate on TVT, which is absent on test wells -> would zero out b_well.
    if "TVT_input" in h.columns and h["TVT_input"].notna().any():
        mask = h["TVT_input"].isna().to_numpy()
        ms = int(np.flatnonzero(mask)[0]) if mask.any() else len(h)
        tvt_input = h["TVT_input"].to_numpy(np.float64)
    else:
        ms = 0
        tvt_input = np.full(len(h), np.nan)
    return dict(
        wid=wid,
        x=h["X"].to_numpy(np.float64), y=h["Y"].to_numpy(np.float64),
        z=h["Z"].to_numpy(np.float64), gr=h["GR"].to_numpy(np.float64),
        tvt_input=tvt_input, ms=ms,
        x_med=float(h["X"].median()), y_med=float(h["Y"].median()),
        ancc_med=float(h["ANCC"].median()) if ("ANCC" in h.columns and h["ANCC"].notna().any()) else np.nan,
    )


def gp_rows(d, exclude_self):
    """GP posterior mean+std along well d's trajectory. exclude_self=LOO for train."""
    XYn, ANCC = _G["XYn"], _G["ANCC"]
    mu, sd = _G["mu"], _G["sd"]
    k1, kf = _G["k1"], _G["kf"]
    ymean, const = _G["ymean"], _G["const"]

    if exclude_self and d.get("ref_idx") is not None:
        ref = np.delete(np.arange(len(XYn)), d["ref_idx"])
    else:
        ref = np.arange(len(XYn))
    Xr, yr = XYn[ref], ANCC[ref]
    K = kf(Xr)
    L = cholesky(K + 1e-6 * np.eye(len(Xr)), lower=True)
    alpha = solve_triangular(L.T, solve_triangular(L, yr - ymean, lower=True), lower=False)

    qxy = np.column_stack([(d["x"] - mu[0]) / sd[0], (d["y"] - mu[1]) / sd[1]])
    Ks = k1(qxy, Xr)                       # signal-only cross cov (nrows x nref)
    mean = ymean + Ks @ alpha
    v = solve_triangular(L, Ks.T, lower=True)        # (nref x nrows)
    var = const - np.sum(v * v, axis=0)              # k1(x,x)=const
    std = np.sqrt(np.clip(var, 0.0, None))

    ms = d["ms"]
    if ms > 0:
        b = float(np.median(d["tvt_input"][:ms] + d["z"][:ms] - mean[:ms]))
    else:
        b = 0.0  # test wells with no known prefix -> downstream uses last_known_tvt
    tvt_abs = -d["z"] + mean + b
    return pd.DataFrame({
        "well": d["wid"],
        "row_idx": np.arange(len(d["x"]), dtype=np.int64),
        "gr": d["gr"].astype(np.float32),
        "gp_ancc": mean.astype(np.float32),
        "gp_std": std.astype(np.float32),
        "gp_tvt_abs": tvt_abs.astype(np.float32),
    })


def _worker_train(d):
    return gp_rows(d, exclude_self=True)


def _worker_test(d):
    return gp_rows(d, exclude_self=False)


def main():
    print(">> loading TRAIN trajectories...", flush=True)
    tr_paths = sorted(glob.glob(str(RAW_TR / "*__horizontal_well.csv")))
    train = [load_traj(p) for p in tr_paths]
    train = [d for d in train if d is not None and np.isfinite(d["ancc_med"]) and d["ms"] > 0]
    print(f">> {len(train)} usable train wells", flush=True)

    XY = np.array([[d["x_med"], d["y_med"]] for d in train])
    ANCC = np.array([d["ancc_med"] for d in train])
    mu, sd = XY.mean(0), XY.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    XYn = (XY - mu) / sd
    for i, d in enumerate(train):
        d["ref_idx"] = i  # index into the centroid reference for LOO self-exclusion

    print(">> fitting GP hyperparameters on centroids (one-time)...", flush=True)
    k = (ConstantKernel(1.0, (1e-2, 1e4))
         * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=1.5)
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e4)))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(XYn, ANCC)
    print(f"   learned kernel: {gp.kernel_}", flush=True)

    kf = gp.kernel_
    k1 = kf.k1                                   # Constant*Matern (signal)
    const = float(k1.diag(XYn[:1])[0])           # k1(x,x) constant amplitude
    _G.update(dict(XYn=XYn, ANCC=ANCC, mu=mu, sd=sd, k1=k1, kf=kf,
                   ymean=float(ANCC.mean()), const=const))

    print(">> TRAIN GP (LOO) over wells...", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, r in enumerate(ex.map(_worker_train, train, chunksize=2)):
            parts.append(r)
            if (i + 1) % 100 == 0:
                print(f"   train {i+1}/{len(train)}", flush=True)
    tr_out = pd.concat(parts, ignore_index=True)
    tr_out.to_parquet(OUT_TR)
    print(f">> wrote {OUT_TR}  shape={tr_out.shape}", flush=True)

    print(">> loading TEST trajectories...", flush=True)
    te_paths = sorted(glob.glob(str(RAW_TE / "*__horizontal_well.csv")))
    test = [load_traj(p) for p in te_paths]
    test = [d for d in test if d is not None]
    print(f">> {len(test)} test wells", flush=True)
    if test:
        parts = []
        with ProcessPoolExecutor(max_workers=12) as ex:
            for i, r in enumerate(ex.map(_worker_test, test, chunksize=1)):
                parts.append(r)
        te_out = pd.concat(parts, ignore_index=True)
        te_out.to_parquet(OUT_TE)
        print(f">> wrote {OUT_TE}  shape={te_out.shape}", flush=True)

    print("=== GP FEATURE BUILD DONE ===", flush=True)


if __name__ == "__main__":
    main()
