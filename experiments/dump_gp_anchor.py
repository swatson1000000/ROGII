"""Save the centroid-GP hyperparameters as a kernel artifact + VALIDATE consistency.

The training GP features (gp_feats_{train,test}.parquet) were built by gp_feature_build.py,
which fit the GP once on the 766 train centroids but did NOT persist the hyperparameters.
The inference kernel must reproduce the SAME posterior. This script:

  1. Re-fits the GP on the train centroids (same kernel spec) and dumps
     models/konbu_gp/gp_anchor.json  {mu, sd, ymean, constant_value, length_scale, noise_level}.
  2. VALIDATES: rebuilds the posterior from those saved params and recomputes gp_ancc/gp_std
     for the 3 visible test wells, comparing to the already-built gp_feats_test.parquet.
     If max|Dancc| is tiny, the saved params reproduce the build -> kernel will be consistent.
     If not, FAIL LOUD (the fit was not deterministic; rebuild features with fixed seed).

Centroids are computed EXACTLY as in gp_feature_build.load_traj:
  x_med/y_med = median(X)/median(Y) over ALL rows; ancc_med = median(ANCC) over ANCC rows.
"""
from pathlib import Path
import glob, json
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.linalg import cholesky, solve_triangular

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW_TR = ROOT / "data/raw/train"
RAW_TE = ROOT / "data/raw/test"
ART = ROOT / "data/processed/konbu"
MODELSG = ROOT / "models/konbu_gp"


def centroids(paths, need_prefix):
    XY, A = [], []
    for p in paths:
        h = pd.read_csv(p)
        if not {"X", "Y"}.issubset(h.columns):
            continue
        if "ANCC" not in h.columns or not h["ANCC"].notna().any():
            continue
        if need_prefix:
            if "TVT_input" not in h.columns or not h["TVT_input"].isna().any():
                continue
            ms = int(np.flatnonzero(h["TVT_input"].isna().to_numpy())[0])
            if ms == 0:
                continue
        XY.append([float(h["X"].median()), float(h["Y"].median())])
        A.append(float(h["ANCC"].median()))
    return np.array(XY), np.array(A)


def main():
    tr_paths = sorted(glob.glob(str(RAW_TR / "*__horizontal_well.csv")))
    XY, ANCC = centroids(tr_paths, need_prefix=True)
    print(f">> {len(XY)} train centroids", flush=True)

    mu, sd = XY.mean(0), XY.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    XYn = (XY - mu) / sd

    k = (ConstantKernel(1.0, (1e-2, 1e4))
         * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=1.5)
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e4)))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2, random_state=0)
    gp.fit(XYn, ANCC)
    print(f">> learned kernel: {gp.kernel_}", flush=True)

    k1 = gp.kernel_.k1            # Constant*Matern
    const = float(k1.k1.constant_value)
    ls = [float(x) for x in np.atleast_1d(k1.k2.length_scale)]
    noise = float(gp.kernel_.k2.noise_level)
    anchor = dict(mu=[float(x) for x in mu], sd=[float(x) for x in sd],
                  ymean=float(ANCC.mean()), constant_value=const,
                  length_scale=ls, noise_level=noise)
    MODELSG.mkdir(parents=True, exist_ok=True)
    json.dump(anchor, open(MODELSG / "gp_anchor.json", "w"), indent=2)
    print(f">> wrote {MODELSG/'gp_anchor.json'}: {anchor}", flush=True)

    # ---- rebuild posterior from saved params (the kernel's exact path) ----
    k1f = ConstantKernel(const) * Matern(length_scale=ls, nu=1.5)
    kf = k1f + WhiteKernel(noise_level=noise)
    ymean = anchor["ymean"]
    K = kf(XYn)
    L = cholesky(K + 1e-6 * np.eye(len(XYn)), lower=True)
    alpha = solve_triangular(L.T, solve_triangular(L, ANCC - ymean, lower=True), lower=False)

    def impute(xy_q):
        qn = (xy_q - mu) / sd
        Ks = k1f(qn, XYn)
        mean = ymean + Ks @ alpha
        v = solve_triangular(L, Ks.T, lower=True)
        std = np.sqrt(np.clip(const - np.sum(v * v, axis=0), 0.0, None))
        return mean, std

    # ---- validate against the already-built gp_feats_test ----
    gte = pd.read_parquet(ART / "gp_feats_test.parquet")  # well,row_idx,gr,gp_ancc,gp_std,gp_tvt_abs
    te_paths = sorted(glob.glob(str(RAW_TE / "*__horizontal_well.csv")))
    max_da, max_ds = 0.0, 0.0
    for p in te_paths:
        wid = Path(p).name.replace("__horizontal_well.csv", "")
        h = pd.read_csv(p)
        xy = h[["X", "Y"]].to_numpy(np.float64)
        m, s = impute(xy)
        ref = gte[gte["well"] == wid].sort_values("row_idx")
        if len(ref) != len(m):
            print(f"   WARN {wid}: row mismatch {len(ref)} vs {len(m)}", flush=True)
            continue
        da = float(np.max(np.abs(m - ref["gp_ancc"].to_numpy())))
        ds = float(np.max(np.abs(s - ref["gp_std"].to_numpy())))
        max_da, max_ds = max(max_da, da), max(max_ds, ds)
        print(f"   {wid}: max|d ancc|={da:.4g}  max|d std|={ds:.4g}", flush=True)

    print(f"\n>>> CONSISTENCY: max|d ancc|={max_da:.4g} ft  max|d std|={max_ds:.4g}", flush=True)
    ok = max_da < 0.5
    print(">>> VERDICT:", "saved params REPRODUCE the build -> kernel safe"
          if ok else "MISMATCH -> rebuild features with fixed seed before shipping", flush=True)


if __name__ == "__main__":
    main()
