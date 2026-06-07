"""BET 1' — Gate 0: is there a propagatable spatial signal in the ANCC imputation residual?

The transductive lever (BET 1') proposes: if the ~200 hidden wells are spatially
clustered, a test well's prefix-implied ANCC can anchor the formation surface for a
SIBLING test well — an anchor absent from the train set. That only works if the thing
being propagated exists and is shareable between wells.

What gets propagated is the residual of the CURRENT (train-only) imputation:
    r(X,Y) = ANCC_true  -  ANCC_imputed_from_train
A sibling well s reveals r at its prefix; that helps well w iff r is spatially
correlated at the inter-well scale (r at s predicts r at w).

Gate 0 (cheapest; kills the lever before any imputer is built): under honest spatial
block-holdout (hold a whole KMeans block out of the train reference, the OOD condition
that collapsed GP), impute ANCC per held well from the reduced reference, take the
per-well residual r_med, and fit a CENTROID-LEVEL VARIOGRAM of r over the held wells.

Decision:
  * STRUCTURED  (structure ratio = (sill - nugget)/sill is meaningfully > 0,
                 i.e. nearby wells share r): a propagatable signal exists -> proceed to Gate 1.
  * NUGGET-DOMINATED (ratio ~ 0): r is white noise between wells -> nothing to
                 propagate -> BET 1' is DEAD, stop. No imputer to build.

The CRUX disambiguation (answers "is this geology train can't see, or just bias more
train anchors would fix?"): report the variogram RANGE vs the median train-well
nearest-neighbor spacing.
  * range  <  train spacing  -> sub-train-resolution structure: train density genuinely
                 cannot resolve it, but dense test siblings could -> REAL transductive signal.
  * range  >> train spacing  -> coarse train-imputation bias: more TRAIN anchors would
                 capture it too -> not test-unique -> the "transductive" framing is a mirage.

Reuses the block-holdout machinery from experiments/gp_block_holdout.py (same load,
plane-KNN imputer, KMeans blocks). plane-KNN is the production-analog imputer for the
formation surface (the GP arm is the dead one; we gate the live spatial backbone).

Run: nohup python -u experiments/spatial_transductive_gate0.py \
       > log/gate0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
    ancc = h["ANCC"].to_numpy(np.float64)
    ok = np.isfinite(ancc)
    if ok.sum() < 5:
        return None
    return dict(
        wid=Path(p).name.replace("__horizontal_well.csv", ""),
        x=h["X"].to_numpy(np.float64), y=h["Y"].to_numpy(np.float64),
        ancc=ancc, ok=ok,
        x_med=float(h["X"].median()), y_med=float(h["Y"].median()),
        ancc_med=float(np.nanmedian(ancc)),
    )


def plane_impute_rows(d, XYref_raw, Aref):
    """Local weighted-plane ANCC at every row of well d; K nearest reference centroids
    to the well centroid, ANCC ~ a + b*X + c*Y (distance-weighted). Identical to the
    production-analog imputer in gp_block_holdout.plane_impute_well."""
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


def variogram(coords, vals, n_bins=15, max_frac=0.5):
    """Empirical isotropic semivariogram of `vals` at `coords` (N,2).
    Returns (bin_centers, gamma, n_pairs, sill)."""
    n = len(vals)
    ii, jj = np.triu_indices(n, k=1)
    d = np.sqrt(((coords[ii] - coords[jj]) ** 2).sum(1))
    g = 0.5 * (vals[ii] - vals[jj]) ** 2
    dmax = np.quantile(d, max_frac)  # ignore the far tail (few, noisy pairs)
    edges = np.linspace(0, dmax, n_bins + 1)
    centers, gamma, npair = [], [], []
    for k in range(n_bins):
        m = (d >= edges[k]) & (d < edges[k + 1])
        if m.sum() < 30:
            continue
        centers.append(0.5 * (edges[k] + edges[k + 1]))
        gamma.append(float(g[m].mean()))
        npair.append(int(m.sum()))
    sill = float(np.var(vals))  # total variance = variogram sill estimate
    return np.array(centers), np.array(gamma), np.array(npair), sill


def main():
    paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    print(f">> loading {len(paths)} wells...", flush=True)
    wells = [load_well(p) for p in paths]
    wells = [w for w in wells if w is not None]
    n = len(wells)
    print(f">> {n} usable wells", flush=True)

    XY = np.array([[w["x_med"], w["y_med"]] for w in wells])
    A = np.array([w["ancc_med"] for w in wells])
    mu, sd = XY.mean(0), XY.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    XYn = (XY - mu) / sd

    # train-well nearest-neighbor spacing (raw feet) -- the resolution train density gives us
    from scipy.spatial import cKDTree
    tree = cKDTree(XY)
    nn_d = tree.query(XY, k=2)[0][:, 1]
    nn_med = float(np.median(nn_d))
    print(f">> train-well nearest-neighbor spacing: median={nn_med:.0f} ft  "
          f"p25={np.percentile(nn_d,25):.0f}  p75={np.percentile(nn_d,75):.0f}", flush=True)

    # ---- honest block-holdout residual r = ANCC_true - ANCC_imputed_from_reduced_train ----
    print(f"\n>> block-holdout (K={K_BLOCKS} KMeans blocks); per-well residual r_med ...", flush=True)
    labels = KMeans(n_clusters=K_BLOCKS, n_init=10, random_state=0).fit_predict(XYn)
    r_med = np.full(n, np.nan)
    for b in range(K_BLOCKS):
        held = np.flatnonzero(labels == b)
        ref = np.flatnonzero(labels != b)
        XYref, Aref = XY[ref], A[ref]
        for j in held:
            d = wells[j]
            anc_imp = plane_impute_rows(d, XYref, Aref)
            ok = d["ok"]
            r_med[j] = float(np.median(d["ancc"][ok] - anc_imp[ok]))
        print(f"   block {b}: {len(held)} wells  "
              f"r_med RMSE={np.sqrt(np.mean(r_med[held]**2)):.2f}", flush=True)

    # ---- centroid-level variogram of the residual field ----
    print("\n>> centroid-level variogram of r_med (the propagatable-signal test) ...", flush=True)
    centers, gamma, npair, sill = variogram(XY, r_med)
    print(f"   sill (Var[r_med]) = {sill:.2f}   (RMSE of r_med = {np.sqrt(np.mean(r_med**2)):.2f})", flush=True)
    print(f"   {'lag_ft':>10} {'gamma':>10} {'n_pairs':>9}", flush=True)
    for c, g, npp in zip(centers, gamma, npair):
        print(f"   {c:10.0f} {g:10.2f} {npp:9d}", flush=True)

    # nugget = variogram at the shortest resolved lag; structure ratio = (sill - nugget)/sill
    nugget = float(gamma[0]) if len(gamma) else float("nan")
    struct_ratio = (sill - nugget) / sill if sill > 0 else float("nan")
    # correlation range: first lag where gamma reaches 95% of sill (proxy for the range)
    reached = np.flatnonzero(gamma >= 0.95 * sill)
    rng = float(centers[reached[0]]) if len(reached) else float("nan")

    print("\n=== GATE 0 VERDICT ===", flush=True)
    print(f"  nugget (gamma @ {centers[0]:.0f} ft) = {nugget:.2f}", flush=True)
    print(f"  sill                  = {sill:.2f}", flush=True)
    print(f"  structure ratio (sill-nugget)/sill = {struct_ratio:.3f}", flush=True)
    print(f"  estimated range (gamma->95% sill)  = {rng:.0f} ft", flush=True)
    print(f"  train-well NN spacing (median)     = {nn_med:.0f} ft", flush=True)
    print("", flush=True)
    if not np.isfinite(struct_ratio) or struct_ratio < 0.10:
        print("  >> NUGGET-DOMINATED: r is ~white noise between wells. Nothing to", flush=True)
        print("     propagate from a sibling test well. BET 1' is DEAD -- stop, no imputer.", flush=True)
    else:
        print("  >> STRUCTURED: nearby wells share residual -> a propagatable signal EXISTS.", flush=True)
        if np.isfinite(rng) and rng < nn_med:
            print(f"     RANGE ({rng:.0f} ft) < TRAIN SPACING ({nn_med:.0f} ft): sub-train-resolution", flush=True)
            print("     structure -- train density cannot resolve it, dense test siblings could.", flush=True)
            print("     => REAL transductive signal. Proceed to Gate 1 (mutual-anchoring probe).", flush=True)
        else:
            print(f"     RANGE ({rng:.0f} ft) >= TRAIN SPACING ({nn_med:.0f} ft): coarse bias that", flush=True)
            print("     MORE TRAIN ANCHORS would also fix -> not test-unique. The 'transductive'", flush=True)
            print("     framing is a MIRAGE; Gate 1 would over-credit. Treat as DEAD unless", flush=True)
            print("     Gate 1 shows test-anchors beat an equal number of extra train-anchors.", flush=True)
    print("GATE0 DONE", flush=True)


if __name__ == "__main__":
    main()
