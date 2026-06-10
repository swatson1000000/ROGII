"""BET 5 Stage B (1/4): build UK1 dip-trend-kriging ANCC features for the TRAIN matrix, honest LOO.

Adds a UK-based TVT estimate parallel to the production dense-RowKNN path (tvt_dense), so on OOD
wells where dense ANCC collapses (block-holdout showed 147 ft / max 1093) the UK estimate stays sane
(36 ft / max 203). Two new features, mirroring the kernel's dense path (rogii_frontier_inference.py
L825-831, L901):
    tvt_uk_d   = (-Z + uk_ancc + b_uk) - last_tvt            # parallel to tvt_dense_d
    uk_vs_dense= tvt_uk_d - tvt_dense_d                       # disagreement (parallel to spatial_vs_dense)
where uk_ancc = UK1(train-centroid ANCC) at the row (X,Y), b_uk = median(ktvt + Z_kn - uk_kn) over the
known prefix (mirrors b_d). b_well absorbs the constant offset, so the OOD value lives in tvt_uk_d's
ALONG-WELL gradient = the dip ([[bet5-uk1-ood-robust-but-test-interpolates]], [[transductive-thread-dead]]).

HONESTY: per-well LOO -- each train well's uk_ancc uses the OTHER wells' centroids as the kriging
reference (no self-leak), matching how the kernel computes it at inference (full-train ref, test well
genuinely absent). Covariance hyperparams fit once on all centroids (global, like GP/kriging norm; the
kernel uses the same fit). Aligns to train_feats by id (row index parsed from '<wid>_<i>').

Run: nohup python -u experiments/bet5_build_uk_feats.py > log/bet5_ukfeat_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os, importlib.util
import numpy as np
import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
FR = ROOT / "data/processed/frontier_seeded"
OUT = ROOT / "data/processed/uk_feats.parquet"
CENT_OUT = ROOT / "models/frontier/uk_centroids.npz"   # artifact for the kernel (Stage B 4/4)

_spec = importlib.util.spec_from_file_location("b5", str(ROOT / "experiments/bet5_kriging_block_holdout.py"))
b5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(b5)
trend_basis, _pdist2, fit_uk_cov = b5.trend_basis, b5._pdist2, b5.fit_uk_cov
DEGREE = 1


def uk_predict(XYr_n, Ar, cov, Qn):
    """Universal kriging (dual form) of ANCC: ref=(XYr_n std, Ar), queries Qn std. Returns pred[len(Qn)]."""
    sill, nugget, ell = cov
    n = len(XYr_n); F = trend_basis(XYr_n, DEGREE); p = F.shape[1]
    C = sill * np.exp(-np.sqrt(_pdist2(XYr_n, XYr_n)) / ell)
    C[np.diag_indices_from(C)] += nugget
    M = np.zeros((n + p, n + p)); M[:n, :n] = C; M[:n, n:] = F; M[n:, :n] = F.T
    cq = sill * np.exp(-np.sqrt(_pdist2(XYr_n, Qn)) / ell)
    RHS = np.vstack([cq, trend_basis(Qn, DEGREE).T])
    try:
        sol = np.linalg.solve(M, RHS)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(M, RHS, rcond=None)[0]
    return (sol[:n, :].T @ Ar)


def main():
    print(">> load train_feats (well,id,z,last_known_tvt,tvt_dense_d)...", flush=True)
    tf = pd.read_parquet(FR / "train_feats.parquet",
                         columns=["well", "id", "z", "last_known_tvt", "tvt_dense_d"])
    tf["_ri"] = tf["id"].str.rsplit("_", n=1).str[-1].astype(int)
    wells_in_tf = list(tf["well"].unique())
    print(f"   {len(tf):,} rows, {len(wells_in_tf)} wells", flush=True)

    # ---- centroids (x_med,y_med,ancc_med) for every train well present in the raw dir ----
    print(">> build centroids from raw...", flush=True)
    paths = {os.path.basename(p).replace("__horizontal_well.csv", ""): p
             for p in glob.glob(str(RAW / "*__horizontal_well.csv"))}
    cent = {}
    for wid, p in paths.items():
        h = pd.read_csv(p, usecols=lambda c: c in ("X", "Y", "ANCC"))
        if "ANCC" not in h.columns or not h["ANCC"].notna().any():
            continue
        cent[wid] = (float(h["X"].median()), float(h["Y"].median()), float(h["ANCC"].median()))
    cw = list(cent.keys())
    XY = np.array([[cent[w][0], cent[w][1]] for w in cw])
    A = np.array([cent[w][2] for w in cw])
    mu, sd = XY.mean(0), XY.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
    XYn = (XY - mu) / sd
    cidx = {w: i for i, w in enumerate(cw)}
    cov = fit_uk_cov(XYn, A, DEGREE)
    print(f"   {len(cw)} centroids; UK1 cov sill={cov[0]:.4g} nugget={cov[1]:.4g} ell={cov[2]}", flush=True)
    np.savez(CENT_OUT, xy=XY, ancc=A, mu=mu, sd=sd,
             sill=cov[0], nugget=cov[1], ell=cov[2], degree=DEGREE, wids=np.array(cw))
    print(f"   saved kernel artifact -> {CENT_OUT}", flush=True)

    # ---- per-well LOO uk_ancc at the train_feats eval rows + known-zone b_uk ----
    print(">> per-well LOO UK imputation...", flush=True)
    grp = {w: g for w, g in tf.groupby("well", sort=False)}
    rec_id, rec_ukd, rec_anc = [], [], []
    miss = 0
    for k, wid in enumerate(wells_in_tf):
        g = grp[wid]
        p = paths.get(wid)
        h = pd.read_csv(p, usecols=lambda c: c in ("X", "Y", "Z", "TVT_input"))
        kn = h["TVT_input"].notna().to_numpy()
        if wid not in cidx or kn.sum() < 5:
            miss += len(g); continue
        ref = np.array([i for i in range(len(cw)) if i != cidx[wid]])
        Xa = h["X"].to_numpy(np.float64); Ya = h["Y"].to_numpy(np.float64); Za = h["Z"].to_numpy(np.float64)
        # known-zone b_uk
        Qk = np.column_stack([(Xa[kn] - mu[0]) / sd[0], (Ya[kn] - mu[1]) / sd[1]])
        uk_kn = uk_predict(XYn[ref], A[ref], cov, Qk)
        ktvt = h["TVT_input"].to_numpy(np.float64)[kn]
        b_uk = float(np.median(ktvt + Za[kn] - uk_kn))
        last_tvt = float(ktvt[-1])
        # eval rows = the train_feats rows for this well (align by row index)
        ri = g["_ri"].to_numpy()
        Qe = np.column_stack([(Xa[ri] - mu[0]) / sd[0], (Ya[ri] - mu[1]) / sd[1]])
        uk_ev = uk_predict(XYn[ref], A[ref], cov, Qe)
        tvt_uk = -Za[ri] + uk_ev + b_uk
        rec_id.append(g["id"].to_numpy())
        rec_ukd.append((tvt_uk - last_tvt).astype(np.float32))
        rec_anc.append(uk_ev.astype(np.float32))
        if (k + 1) % 100 == 0:
            print(f"   {k+1}/{len(wells_in_tf)}", flush=True)
    out = pd.DataFrame({"id": np.concatenate(rec_id),
                        "tvt_uk_d": np.concatenate(rec_ukd),
                        "uk_ancc": np.concatenate(rec_anc)})
    out = tf[["id", "tvt_dense_d"]].merge(out, on="id", how="left")
    out["uk_vs_dense"] = (out["tvt_uk_d"] - out["tvt_dense_d"]).astype(np.float32)
    nbad = int(out["tvt_uk_d"].isna().sum())
    print(f">> {len(out):,} rows, {nbad} unmatched (missing centroid / short prefix)", flush=True)
    out[["id", "tvt_uk_d", "uk_ancc", "uk_vs_dense"]].to_parquet(OUT)
    print(f">> saved {OUT}", flush=True)
    print("BET5 UKFEAT DONE", flush=True)


if __name__ == "__main__":
    main()
