"""BET 1' — Gate 1: does test-sibling mutual-anchoring reduce the b_well-adjusted
eval-zone tvt error under spatial block-holdout?

Gate 0 found the ANCC imputation residual r = ANCC_true - ANCC_train_imputed is
strongly STRUCTURED but long-wavelength (range ~21,000 ft). The catch: per-well
b_well_est = b_well_true + mean_prefix(r) already ABSORBS the local (long-wavelength)
residual, so the eval-zone tvt error is the WITHIN-WELL DRIFT of r, not its absolute
value. Gate 0 measured one layer too high. Gate 1 measures the layer that matters.

Mechanism under test: each test well's KNOWN PREFIX gives an implied ANCC anchor
    implied_ANCC(prefix row) = TVT_input + Z - b_well_est
with b_well_est from train-only imputation. Algebra: implied_ANCC = ANCC_true -
mean_prefix(r), i.e. the b_well subtraction strips each anchor's prefix-mean residual,
leaving only the de-meaned (within-well) residual structure to propagate spatially.
A SIBLING test well's prefix anchors near well w's eval rows can sharpen w's ANCC
imputation there -- IF that de-meaned residual is spatially coherent at the inter-well
scale present inside a (clustered) held block.

Three reference conditions, SAME local-weighted-plane imputer, scored as the
b_well-adjusted eval tvt error (= what the production tvt_formula feature would be):
  base : impute eval ANCC from TRAIN rows only                  (current method)
  self : train + the well's OWN prefix-implied anchors          (self-transductive;
         always available, does NOT depend on hidden clustering)
  sib  : train + OTHER held wells' prefix-implied anchors,      (the transductive lever;
         self-excluded                                           depends on clustering)

Decomposition that answers the strategy question:
  * base->self large, sib ~= self  -> the win is SELF-anchoring (simple, robust, no
        clustering dependence). Build a self-prefix ANCC feature; skip sibling pooling.
  * sib >> self                     -> genuine SIBLING-pooling value (clustering-dependent;
        bet rides on the hidden ~200 wells being clustered, which we can't observe).
  * all three ~equal                -> DEAD. b_well already captured everything; the
        live spatial imputer has no OOD hole for prefix anchors to fill. Stop.

This also yields the FIRST faithful block-holdout number for the LIVE (plane) imputer
(gp_block_holdout's plane arm was a weak reimpl; only its GP arm was trustworthy).

Reuses load/blocks from gp_block_holdout.py. Honest block-holdout: hold a whole KMeans
block out of the train reference (simulates a hidden cluster with no train neighbor).

Run: nohup python -u experiments/spatial_transductive_gate1.py \
       > log/gate1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
K_BLOCKS = 6
K_NN = 40            # nearest anchor ROWS for the local weighted plane
S_TRAIN = 20         # stride: train anchor rows (the surface reference)
S_PREF = 10          # stride: prefix anchor rows contributed by a well
S_EVAL = 30          # stride: eval rows scored (RMSE estimate; keeps it tractable)
HELD_BUF = 80        # over-query buffer on held tree before wid filtering


def load_well(p):
    h = pd.read_csv(p)
    for c in ("TVT_input", "TVT", "ANCC", "X", "Y", "Z"):
        if c not in h.columns:
            return None
    if not h["ANCC"].notna().any():
        return None
    mask = h["TVT_input"].isna().to_numpy()
    if not mask.any():
        return None
    ms = int(np.flatnonzero(mask)[0])
    if ms < 20:                          # need a usable prefix for b_well + anchors
        return None
    ancc = h["ANCC"].to_numpy(np.float64)
    return dict(
        wid=Path(p).name.replace("__horizontal_well.csv", ""),
        x=h["X"].to_numpy(np.float64), y=h["Y"].to_numpy(np.float64),
        z=h["Z"].to_numpy(np.float64), tvt=h["TVT"].to_numpy(np.float64),
        tvt_input=h["TVT_input"].to_numpy(np.float64),
        ancc=ancc, ancc_ok=np.isfinite(ancc), ms=ms,
        x_med=float(h["X"].median()), y_med=float(h["Y"].median()),
    )


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def idw_at(qx, qy, nx, ny, nval):
    """Inverse-distance-weighted ANCC over the K neighbor anchors (nx,ny,nval) of one
    query point. Robust row-level estimator matching the production RowKNN (a local
    plane is ill-conditioned here -- row anchors lie on a near-1D well trajectory, so a
    plane fit extrapolates wildly; IDW does not)."""
    dn = np.sqrt((nx - qx) ** 2 + (ny - qy) ** 2)
    w = 1.0 / (dn + 1e-6) ** 2
    return float(np.dot(w, nval) / w.sum())


def impute_points(qx, qy, qwid, tree_tr, tr_xy, tr_val,
                  tree_hd, hd_xy, hd_val, hd_wid, mode):
    """Impute ANCC at query points (qx,qy) belonging to wells qwid.
    mode: 'base' (train only) | 'self' (train + own-wid held) | 'sib'
    (train + held with wid != own). Local weighted plane over K_NN nearest
    merged anchors."""
    out = np.empty(len(qx))
    # batched train neighbors
    dtr, itr = tree_tr.query(np.column_stack([qx, qy]), k=K_NN)
    if tree_hd is not None:
        khd = min(K_NN + HELD_BUF, len(hd_val))
        dhd, ihd = tree_hd.query(np.column_stack([qx, qy]), k=khd)
    for q in range(len(qx)):
        nx = tr_xy[itr[q], 0]; ny = tr_xy[itr[q], 1]; nv = tr_val[itr[q]]
        nd = dtr[q]
        if tree_hd is not None and mode != "base":
            hi = ihd[q]
            hw = hd_wid[hi]
            keep = (hw == qwid[q]) if mode == "self" else (hw != qwid[q])
            if keep.any():
                hi = hi[keep]; hd = dhd[q][keep]
                nx = np.concatenate([nx, hd_xy[hi, 0]])
                ny = np.concatenate([ny, hd_xy[hi, 1]])
                nv = np.concatenate([nv, hd_val[hi]])
                nd = np.concatenate([nd, hd])
        # take the K_NN nearest of the merged set
        if len(nv) > K_NN:
            sel = np.argpartition(nd, K_NN)[:K_NN]
            nx, ny, nv = nx[sel], ny[sel], nv[sel]
        out[q] = idw_at(qx[q], qy[q], nx, ny, nv)
    return out


def main():
    paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    print(f">> loading {len(paths)} wells...", flush=True)
    wells = [load_well(p) for p in paths]
    wells = [w for w in wells if w is not None]
    n = len(wells)
    print(f">> {n} usable wells", flush=True)

    XYc = np.array([[w["x_med"], w["y_med"]] for w in wells])
    mu, sd = XYc.mean(0), XYc.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
    labels = KMeans(n_clusters=K_BLOCKS, n_init=10, random_state=0).fit_predict((XYc - mu) / sd)
    for b in range(K_BLOCKS):
        print(f"   block {b}: {(labels==b).sum()} wells", flush=True)

    acc = {m: [] for m in ("base", "self", "sib")}
    blk_rmse = []
    for b in range(K_BLOCKS):
        held = np.flatnonzero(labels == b)
        ref = np.flatnonzero(labels != b)

        # ---- train anchor rows (the surface reference) ----
        tx, ty, tv = [], [], []
        for j in ref:
            d = wells[j]
            ok = d["ancc_ok"]
            idx = np.flatnonzero(ok)[::S_TRAIN]
            tx.append(d["x"][idx]); ty.append(d["y"][idx]); tv.append(d["ancc"][idx])
        tr_xy = np.column_stack([np.concatenate(tx), np.concatenate(ty)])
        tr_val = np.concatenate(tv)
        tree_tr = cKDTree(tr_xy)

        # ---- held wells' prefix-implied ANCC anchors (b_well from TRAIN-only impute) ----
        hx, hy, hv, hw = [], [], [], []
        b_est = {}
        for j in held:
            d = wells[j]; ms = d["ms"]
            # b_well_est = median(TVT_input + Z - ANCC_train_imputed) over prefix
            pidx = np.arange(0, ms, max(1, ms // 60))      # ~60 prefix samples
            anc_pre = impute_points(d["x"][pidx], d["y"][pidx], None,
                                    tree_tr, tr_xy, tr_val, None, None, None, None, "base")
            be = float(np.median(d["tvt_input"][pidx] + d["z"][pidx] - anc_pre))
            b_est[j] = be
            aidx = np.arange(0, ms, S_PREF)
            implied = d["tvt_input"][aidx] + d["z"][aidx] - be
            hx.append(d["x"][aidx]); hy.append(d["y"][aidx])
            hv.append(implied); hw.append(np.full(len(aidx), j))
        hd_xy = np.column_stack([np.concatenate(hx), np.concatenate(hy)])
        hd_val = np.concatenate(hv); hd_wid = np.concatenate(hw)
        tree_hd = cKDTree(hd_xy)

        # ---- score eval rows of each held well under base/self/sib ----
        qx, qy, qw, qz, qt = [], [], [], [], []
        for j in held:
            d = wells[j]; ms = d["ms"]
            eidx = np.arange(ms, len(d["x"]), S_EVAL)
            qx.append(d["x"][eidx]); qy.append(d["y"][eidx]); qz.append(d["z"][eidx])
            qt.append(d["tvt"][eidx]); qw.append(np.full(len(eidx), j))
        qx = np.concatenate(qx); qy = np.concatenate(qy); qz = np.concatenate(qz)
        qt = np.concatenate(qt); qw = np.concatenate(qw)
        bvec = np.array([b_est[j] for j in qw])     # each held well's b_well_est

        blk = {}
        for m in ("base", "self", "sib"):
            anc = impute_points(qx, qy, qw, tree_tr, tr_xy, tr_val,
                                tree_hd, hd_xy, hd_val, hd_wid, m)
            err = (-qz + anc + bvec) - qt
            acc[m].append(err)
            blk[m] = rmse(err)
        blk_rmse.append((b, len(held), blk))
        print(f"   block {b}: base={blk['base']:.2f}  self={blk['self']:.2f}  "
              f"sib={blk['sib']:.2f}  (n_eval={len(qt):,})", flush=True)

    print("\n=== GATE 1 VERDICT (b_well-adjusted eval tvt RMSE, block-holdout) ===", flush=True)
    R = {m: rmse(np.concatenate(acc[m])) for m in acc}
    print(f"  base (train only)            RMSE = {R['base']:.3f}", flush=True)
    print(f"  self (+ own prefix anchors)  RMSE = {R['self']:.3f}   (delta {R['self']-R['base']:+.3f})", flush=True)
    print(f"  sib  (+ sibling anchors)     RMSE = {R['sib']:.3f}   (delta {R['sib']-R['base']:+.3f})", flush=True)
    print(f"  sibling-on-top-of-self       delta = {R['sib']-R['self']:+.3f}", flush=True)
    print("", flush=True)
    d_self = R["base"] - R["self"]
    d_sib_extra = R["self"] - R["sib"]
    if d_self < 0.25 and (R["base"] - R["sib"]) < 0.25:
        print("  >> ALL THREE ~EQUAL: b_well already captured the surface; prefix anchors", flush=True)
        print("     add ~nothing. The live plane imputer has no OOD hole to fill. BET 1' DEAD.", flush=True)
    elif d_sib_extra < 0.10:
        print("  >> WIN IS SELF-ANCHORING (sibling adds ~nothing on top). Build a self-prefix", flush=True)
        print("     ANCC feature -- simple, robust, NO dependence on hidden clustering. Skip", flush=True)
        print("     sibling pooling. Then gate the FEATURE in-stack via BLOCK-HOLDOUT OOF (Bet 3).", flush=True)
    else:
        print("  >> GENUINE SIBLING-POOLING VALUE. Transductive lever is real but its EV rides", flush=True)
        print("     on the hidden ~200 wells being CLUSTERED (block-holdout's geometry), which we", flush=True)
        print("     cannot observe. Quantify recovery vs the unobservable; decide if worth a", flush=True)
        print("     submission probe. Gate any feature in-stack via BLOCK-HOLDOUT OOF (Bet 3).", flush=True)
    print("GATE1 DONE", flush=True)


if __name__ == "__main__":
    main()
