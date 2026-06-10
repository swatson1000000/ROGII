"""BET 5 — CORRECTED Stage A gate: UK1 (dip-trend kriging) vs the REAL PRODUCTION ANCC imputer
(dense RowKNN, the dominant Phase-3 feature by GBM gain) under IDENTICAL block-holdout.

WHY THIS RE-RUN. The first gate (bet5_kriging_block_holdout.py) compared UK1 only against the weak
centroid-plane reimpl (LOO 100, a known artifact) and a GP reference. UK1 won there (block 36.05 vs
plane 57.11 vs GP 47.95, tighter tail, half GP's degradation) -- but that did NOT test the imputer
that actually feeds the GBM: dense RowKNN (src/spatial.py RowKNN, K=20 IDW over stride-3 anchor
rows, ~17 ft combined LOO). If RowKNN degrades less OOD than UK1, BET 5 dies; if UK1 still wins
head-to-head against RowKNN under block-holdout, it earns Stage B. The first gate's degradation
side-clause also misfired (benchmarked vs plane's nonsensical -43 degradation from its broken LOO);
here the density-coupling tell is referenced to GP (clean LOO 24.50), the estimator that actually
regressed the LB.

FIDELITY. Under block-holdout the held block is GENUINELY absent from the reference, so RowKNN runs
in its TEST mode (self_wid=None, no self-exclusion) -- exactly how it runs on the hidden set at
inference. Both imputers are scored by the same b_well-calibrated tvt_formula (score_well): b absorbs
the constant ANCC offset, so only the along-well ANCC gradient (the dip) survives -> this isolates
the dip-placement quality, the thing BET 5 is about.

==================  PRE-REGISTERED GATE (set BEFORE results; references GP, not broken-plane)  =====
Decision is on BLOCK-HOLDOUT. Let R = RowKNN block RMSE, U = UK1 block RMSE; deg(e)=block-LOO.
  CLEAN PASS (-> Stage B): U <= 0.95 * R  AND  deg(UK1) <= deg(GP)   (UK1 beats production imputer
     OOD by >=5% AND is less density-coupled than the LB-regressing GP).
  KILL: U >= R   (UK1 does not beat the production imputer OOD -- BET 5 dies, cleanly).
  AMBIGUOUS (0.95*R < U < R): treat as NO (the GP lesson: ambiguous in this family = no).
Stage A is necessary-not-sufficient: a clean pass earns the GBM rebuild + OOF gate vs 9.30, not a
submission.
===================================================================================================

Run: nohup python -u experiments/bet5_corrected_gate.py > log/bet5_corrected_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import importlib.util
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
K_BLOCKS = 6
ROW_K = 20
ROW_STRIDE = 3
LOO_EVAL_STRIDE = 3        # subsample eval rows for the (expensive) RowKNN/UK LOO RMSE estimate

# reuse the validated math from the first gate (no reimpl drift)
_spec = importlib.util.spec_from_file_location("b5", str(ROOT / "experiments/bet5_kriging_block_holdout.py"))
b5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b5)
rmse, report, score_well = b5.rmse, b5.report, b5.score_well
fit_uk_cov, uk_impute_well = b5.fit_uk_cov, b5.uk_impute_well
fit_gp_kernel, gp_impute_well = b5.fit_gp_kernel, b5.gp_impute_well


def load_well(p):
    """Like b5.load_well but also keep the well's stride-3 anchor (X,Y,ANCC) rows for the
    dense RowKNN reference."""
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
    anc = h[["X", "Y", "ANCC"]].dropna().iloc[::ROW_STRIDE]
    return dict(
        wid=Path(p).name.replace("__horizontal_well.csv", ""),
        x=h["X"].to_numpy(np.float64), y=h["Y"].to_numpy(np.float64),
        z=h["Z"].to_numpy(np.float64), tvt=h["TVT"].to_numpy(np.float64),
        tvt_input=h["TVT_input"].to_numpy(np.float64), ms=ms,
        x_med=float(h["X"].median()), y_med=float(h["Y"].median()),
        ancc_med=float(h["ANCC"].median()),
        anc_xy=anc[["X", "Y"]].to_numpy(np.float64), anc_a=anc["ANCC"].to_numpy(np.float64),
    )


def build_rowknn_ref(wells, idx):
    """Dense anchor-row reference (X,Y,ANCC) + per-row well id, from wells[idx]."""
    xy = np.concatenate([wells[j]["anc_xy"] for j in idx])
    a = np.concatenate([wells[j]["anc_a"] for j in idx])
    wid = np.concatenate([np.full(len(wells[j]["anc_a"]), j) for j in idx])
    scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
    return dict(xy=xy, a=a, wid=wid, scale=scale, tree=cKDTree(xy / scale),
               maxself=int(pd.Series(wid).value_counts().max()))


def rowknn_impute(ref, xy_q, self_code=None, k=ROW_K):
    """Production RowKNN IDW (src/spatial.py logic). self_code excludes that well's rows
    (LOO mode); None = test mode (block fully held -> no exclusion)."""
    if self_code is None:
        nq = min(k + 5, len(ref["a"]))
    else:
        nq = min(ref["maxself"] + k + 5, len(ref["a"]))
    dist, idx = ref["tree"].query(xy_q / ref["scale"], k=nq, workers=-1)
    if nq == 1:
        dist, idx = dist[:, None], idx[:, None]
    if self_code is not None:
        dist = np.where(ref["wid"][idx] == self_code, np.inf, dist)
    order = np.argpartition(dist, kth=min(k - 1, nq - 1), axis=1)[:, :k]
    d_k = np.take_along_axis(dist, order, 1)
    idx_k = np.take_along_axis(idx, order, 1)
    valid = np.isfinite(d_k)
    w = np.where(valid, 1.0 / (d_k + 1e-3), 0.0)
    sw = w.sum(1)
    no_n = sw < 1e-9
    safe = np.where(no_n, 1.0, sw)
    pred = (ref["a"][idx_k] * w).sum(1) / safe
    return np.where(no_n, ref["a"].mean(), pred)


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

    # ---------------- (a) LOO (degradation reference) ----------------
    print("\n>> (a) SINGLE-WELL LOO (stride-%d eval subsample for RowKNN/UK) ..." % LOO_EVAL_STRIDE, flush=True)
    kf_all = fit_gp_kernel(XYn, A)
    ymean_all = A.mean()
    cov1 = fit_uk_cov(XYn, A, 1)
    print(f"   UK1 cov: sill={cov1[0]:.3g} nugget={cov1[1]:.3g} ell={cov1[2]}", flush=True)
    ref_loo = build_rowknn_ref(wells, np.arange(n))   # full dense ref; LOO via self_code mask
    loo = {"GP": [], "UK1": [], "RowKNN": []}
    for j, d in enumerate(wells):
        ref = np.delete(np.arange(n), j)
        ev = slice(d["ms"], None)
        # subsample eval rows for the estimate (RowKNN LOO over-query is the expensive part)
        sl = np.arange(d["ms"], len(d["z"]))[::LOO_EVAL_STRIDE]
        def sc(pred_full):
            b = np.median(d["tvt_input"][:d["ms"]] + d["z"][:d["ms"]] - pred_full[:d["ms"]])
            return (-d["z"][sl] + pred_full[sl] + b) - d["tvt"][sl]
        loo["GP"].append(sc(gp_impute_well(d, XYn[ref], A[ref], kf_all, ymean_all, mu, sd)))
        loo["UK1"].append(sc(uk_impute_well(d, XYn[ref], A[ref], 1, cov1, mu, sd)))
        xyq = np.column_stack([d["x"], d["y"]])
        loo["RowKNN"].append(sc(rowknn_impute(ref_loo, xyq, self_code=j)))
        if (j + 1) % 200 == 0:
            print(f"   loo {j+1}/{n}", flush=True)
    loo = {k: np.concatenate(v) for k, v in loo.items()}
    print("  --- LOO ---", flush=True)
    for k in ("RowKNN", "GP", "UK1"):
        report(f"{k} LOO", loo[k])

    # ---------------- (b) spatial block-holdout ----------------
    print(f"\n>> (b) SPATIAL BLOCK-HOLDOUT (K={K_BLOCKS} KMeans blocks) ...", flush=True)
    labels = KMeans(n_clusters=K_BLOCKS, n_init=10, random_state=0).fit_predict(XYn)
    blk = {"GP": [], "UK1": [], "RowKNN": []}
    for b in range(K_BLOCKS):
        held = np.flatnonzero(labels == b)
        ref = np.flatnonzero(labels != b)
        kf_b = fit_gp_kernel(XYn[ref], A[ref])
        ymean_b = A[ref].mean()
        cov_b = fit_uk_cov(XYn[ref], A[ref], 1)
        rk_ref = build_rowknn_ref(wells, ref)   # dense ref from out-of-block wells only
        ge, ue, re = [], [], []
        for j in held:
            d = wells[j]
            ge.append(score_well(d, gp_impute_well(d, XYn[ref], A[ref], kf_b, ymean_b, mu, sd)))
            ue.append(score_well(d, uk_impute_well(d, XYn[ref], A[ref], 1, cov_b, mu, sd)))
            re.append(score_well(d, rowknn_impute(rk_ref, np.column_stack([d["x"], d["y"]]), self_code=None)))
        blk["GP"].append(np.concatenate(ge)); blk["UK1"].append(np.concatenate(ue)); blk["RowKNN"].append(np.concatenate(re))
        print(f"   block {b} (n={len(held)}): RowKNN={rmse(np.concatenate(re)):.2f}  "
              f"GP={rmse(np.concatenate(ge)):.2f}  UK1={rmse(np.concatenate(ue)):.2f}", flush=True)
    blk = {k: np.concatenate(v) for k, v in blk.items()}
    print("  --- BLOCK-HOLDOUT ---", flush=True)
    for k in ("RowKNN", "GP", "UK1"):
        report(f"{k} BLOCK", blk[k])

    # ---------------- pre-registered gate ----------------
    print("\n=== DEGRADATION (LOO -> BLOCK; ref = GP) ===", flush=True)
    deg = {k: rmse(blk[k]) - rmse(loo[k]) for k in blk}
    for k in ("RowKNN", "GP", "UK1"):
        print(f"  {k:7s}: LOO {rmse(loo[k]):7.2f} -> BLOCK {rmse(blk[k]):7.2f}  (deg {deg[k]:+.2f})", flush=True)

    R, U = rmse(blk["RowKNN"]), rmse(blk["UK1"])
    print("\n=== PRE-REGISTERED GATE VERDICT ===", flush=True)
    print(f"  production RowKNN block = {R:.2f}   UK1 block = {U:.2f}   (U/R = {U/R:.3f})", flush=True)
    print(f"  degradation: RowKNN {deg['RowKNN']:+.2f}   GP {deg['GP']:+.2f}   UK1 {deg['UK1']:+.2f}", flush=True)
    if U >= R:
        print(f"  >> KILL: UK1 does NOT beat the production RowKNN imputer OOD (U/R={U/R:.3f} >= 1). BET 5 dies.", flush=True)
    elif U <= 0.95 * R and deg["UK1"] <= deg["GP"]:
        print(f"  >> CLEAN PASS: UK1 beats production RowKNN by {(1-U/R)*100:.1f}% OOD and is less "
              f"density-coupled than GP -> Stage B (rebuild ANCC feats w/ UK1, retrain GBM, OOF-gate vs 9.30).", flush=True)
    else:
        why = []
        if not U <= 0.95 * R:
            why.append(f"margin {(1-U/R)*100:.1f}% (<5%)")
        if not deg["UK1"] <= deg["GP"]:
            why.append(f"degrades {deg['UK1']:+.2f} >= GP {deg['GP']:+.2f}")
        print(f"  >> AMBIGUOUS -> treat as NO ({'; '.join(why)}). Record null; do NOT touch the GBM.", flush=True)
    print("BET5 CORRECTED DONE", flush=True)


if __name__ == "__main__":
    main()
