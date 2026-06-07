"""Is the BANKED anchor (plane-KNN + row-KNN) also over-credited by single-well LOO?

The GP lever died because its LOO gate (24.50) was a density-favorable artifact: under
spatial BLOCK-holdout it degraded to 47.95 (~2x) and regressed on the LB. But the banked
11.903 stack's #1 feature `fk_tvt_formula` (plane-KNN, LOO 47.25) and workhorse
`rk_tvt_formula` (row-KNN, LOO 27.37) were ALSO gated on single-well LOO. If they degrade
the same way under block-holdout, the banked OOF is itself partly a mirage and the private
LB will undershoot -> the spatial-anchor family is tapped out, gap needs orthogonal signal.

Uses the EXACT kernel classes (FormationPlaneKNN, RowKNN) so this is faithful, not a reimpl.
Same 6 KMeans blocks as experiments/gp_block_holdout.py (seed 0) for comparability. For each
block: build the imputers from the REFERENCE wells only (held block excluded -> held wells
have NO in-block neighbor, the OOD condition), impute ANCC along held trajectories,
tvt_formula = -Z + ANCC + b_well (b_well from prefix), score on hidden rows.

Compare each estimator's block-holdout RMSE to its known single-well-LOO RMSE:
  plane-KNN  LOO 47.25   row-KNN  LOO 27.37   GP  LOO 24.50 -> block 47.95 (from gp_block_holdout)

Run: nohup python -u experiments/anchor_block_holdout.py > log/anchor_block_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
K_BLOCKS = 6

# import the EXACT kernel imputer classes (head only, before the inference main)
src = (ROOT / "jupyter_konbu/rogii_konbu_inference.py").read_text()
ns = {}
exec(src.split("# ---------------- Kaggle inference main")[0], ns)
FormationPlaneKNN = ns["FormationPlaneKNN"]
RowKNN = ns["RowKNN"]


def load_well(p):
    h = pd.read_csv(p)
    if "TVT_input" not in h.columns or "TVT" not in h.columns or "ANCC" not in h.columns:
        return None
    if not h["ANCC"].notna().any():
        return None
    mask = h["TVT_input"].isna().to_numpy()
    if not mask.any():
        return None
    ms = int(np.flatnonzero(mask)[0])
    if ms == 0:
        return None
    return dict(
        path=p, wid=Path(p).name.replace("__horizontal_well.csv", ""),
        x=h["X"].to_numpy(np.float64), y=h["Y"].to_numpy(np.float64),
        z=h["Z"].to_numpy(np.float64), tvt=h["TVT"].to_numpy(np.float64),
        tvt_input=h["TVT_input"].to_numpy(np.float64), ms=ms,
        x_med=float(h["X"].median()), y_med=float(h["Y"].median()),
    )


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def report(tag, err, loo):
    ae = np.abs(err)
    print(f"  {tag:18s} LOO={loo:6.2f} -> BLOCK RMSE={rmse(err):7.2f} "
          f"(degr {rmse(err)-loo:+6.2f})  median={np.median(ae):6.2f}  "
          f"p90={np.percentile(ae,90):6.2f}  p99={np.percentile(ae,99):7.2f}  max={ae.max():8.1f}", flush=True)


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
    mu, sd = XY.mean(0), XY.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    labels = KMeans(n_clusters=K_BLOCKS, n_init=10, random_state=0).fit_predict((XY - mu) / sd)
    for b in range(K_BLOCKS):
        print(f"   block {b}: {(labels==b).sum()} wells", flush=True)

    plane_err, row_err = [], []
    for b in range(K_BLOCKS):
        held = np.flatnonzero(labels == b)
        ref_paths = [Path(wells[j]["path"]) for j in np.flatnonzero(labels != b)]
        print(f"\n>> block {b}: building imputers on {len(ref_paths)} ref wells...", flush=True)
        plane = FormationPlaneKNN(ref_paths)
        row = RowKNN(ref_paths)
        pe, re = [], []
        for j in held:
            d = wells[j]
            xyq = np.column_stack([d["x"], d["y"]])
            formations, _ = plane.impute(xyq)            # ANCC = col 0
            pe.append(score_well(d, formations[:, 0]))
            row_ancc, _, _ = row.impute(xyq)             # no self filter: held well not in ref
            re.append(score_well(d, row_ancc))
        pe = np.concatenate(pe); re = np.concatenate(re)
        plane_err.append(pe); row_err.append(re)
        print(f"   block {b}: plane-KNN RMSE={rmse(pe):.2f}  row-KNN RMSE={rmse(re):.2f}", flush=True)

    plane_err = np.concatenate(plane_err); row_err = np.concatenate(row_err)
    print("\n=== ANCHOR BLOCK-HOLDOUT (held block has NO in-block neighbor) ===", flush=True)
    report("plane-KNN", plane_err, 47.25)
    report("row-KNN", row_err, 27.37)
    print("  (reference: GP  LOO 24.50 -> BLOCK 47.95, from gp_block_holdout.py)", flush=True)
    print("\n  Read: if plane/row-KNN degrade ~2x like GP -> banked 11.903 OOF is also density-inflated,", flush=True)
    print("  private LB will undershoot, spatial anchor is tapped out -> need orthogonal signal.", flush=True)
    print("  If they degrade gracefully (local extrapolation, not mean-reversion) -> banked is solid,", flush=True)
    print("  the GP failure was GP-specific.", flush=True)
    print("ANCHOR BLOCK HOLDOUT DONE", flush=True)


if __name__ == "__main__":
    main()
