"""Drift features for ROGII (see ../plan.md §5-6).

We predict **drift** = TVT - last_known_TVT for rows >= PS, then add the anchor
back at inference. This module builds one feature row per eval row (rows >= PS)
of every well, in two cached layers (base + spatial) merged on id.

Base feature families (Phase 1, ../plan.md §6 families 1-3):
  1. anchor/prefix  — per-well scalars from the known landing section [0, PS):
     anchor TVT, known-TVT slopes vs MD/Z (all + recent-200), spreads, prefix GR.
  2. position       — per-row geometry from PS: row index/fraction, md/x/y/z
     deltas, xy/xyz distance, local trajectory derivatives dz|dx|dy / dmd.
  3. GR             — interpolated GR (+ missing flag), rolling mean/std, diffs,
     lags/leads, GR vs prefix mean, GR vs typewell GR at the anchor TVT.

Spatial feature family (Phase 3, ../plan.md §6 family 5; imputers in src/spatial.py):
  5. offset-well geometry — formation tops imputed at the well's (X, Y) from
     neighboring train wells (plane-fit `fk_*` + dense row-KNN `rk_*`), turned
     into the TVT identity `-Z + top + b_well` and plane-vs-row agreement.

All feature columns are computable at TEST time (they use only MD,X,Y,Z,GR,
TVT_input + the typewell, plus the train-built imputers) — never the train-only
TVT or formation markers of the eval well itself.

Run (CPU, skynet) to build + cache both splits and layers:
    conda activate kaggle-arch && cd <project root>
    python -m src.features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import dataset as D
from . import spatial as S

GR_WINDOWS = (5, 21, 51, 101)
TRAJ_WIN = 15            # window for local trajectory derivatives
RECENT = 200             # "recent" prefix length for anchor-zone slope


def _fill_gr(gr: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN GR within the well (edges filled), + missing mask."""
    missing = gr.isna().to_numpy().astype(np.float32)
    filled = gr.interpolate(limit_direction="both").to_numpy()
    if np.isnan(filled).any():            # whole-column NaN fallback
        filled = np.nan_to_num(filled, nan=float(np.nanmean(gr)) if gr.notna().any() else 0.0)
    return filled.astype(np.float64), missing


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on x; 0 if degenerate."""
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def build_well_features(wid: str, hw: pd.DataFrame, tw: pd.DataFrame, split: str) -> pd.DataFrame:
    """Feature rows for the eval zone (rows >= PS) of one well."""
    ps = D.prediction_start_index(hw)
    n = len(hw)
    md = hw["MD"].to_numpy(); x = hw["X"].to_numpy()
    y = hw["Y"].to_numpy();  z = hw["Z"].to_numpy()
    gr, gr_missing = _fill_gr(hw["GR"])

    # ── family 1: anchor / prefix scalars (from known landing section) ──────────
    anchor = float(hw["TVT_input"].iloc[ps - 1])
    tvt_known = hw["TVT_input"].iloc[:ps].to_numpy()
    md_k, z_k = md[:ps], z[:ps]
    slope_md_all = _slope(md_k, tvt_known)
    slope_z_all = _slope(z_k, tvt_known)
    r0 = max(0, ps - RECENT)
    slope_md_recent = _slope(md_k[r0:], tvt_known[r0:])
    prefix_gr = gr[:ps]

    # ── family 3: GR rolling / diff / lag (over full well, then slice) ──────────
    grs = pd.Series(gr)
    feats: dict[str, np.ndarray] = {}
    for w in GR_WINDOWS:
        feats[f"gr_rmean_{w}"] = grs.rolling(w, min_periods=1, center=True).mean().to_numpy()
        feats[f"gr_rstd_{w}"] = grs.rolling(w, min_periods=1, center=True).std().fillna(0).to_numpy()
    feats["gr_diff1"] = grs.diff(1).fillna(0).to_numpy()
    feats["gr_diff2"] = grs.diff(2).fillna(0).to_numpy()
    for k in (5, 15, 30):
        feats[f"gr_lag_{k}"] = grs.shift(k).bfill().to_numpy()
        feats[f"gr_lead_{k}"] = grs.shift(-k).ffill().to_numpy()

    # ── family 2: local trajectory derivatives (rolling slope vs MD) ────────────
    def _dd(a: np.ndarray) -> np.ndarray:
        s = pd.Series(np.gradient(a, md))            # per-row derivative wrt MD
        return s.rolling(TRAJ_WIN, min_periods=1, center=True).mean().to_numpy()
    dz_dmd, dx_dmd, dy_dmd = _dd(z), _dd(x), _dd(y)

    # ── typewell GR at the anchor TVT (constant per well) ───────────────────────
    tw_tvt, tw_gr = tw["TVT"].to_numpy(), tw["GR"].to_numpy()
    ok = ~np.isnan(tw_gr)
    tw_gr_at_anchor = float(np.interp(anchor, tw_tvt[ok], tw_gr[ok])) if ok.any() else 0.0

    # ── assemble eval-zone rows ────────────────────────────────────────────────
    sl = slice(ps, n)
    eval_len = n - ps
    row_from_ps = np.arange(eval_len, dtype=np.float64)
    frac = row_from_ps / max(eval_len - 1, 1)
    out = pd.DataFrame({
        "id": [f"{wid}_{i}" for i in range(ps, n)],
        # anchor / prefix (broadcast scalars)
        "anchor_tvt": anchor,
        "ps_len": float(ps),
        "eval_len": float(eval_len),
        "tvt_known_std": float(tvt_known.std()),
        "tvt_known_range": float(np.ptp(tvt_known)),
        "slope_tvt_md_all": slope_md_all,
        "slope_tvt_md_recent": slope_md_recent,
        "slope_tvt_z_all": slope_z_all,
        "prefix_gr_mean": float(prefix_gr.mean()),
        "prefix_gr_std": float(prefix_gr.std()),
        "tw_gr_at_anchor": tw_gr_at_anchor,
        # position / trajectory
        "row_from_ps": row_from_ps,
        "row_frac": frac,
        "row_frac_sq": frac ** 2,
        "row_frac_sqrt": np.sqrt(frac),
        "md_from_ps": md[sl] - md[ps - 1],
        "x_from_ps": x[sl] - x[ps - 1],
        "y_from_ps": y[sl] - y[ps - 1],
        "z_from_ps": z[sl] - z[ps - 1],
        "xy_dist": np.hypot(x[sl] - x[ps - 1], y[sl] - y[ps - 1]),
        "xyz_dist": np.sqrt((x[sl] - x[ps - 1]) ** 2 + (y[sl] - y[ps - 1]) ** 2 + (z[sl] - z[ps - 1]) ** 2),
        "dz_dmd": dz_dmd[sl],
        "dx_dmd": dx_dmd[sl],
        "dy_dmd": dy_dmd[sl],
        # GR
        "gr": gr[sl],
        "gr_missing": gr_missing[sl],
        "gr_minus_prefix_mean": gr[sl] - float(prefix_gr.mean()),
        "gr_minus_tw_anchor": gr[sl] - tw_gr_at_anchor,
    })
    for name, arr in feats.items():
        out[name] = arr[sl]

    if split == "train":
        out["drift"] = hw["TVT"].iloc[sl].to_numpy() - anchor   # target
    return out


def build_base_features(split: str, cache: bool = True) -> pd.DataFrame:
    """Phase-1 base features (anchor/prefix/position/GR); cache to data/processed/."""
    path = C.PROC / f"features_base_{split}.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    parts = [build_well_features(wid, hw, tw, split) for wid, hw, tw in D.iter_wells(split)]
    mat = pd.concat(parts, ignore_index=True)
    if cache:
        C.PROC.mkdir(parents=True, exist_ok=True)
        mat.to_parquet(path, index=False)
    return mat


def _spatial_well_features(wid: str, hw: pd.DataFrame, ps: int,
                           fp: S.FormationPlaneKNN, rk: S.RowKNN,
                           self_wid: str | None) -> pd.DataFrame:
    """Phase-3 offset-well geometry features for one well's eval zone (rows >= PS).

    Imputes the 6 formation tops (plane-fit) and dense ANCC (row-KNN) at the
    well's (X, Y), then forms the TVT identity `-Z + top + b_well` per top, with
    `b_well` calibrated on the known landing section. `self_wid` excludes the
    query well from its own neighbor set (train LOO); pass None for test.
    """
    z = hw["Z"].to_numpy(float)
    xy = hw[["X", "Y"]].to_numpy(float)
    forms, mind = fp.impute(xy, self_wid=self_wid)
    r_ancc, r_std, r_dist = rk.impute(xy, self_wid=self_wid)

    kt = hw["TVT_input"].to_numpy()[:ps]          # known TVT on the landing section
    kz = z[:ps]
    last = float(hw["TVT_input"].iloc[ps - 1])
    b_fk = float(np.median(kt + kz - forms[:ps, 0]))   # ANCC plane bias
    b_rk = float(np.median(kt + kz - r_ancc[:ps]))     # ANCC row-KNN bias

    sl = slice(ps, len(hw))
    out = {"id": [f"{wid}_{i}" for i in range(ps, len(hw))],
           "fk_b_well": b_fk, "fk_min_dist": mind[sl], "fk_min_dist_log": np.log1p(mind[sl])}
    for fi, f in enumerate(S.FORMATIONS):
        out[f"fk_{f}"] = forms[sl, fi]
        out[f"fk_{f}_dz"] = z[sl] - forms[sl, fi]
    out["fk_tvt_formula"] = (-z[sl] + forms[sl, 0] + b_fk) - last   # drift, ANCC plane
    out["rk_ANCC"] = r_ancc[sl]
    out["rk_ANCC_dz"] = z[sl] - r_ancc[sl]
    out["rk_ANCC_std"] = r_std[sl]
    out["rk_dist"] = r_dist[sl]
    out["rk_b_well"] = b_rk
    out["rk_tvt_formula"] = (-z[sl] + r_ancc[sl] + b_rk) - last     # drift, ANCC row-KNN
    out["fk_vs_rk_ANCC"] = forms[sl, 0] - r_ancc[sl]                # plane-vs-row agreement
    return pd.DataFrame(out)


def build_spatial_features(split: str, cache: bool = True) -> pd.DataFrame:
    """Phase-3 spatial-imputation features keyed by id; cache to data/processed/.

    The two imputers are built ONCE from train wells. Train uses leave-one-well-out
    (self_wid=wid); test uses the full train reference (self_wid=None) to mirror how
    the hidden test is scored.
    """
    path = C.PROC / f"spatial_{split}.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    fp = S.FormationPlaneKNN()
    rk = S.RowKNN()
    parts = []
    for wid, hw, _ in D.iter_wells(split):
        ps = D.prediction_start_index(hw)
        if ps == 0 or ps >= len(hw):
            continue
        self_wid = wid if split == "train" else None
        parts.append(_spatial_well_features(wid, hw, ps, fp, rk, self_wid))
    mat = pd.concat(parts, ignore_index=True)
    if cache:
        C.PROC.mkdir(parents=True, exist_ok=True)
        mat.to_parquet(path, index=False)
    return mat


def build_feature_matrix(split: str, cache: bool = True) -> pd.DataFrame:
    """Full Phase-3 matrix: base features left-joined with spatial features on id."""
    base = build_base_features(split, cache=cache)
    spatial = build_spatial_features(split, cache=cache)
    return base.merge(spatial, on="id", how="left")


def feature_columns(mat: pd.DataFrame) -> list[str]:
    """Model input columns (everything except id and the target)."""
    return [c for c in mat.columns if c not in ("id", "drift")]


def main() -> None:
    for split in ("train", "test"):
        base = build_base_features(split, cache=True)
        spatial = build_spatial_features(split, cache=True)
        mat = base.merge(spatial, on="id", how="left")
        n_missing = int(mat[spatial.columns[1]].isna().sum())   # any unmatched spatial rows
        print(f"[features] {split}: {mat.shape[0]:,} rows x {mat.shape[1]} cols "
              f"(base {base.shape[1]} + spatial {spatial.shape[1] - 1}); "
              f"{n_missing} rows missing spatial features")
        if split == "train":
            print(f"[features] {len(feature_columns(mat))} model features: {feature_columns(mat)}")


if __name__ == "__main__":
    main()
