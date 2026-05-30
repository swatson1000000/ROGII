"""Phase-3 offset-well geometry: spatial imputation of formation tops.

The exact within-well identity (EDA-verified, r=1.0000, resid std ~0.0065 ft) is

    TVT = -Z + ANCC + b_well

where ANCC (and the other formation tops) is a smooth surface in (X, Y) but is
**train-only**. The lever is therefore to *impute* the formation top at the eval
well's (X, Y) from neighboring training wells, then recover TVT via the identity
with a per-well bias `b_well` calibrated on the known landing section.

Two imputers (both built from TRAIN wells only):

  FormationPlaneKNN — for each of the 6 tops, fit a distance-weighted 2-D plane
    through the K nearest *well centroids* and evaluate at the query (X, Y).
    Coarse (one value per well) but well-conditioned.

  RowKNN — dense IDW over every anchor row's ANCC across all train wells.
    Finer spatial resolution; the dominant Phase-3 feature by GBM gain.

Two gotchas, both learned the hard way during probing (see PICK_UP_HERE.md):

  * **Plane fit is in RAW (X, Y) through well CENTROIDS.** Fitting a local plane
    through dense per-row points blows up — the points are near-collinear along
    one trajectory, so the intercept extrapolates to garbage (saw 2107 ft).
  * **RowKNN LOO buffer.** A well's own rows are its nearest neighbors, so to
    exclude self we must query k >= (max per-well row count) + K before masking,
    else we leak or run short of neighbors.

LOO policy: for the TRAIN split, exclude the query well from its own neighbor set
(`self_wid=wid`). For TEST, use the full train reference with no exclusion — at
real inference the hidden test wells are genuinely absent from the train
reference, so this mirrors scoring. (For the 3 visible test wells, which are
blanked train wells, this is a mild optimism, consistent with predict.py's
already-flagged optimistic sanity check.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config as C
from . import dataset as D

FORMATIONS = C.FORMATION_MARKERS          # ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10
ROW_K = 20
ROW_STRIDE = 3                            # subsample dense ANCC rows (stride-3 = ~1.68M pts)
ROW_NQ = 400                              # min over-query before LOO masking


class FormationPlaneKNN:
    """Distance-weighted 2-D plane fit through the K nearest train-well centroids,
    one plane per formation top, evaluated at a query (X, Y)."""

    def __init__(self) -> None:
        rows = []
        for wid in C.train_well_ids():
            df = D.load_horizontal(wid, "train")
            if not set(["X", "Y", *FORMATIONS]).issubset(df.columns):
                continue
            d = df[["X", "Y", *FORMATIONS]].dropna()
            if len(d) == 0:
                continue
            r = {"wid": wid, "x": d["X"].median(), "y": d["Y"].median()}
            for c in FORMATIONS:
                r[f"{c}_med"] = d[c].median()
            rows.append(r)
        self.df = pd.DataFrame(rows)
        self.wid_idx = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy(float)
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))   # KD-tree scaling only
        self.tree = cKDTree(xy / self.scale)
        self.x = self.df["x"].to_numpy()
        self.y = self.df["y"].to_numpy()
        self.farr = self.df[[f"{c}_med" for c in FORMATIONS]].to_numpy(float)

    def impute(self, xy_q: np.ndarray, self_wid: str | None = None, k: int = PLANE_K):
        """Return (pred[n, 6] formation tops, min_neighbor_dist[n]) at queries xy_q."""
        q = xy_q / self.scale
        nq = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nq)
        if self_wid in self.wid_idx:
            dist = np.where(idx == self.wid_idx[self_wid], np.inf, dist)
        order = np.argpartition(dist, kth=min(k - 1, nq - 1), axis=1)[:, :k]
        d_k = np.take_along_axis(dist, order, 1)
        idx_k = np.take_along_axis(idx, order, 1)
        valid = np.isfinite(d_k)
        w = np.where(valid, 1.0 / (d_k + 1e-3), 0.0)
        xn = self.x[idx_k]
        yn = self.y[idx_k]
        wx, wy = w * xn, w * yn
        # weighted least squares for plane coefficients (a*x + b*y + c) per query
        ATWA = np.zeros((len(xy_q), 3, 3))
        ATWA[:, 0, 0] = (wx * xn).sum(1)
        ATWA[:, 0, 1] = ATWA[:, 1, 0] = (wx * yn).sum(1)
        ATWA[:, 0, 2] = ATWA[:, 2, 0] = wx.sum(1)
        ATWA[:, 1, 1] = (wy * yn).sum(1)
        ATWA[:, 1, 2] = ATWA[:, 2, 1] = wy.sum(1)
        ATWA[:, 2, 2] = w.sum(1)
        ATWA[:, [0, 1, 2], [0, 1, 2]] += 1e-9
        fn = self.farr[idx_k]
        rhs = np.stack([(wx[:, :, None] * fn).sum(1),
                        (wy[:, :, None] * fn).sum(1),
                        (w[:, :, None] * fn).sum(1)], axis=1)
        try:
            coef = np.linalg.solve(ATWA, rhs)
        except np.linalg.LinAlgError:
            coef = np.stack([np.linalg.pinv(ATWA[r]) @ rhs[r] for r in range(len(xy_q))])
        pred = (xy_q[:, 0:1] * coef[:, 0, :] + xy_q[:, 1:2] * coef[:, 1, :] + coef[:, 2, :])
        no_n = (~valid).all(1)
        if no_n.any():
            pred[no_n] = self.farr.mean(0)
        return pred, np.where(valid, d_k, np.inf).min(1)


class RowKNN:
    """Dense inverse-distance KNN over every anchor row's ANCC across train wells."""

    def __init__(self) -> None:
        xs, ys, a, wid_arr = [], [], [], []
        for wid in C.train_well_ids():
            df = D.load_horizontal(wid, "train")
            if not set(["X", "Y", "ANCC"]).issubset(df.columns):
                continue
            d = df[["X", "Y", "ANCC"]].dropna().iloc[::ROW_STRIDE]
            if len(d) == 0:
                continue
            xs.append(d["X"].to_numpy())
            ys.append(d["Y"].to_numpy())
            a.append(d["ANCC"].to_numpy())
            wid_arr += [wid] * len(d)
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(a).astype(np.float64)
        self.wids = np.array(wid_arr)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)
        self.maxself = int(pd.Series(self.wids).value_counts().max())

    def impute(self, xy_q: np.ndarray, self_wid: str | None = None, k: int = ROW_K):
        """Return (ANCC[n], ANCC_std[n], min_neighbor_dist[n]) at queries xy_q.

        Over-queries to maxself+k+5 so that after masking the query well's own
        rows (which are its nearest neighbors) at least k true neighbors remain.
        """
        nq = min(max(ROW_NQ, self.maxself + k + 5), len(self.ancc))
        dist, idx = self.tree.query(xy_q / self.scale, k=nq, workers=-1)
        if self_wid is not None:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        order = np.argpartition(dist, kth=min(k - 1, nq - 1), axis=1)[:, :k]
        d_k = np.take_along_axis(dist, order, 1)
        idx_k = np.take_along_axis(idx, order, 1)
        valid = np.isfinite(d_k)
        w = np.where(valid, 1.0 / (d_k + 1e-3), 0.0)
        sw = w.sum(1)
        no_n = sw < 1e-9
        safe = np.where(no_n, 1.0, sw)
        pred = (self.ancc[idx_k] * w).sum(1) / safe
        pred = np.where(no_n, self.ancc.mean(), pred)
        var = (((self.ancc[idx_k] - pred[:, None]) ** 2) * w).sum(1) / safe
        return pred, np.sqrt(np.maximum(var, 0.0)), np.where(valid, d_k, np.inf).min(1)
