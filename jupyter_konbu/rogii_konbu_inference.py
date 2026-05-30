"""ROGII Wellbore Geology Prediction
Public Score 11.912 — Plane-fit Formation Top + Row-level K-NN ANCC + LGB×3 + XGB Ridge stack

# Key insight (EDA-validated)
The relationship between TVT and the geological surface ANCC is nearly
**perfectly linear** within each well:

    TVT = -1.0 * (Z - ANCC) + b_well     (Pearson = -1.0000, resid_std ~ 0.007 ft)

If we can predict ANCC at any (X, Y) accurately, TVT is essentially
determined modulo a per-well constant b_well that we estimate from
the known prefix.

# Pipeline
1. Per-well centroid K-NN with weighted 2D plane fit for the 6
   formation tops (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA).
   Median per-formation imputation RMSE ~ 17 ft (vs 47 ft for IDW).

2. Row-level (X, Y) K-NN of ANCC over the full ~5M training rows,
   self-well filtered, providing a finer complement.

3. ~80 leakage-aware features per post-PS row: position/distance,
   GR rolling stats, GR lag/lead, typewell offset diffs, beam-search
   typewell path, prefix typewell-RMSE quality, plus the formation-top
   imputation outputs and a closed-form geological prediction
   tvt_formula = -Z + ANCC + b_well_prefix.

4. GroupKFold(5) ensemble: LightGBM x 3 seeds + XGBoost (GPU), then
   Ridge stacking with non-negative weights chosen on OOF.

5. Train target = TVT - last_known_TVT; final test prediction adds the
   shift back to the anchor.

# Results
- Local OOF RMSE: 12.11
- Public LB: 11.912
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


ROOT = Path("/home/swatson/work/kaggle/ROGII/data/raw")
OUTPUT = Path("/tmp/konbu_submission.csv")
N_SPLITS = 5
SPLIT_SEED = 42
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10
ROW_K = 20
ROW_NQ = 12_000
ROW_QUERY_WORKERS = 1   # 1 when parallelizing across wells (avoid oversubscription)

LGB_PARAMS = dict(
    boosting_type="gbdt",
    learning_rate=0.06,
    num_leaves=89,
    min_child_samples=10,
    min_child_weight=0.5,
    n_estimators=5000,
    n_jobs=-1,
    reg_alpha=2.03,
    reg_lambda=87.28,
    subsample=0.645,
    subsample_freq=1,
    colsample_bytree=0.821,
    objective="regression",
    metric="rmse",
    verbose=-1,

    device_type="cuda",   # GPU LGB built on skynet (sm_121); ~6.7x faster, ~ same RMSE
    max_bin=255,
)
XGB_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    learning_rate=0.06,
    max_depth=8,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.85,
    reg_alpha=1.0,
    reg_lambda=20.0,
    tree_method="hist",
    device="cuda",   # GB10 GPU works for XGB (LGB has no GPU build here)
    n_jobs=-1,
)
LGB_SEEDS = [42, 7, 123]


def recent_mean_diff(values, window):
    values = values[-(window + 1):]
    if len(values) < 2:
        return 0.0
    return float(np.diff(values).mean())


def recent_slope(y, x, window):
    y = y[-window:]; x = x[-window:]
    if len(y) < 2:
        return 0.0
    cx = x - x.mean()
    d = float(np.dot(cx, cx))
    return 0.0 if d == 0.0 else float(np.dot(cx, y - y.mean()) / d)


def nearest_index(sorted_values, target):
    idx = int(np.searchsorted(sorted_values, target, side="left"))
    if idx >= len(sorted_values):
        return len(sorted_values) - 1
    if idx > 0 and abs(sorted_values[idx - 1] - target) <= abs(sorted_values[idx] - target):
        return idx - 1
    return idx


def fill_and_smooth_gr(values, fallback, radius):
    s = pd.Series(values, dtype="float32").interpolate(limit_direction="both").fillna(fallback)
    if radius <= 0:
        return s.to_numpy(dtype=np.float32)
    return s.rolling(radius * 2 + 1, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)


def beam_predict(gr_values, tw_tvt, tw_gr, start_tvt, beam_size, move_cost, emit_scale, radius):
    start_idx = nearest_index(tw_tvt, start_tvt)
    smoothed = fill_and_smooth_gr(gr_values, float(np.nanmean(tw_gr)), radius)
    states = {start_idx: 0.0}
    backpointers = []
    for gr_value in smoothed:
        candidates, parents = {}, {}
        for idx, cost in states.items():
            for delta in (-1, 0, 1):
                ni = idx + delta
                if ni < 0 or ni >= len(tw_tvt):
                    continue
                emit = ((gr_value - tw_gr[ni]) ** 2) / emit_scale
                tot = cost + emit + move_cost * abs(delta)
                prev = candidates.get(ni)
                if prev is None or tot < prev:
                    candidates[ni] = tot
                    parents[ni] = idx
        kept = sorted(candidates.items(), key=lambda kv: kv[1])[:beam_size]
        states = {idx: cost for idx, cost in kept}
        backpointers.append({idx: parents[idx] for idx, _ in kept})
    if not states:
        return np.full(len(smoothed), tw_tvt[start_idx], dtype=np.float32)
    final_idx = min(states, key=states.get)
    path = [final_idx]
    for step in range(len(backpointers) - 1, 0, -1):
        path.append(backpointers[step][path[-1]])
    path.reverse()
    return tw_tvt[np.asarray(path, dtype=np.int32)]


def gr_fft_features(gr_post):
    valid = gr_post[~np.isnan(gr_post)]
    if len(valid) < 32:
        return 0.0, 0.0
    centered = valid - valid.mean()
    spec = np.abs(np.fft.rfft(centered)) ** 2
    if len(spec) < 3:
        return 0.0, 0.0
    dom = int(np.argmax(spec[1:])) + 1
    return float(dom / len(valid)), float(np.log1p(spec[dom]))


# ---------------- NEW: centroid-based plane fit imputer ----------------

class FormationPlaneKNN:
    """K=10 nearest non-self centroids, weighted 2D plane fit per row."""

    def __init__(self, train_paths):
        rows = []
        for p in train_paths:
            wid = p.stem.replace("__horizontal_well", "")
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_med"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wid_idx = {w: i for i, w in enumerate(self.df["wid"].to_numpy())}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = xy.std(axis=0)
        self.scale = np.where(self.scale < 1e-3, 1.0, self.scale)
        self.tree = cKDTree(xy / self.scale)
        self.x_arr = self.df["x"].to_numpy()
        self.y_arr = self.df["y"].to_numpy()
        self.formation_arr = self.df[[f"{c}_med" for c in FORMATIONS]].to_numpy(dtype=np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale
        n_q = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=n_q)
        if self_wid is not None and self_wid in self.wid_idx:
            self_i = self.wid_idx[self_wid]
            mask_self = idx == self_i
            dist = np.where(mask_self, np.inf, dist)
        order = np.argpartition(dist, kth=min(k - 1, n_q - 1), axis=1)[:, :k]
        d_k = np.take_along_axis(dist, order, axis=1)
        idx_k = np.take_along_axis(idx, order, axis=1)
        valid_k = np.isfinite(d_k)
        w = np.where(valid_k, 1.0 / (d_k + 1e-3), 0.0).astype(np.float64)  # (N, K)
        # neighbour positions
        x_n = self.x_arr[idx_k]  # (N, K)
        y_n = self.y_arr[idx_k]  # (N, K)
        # build weighted normal equations: A.T W A coef = A.T W b
        # A[r, i, :] = [x_n[r,i], y_n[r,i], 1]; b[r, i, fi] = formation_arr[idx_k[r,i], fi]
        # AᵀWA[r] = sum_i w[r,i] * [[x², xy, x], [xy, y², y], [x, y, 1]]
        wx = w * x_n
        wy = w * y_n
        ATWA_xx = (wx * x_n).sum(axis=1)
        ATWA_xy = (wx * y_n).sum(axis=1)
        ATWA_xc = wx.sum(axis=1)
        ATWA_yy = (wy * y_n).sum(axis=1)
        ATWA_yc = wy.sum(axis=1)
        ATWA_cc = w.sum(axis=1)
        # 3x3 matrix per row
        ATWA = np.zeros((len(xy_q), 3, 3))
        ATWA[:, 0, 0] = ATWA_xx
        ATWA[:, 0, 1] = ATWA_xy
        ATWA[:, 0, 2] = ATWA_xc
        ATWA[:, 1, 0] = ATWA_xy
        ATWA[:, 1, 1] = ATWA_yy
        ATWA[:, 1, 2] = ATWA_yc
        ATWA[:, 2, 0] = ATWA_xc
        ATWA[:, 2, 1] = ATWA_yc
        ATWA[:, 2, 2] = ATWA_cc
        # add tiny ridge to guard against singularity
        ATWA[:, 0, 0] += 1e-9
        ATWA[:, 1, 1] += 1e-9
        ATWA[:, 2, 2] += 1e-9

        # Build ATWb for each formation: (N, K) → (N, 3, 6)
        # ATWb[r, :, fi] = sum_i w[r,i] * [x_n[r,i], y_n[r,i], 1] * formation[idx_k[r,i], fi]
        f_n = self.formation_arr[idx_k]  # (N, K, 6)
        ATWb_x = ((wx[:, :, None] * f_n).sum(axis=1))  # (N, 6)
        ATWb_y = ((wy[:, :, None] * f_n).sum(axis=1))  # (N, 6)
        ATWb_c = ((w[:, :, None] * f_n).sum(axis=1))   # (N, 6)
        # solve: coef[r, :, fi] = ATWA[r]^-1 @ [ATWb_x[r,fi], ATWb_y[r,fi], ATWb_c[r,fi]]
        # vectorize via batched solve
        # build rhs: (N, 3, 6)
        rhs = np.stack([ATWb_x, ATWb_y, ATWb_c], axis=1)  # (N, 3, 6)
        try:
            coef = np.linalg.solve(ATWA, rhs)  # (N, 3, 6)
        except np.linalg.LinAlgError:
            # fallback: pseudo-inverse per row
            coef = np.zeros((len(xy_q), 3, 6))
            for r in range(len(xy_q)):
                try:
                    coef[r] = np.linalg.pinv(ATWA[r]) @ rhs[r]
                except Exception:
                    coef[r] = 0
        # predict at query positions: pred[r, fi] = X_q[r]*coef[r,0,fi] + Y_q[r]*coef[r,1,fi] + coef[r,2,fi]
        X_q = xy_q[:, 0]
        Y_q = xy_q[:, 1]
        formations = (X_q[:, None] * coef[:, 0, :]
                      + Y_q[:, None] * coef[:, 1, :]
                      + coef[:, 2, :]).astype(np.float32)
        # uncertainty / fall-back where K-NN had no valid neighbours
        no_n = (~valid_k).all(axis=1)
        if no_n.any():
            global_mean = self.formation_arr.mean(axis=0)
            formations[no_n] = global_mean
        d_finite = np.where(valid_k, d_k, np.inf)
        min_dist = d_finite.min(axis=1).astype(np.float32)
        return formations, min_dist


# ---------------- row-level (X, Y) ANCC K-NN (from exp011) ----------------

class RowKNN:
    def __init__(self, train_paths):
        xs, ys, anccs, wid_arr = [], [], [], []
        for p in train_paths:
            wid = p.stem.replace("__horizontal_well", "")
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            xs.append(df["X"].to_numpy())
            ys.append(df["Y"].to_numpy())
            anccs.append(df["ANCC"].to_numpy())
            wid_arr.extend([wid] * len(df))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wid_arr)
        self.scale = self.xy.std(axis=0)
        self.scale = np.where(self.scale < 1e-3, 1.0, self.scale)
        self.tree = cKDTree(self.xy / self.scale)
        self.maxself = int(pd.Series(self.wids).value_counts().max())

    def impute(self, xy_q, self_wid=None, k=ROW_K, n_q=ROW_NQ):
        q = xy_q / self.scale
        # Adaptive over-query: enough to survive LOO self-masking, then capped.
        # Faithful to konbu's intent (same true neighbors) but avoids k=12000.
        n_q = min(max(self.maxself + k + 50, k + 50), len(self.ancc))
        dist, idx = self.tree.query(q, k=n_q, workers=ROW_QUERY_WORKERS)
        if self_wid is not None:
            mask_self = self.wids[idx] == self_wid
            dist = np.where(mask_self, np.inf, dist)
        order = np.argpartition(dist, kth=min(k - 1, n_q - 1), axis=1)[:, :k]
        d_k = np.take_along_axis(dist, order, axis=1)
        idx_k = np.take_along_axis(idx, order, axis=1)
        valid_k = np.isfinite(d_k)
        w = np.where(valid_k, 1.0 / (d_k + 1e-3), 0.0)
        sw = w.sum(axis=1)
        no_n = sw < 1e-9
        safe = np.where(no_n, 1.0, sw)
        ancc_pred = (self.ancc[idx_k] * w).sum(axis=1) / safe
        ancc_pred = np.where(no_n, float(self.ancc.mean()), ancc_pred)
        diff_sq = (self.ancc[idx_k] - ancc_pred[:, None]) ** 2
        var = (diff_sq * w).sum(axis=1) / safe
        std = np.sqrt(np.maximum(var, 0.0))
        d_finite = np.where(valid_k, d_k, np.inf)
        min_dist = d_finite.min(axis=1)
        return (ancc_pred.astype(np.float32),
                std.astype(np.float32),
                min_dist.astype(np.float32))


# ---------------- per-row feature builder ----------------

def build_hidden_features(h, t, wid, is_train, formation_imputer, row_imputer):
    mask = h["TVT_input"].isna().to_numpy()
    if not mask.any():
        return None
    mask_start = int(np.flatnonzero(mask)[0])
    if mask_start == 0:
        return None
    known = h.iloc[:mask_start].copy()
    hidden = h.iloc[mask_start:].copy()
    last_known = known.iloc[-1]

    tw_tvt = t["TVT"].to_numpy(dtype=np.float32)
    tw_gr = t["GR"].to_numpy(dtype=np.float32)

    gr_full = h["GR"].interpolate(limit_direction="both")
    if gr_full.isna().any():
        gr_full = gr_full.fillna(float(np.nanmean(tw_gr)))

    gr_roll5 = gr_full.rolling(5, center=True, min_periods=1).mean()
    gr_roll21 = gr_full.rolling(21, center=True, min_periods=1).mean()
    gr_grad = gr_full.diff().fillna(0.0)
    gr_std5 = gr_full.rolling(5, center=True, min_periods=1).std().fillna(0.0)
    gr_std21 = gr_full.rolling(21, center=True, min_periods=1).std().fillna(0.0)
    gr_lag1 = gr_full.shift(1).bfill()
    gr_lead1 = gr_full.shift(-1).ffill()
    gr_lag5 = gr_full.shift(5).bfill()
    gr_lead5 = gr_full.shift(-5).ffill()
    gr_cumsum = gr_full.cumsum()

    known_tvt = known["TVT_input"].to_numpy(dtype=np.float32)
    known_md = known["MD"].to_numpy(dtype=np.float32)
    known_z = known["Z"].to_numpy(dtype=np.float32)

    prefix_tw_gr = np.interp(known_tvt, tw_tvt, tw_gr)
    prefix_gr = gr_full.iloc[:mask_start].to_numpy(dtype=np.float32)
    prefix_residual = prefix_gr - prefix_tw_gr
    prefix_tw_rmse = float(np.sqrt(np.mean(prefix_residual ** 2)))
    prefix_tw_mae = float(np.mean(np.abs(prefix_residual)))

    last_known_tvt = float(last_known["TVT_input"])
    hidden_gr = hidden["GR"].to_numpy(dtype=np.float32)

    beam_cons = beam_predict(hidden_gr, tw_tvt, tw_gr, last_known_tvt, 10, 20.0, 144.0, 2)
    beam_loose = beam_predict(hidden_gr, tw_tvt, tw_gr, last_known_tvt, 10, 8.0, 64.0, 2)

    hidden_gr_filled = gr_full.iloc[mask_start:].to_numpy(dtype=np.float32)
    offsets = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], dtype=np.float32)
    offset_diffs = {
        f"tw_diff_{int(off)}": hidden_gr_filled
        - np.float32(np.interp(last_known_tvt + float(off), tw_tvt, tw_gr))
        for off in offsets
    }

    # Formation top imputation: PLANE FIT (NEW for exp012)
    xy_full = h[["X", "Y"]].to_numpy(dtype=np.float64)
    self_wid_for_train = wid if is_train else None
    formations_full, knn_min_dist_full = formation_imputer.impute(
        xy_full, self_wid=self_wid_for_train)
    formations_post = formations_full[mask_start:]
    knn_min_dist_post = knn_min_dist_full[mask_start:]
    z_full = h["Z"].to_numpy(dtype=np.float32)
    pred_ancc_centroid = formations_full[:, 0]
    if mask_start > 0:
        b_per_row = known_tvt + known_z - pred_ancc_centroid[:mask_start]
        b_well_centroid = float(np.median(b_per_row))
    else:
        b_well_centroid = 0.0
    z_post = hidden["Z"].to_numpy(dtype=np.float32)
    tvt_formula_pred_centroid = -z_post + formations_post[:, 0] + b_well_centroid

    # Row-level (X, Y) ANCC K-NN (from exp011)
    row_ancc_full, row_ancc_std_full, row_min_dist_full = row_imputer.impute(
        xy_full, self_wid=self_wid_for_train)
    row_ancc_post = row_ancc_full[mask_start:]
    row_ancc_std_post = row_ancc_std_full[mask_start:]
    row_min_dist_post = row_min_dist_full[mask_start:]
    if mask_start > 0:
        b_per_row_row = known_tvt + known_z - row_ancc_full[:mask_start]
        b_well_row = float(np.median(b_per_row_row))
    else:
        b_well_row = 0.0
    tvt_formula_pred_row = -z_post + row_ancc_post + b_well_row

    feats = pd.DataFrame({
        "well": wid,
        "prediction_id": [f"{wid}_{i}" for i in hidden.index],
        "row_idx": hidden.index.to_numpy(dtype=np.int32),
        "last_known_tvt": np.float32(last_known_tvt),
        "known_len": np.int32(mask_start),
        "hidden_len": np.int32(len(hidden)),
        "frac_hidden": ((hidden.index - mask_start) / max(len(hidden) - 1, 1)).astype(np.float32),
        "md": hidden["MD"].to_numpy(dtype=np.float32),
        "z": hidden["Z"].to_numpy(dtype=np.float32),
        "x": hidden["X"].to_numpy(dtype=np.float32),
        "y": hidden["Y"].to_numpy(dtype=np.float32),
        "gr": hidden_gr_filled,
        "gr_missing": hidden["GR"].isna().to_numpy(dtype=np.int8),
        "gr_roll5": gr_roll5.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_roll21": gr_roll21.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_grad": gr_grad.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_std5": gr_std5.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_std21": gr_std21.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_lag1": gr_lag1.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_lead1": gr_lead1.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_lag5": gr_lag5.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_lead5": gr_lead5.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_cumsum": (gr_cumsum.iloc[mask_start:] - gr_cumsum.iloc[mask_start - 1]).to_numpy(dtype=np.float32),
        "dmd": (hidden["MD"] - float(last_known["MD"])).to_numpy(dtype=np.float32),
        "dz": (hidden["Z"] - float(last_known["Z"])).to_numpy(dtype=np.float32),
        "dx": (hidden["X"] - float(last_known["X"])).to_numpy(dtype=np.float32),
        "dy": (hidden["Y"] - float(last_known["Y"])).to_numpy(dtype=np.float32),
        "dx_dmd": ((hidden["X"] - float(last_known["X"]))
                   / np.maximum(hidden["MD"] - float(last_known["MD"]), 1e-5)).to_numpy(dtype=np.float32),
        "dy_dmd": ((hidden["Y"] - float(last_known["Y"]))
                   / np.maximum(hidden["MD"] - float(last_known["MD"]), 1e-5)).to_numpy(dtype=np.float32),
        "dz_dmd": ((hidden["Z"] - float(last_known["Z"]))
                   / np.maximum(hidden["MD"] - float(last_known["MD"]), 1e-5)).to_numpy(dtype=np.float32),
        "dist_xy": np.sqrt((hidden["X"] - float(last_known["X"])) ** 2
                           + (hidden["Y"] - float(last_known["Y"])) ** 2).to_numpy(dtype=np.float32),
        "dist_xyz": np.sqrt((hidden["X"] - float(last_known["X"])) ** 2
                            + (hidden["Y"] - float(last_known["Y"])) ** 2
                            + (hidden["Z"] - float(last_known["Z"])) ** 2).to_numpy(dtype=np.float32),
        "prefix_tvt_step20": np.float32(recent_mean_diff(known_tvt, 20)),
        "prefix_tvt_step100": np.float32(recent_mean_diff(known_tvt, 100)),
        "prefix_tvt_md_slope100": np.float32(recent_slope(known_tvt, known_md, 100)),
        "prefix_tvt_z_slope100": np.float32(recent_slope(known_tvt, known_z, 100)),
        "prefix_tw_rmse": np.float32(prefix_tw_rmse),
        "prefix_tw_mae": np.float32(prefix_tw_mae),
        "beam_cons_delta": (beam_cons - np.float32(last_known_tvt)).astype(np.float32),
        "beam_loose_delta": (beam_loose - np.float32(last_known_tvt)).astype(np.float32),
        "beam_gap": (beam_loose - beam_cons).astype(np.float32),
    })
    for name, vals in offset_diffs.items():
        feats[name] = vals.astype(np.float32)

    slc = (tw_tvt >= last_known_tvt - 40.0) & (tw_tvt <= last_known_tvt + 40.0)
    if slc.sum() >= 5 and (~np.isnan(hidden_gr)).any():
        gr_ok = hidden_gr[~np.isnan(hidden_gr)]
        tvt_s, gr_s = tw_tvt[slc], tw_gr[slc]
        d = np.abs(gr_ok[:, None] - gr_s[None, :])
        nn = np.argmin(d, axis=1)
        matched = tvt_s[nn]
        feats["ncc_med_shift_well"] = np.float32(np.median(matched) - last_known_tvt)
        feats["ncc_mean_shift_well"] = np.float32(np.mean(matched) - last_known_tvt)
    else:
        feats["ncc_med_shift_well"] = np.float32(0.0)
        feats["ncc_mean_shift_well"] = np.float32(0.0)

    fft_freq, fft_pow = gr_fft_features(hidden_gr)
    feats["gr_fft_dom_freq"] = np.float32(fft_freq)
    feats["gr_fft_dom_power"] = np.float32(fft_pow)

    if len(tw_tvt):
        tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())
        feats["anchor_t_pos"] = np.float32((last_known_tvt - tmin) / max(tmax - tmin, 1e-3))
    else:
        feats["anchor_t_pos"] = np.float32(0.0)
    feats["spatial_knn_delta"] = np.float32(0.0)

    # Formation features (now plane-fit imputed)
    for fi, fname in enumerate(FORMATIONS):
        feats[f"fk_{fname}"] = formations_post[:, fi].astype(np.float32)
        feats[f"fk_{fname}_dz"] = (z_post - formations_post[:, fi]).astype(np.float32)
    feats["fk_b_well"] = np.float32(b_well_centroid)
    feats["fk_min_dist"] = knn_min_dist_post.astype(np.float32)
    feats["fk_tvt_formula"] = (tvt_formula_pred_centroid - np.float32(last_known_tvt)).astype(np.float32)

    # Row-level (X, Y) features (from exp011)
    feats["knn_row_ANCC"] = row_ancc_post.astype(np.float32)
    feats["knn_row_ANCC_dz"] = (z_post - row_ancc_post).astype(np.float32)
    feats["knn_row_ANCC_std"] = row_ancc_std_post.astype(np.float32)
    feats["knn_row_dist"] = row_min_dist_post.astype(np.float32)
    feats["knn_row_b_well"] = np.float32(b_well_row)
    feats["knn_row_tvt_pred_delta"] = (tvt_formula_pred_row - np.float32(last_known_tvt)).astype(np.float32)
    feats["fk_vs_row_ANCC_diff"] = (formations_post[:, 0] - row_ancc_post).astype(np.float32)

    if is_train:
        feats["target"] = (hidden["TVT"].to_numpy(dtype=np.float32)
                           - np.float32(last_known_tvt)).astype(np.float32)
    return feats


def build_dataset(paths, formation_imputer, row_imputer, is_train, label):
    parts = []
    for i, p in enumerate(paths):
        wid = p.stem.replace("__horizontal_well", "")
        h = pd.read_csv(p)
        t = pd.read_csv(p.parent / f"{wid}__typewell.csv")
        if is_train and "TVT" not in h.columns:
            continue
        feats = build_hidden_features(h, t, wid, is_train=is_train,
                                      formation_imputer=formation_imputer,
                                      row_imputer=row_imputer)
        if feats is not None:
            parts.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  {label}: {i + 1}/{len(paths)}", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()




# ---------------- Kaggle inference main (konbu recipe, OOF 11.885) ----------------
# Self-contained: rebuilds the train-reference imputers from the competition train/,
# builds features for the hidden test wells, loads the banked LGB x3 + XGB models from
# the artifacts dataset, and Ridge-blends. Prediction is CPU-only (no GPU needed).
import json
import sys

INPUT = Path("/kaggle/input")
ROW_QUERY_WORKERS = -1   # serial outer loop here -> let each k-NN query use all cores


def _all_dirs(root, maxdepth=5):
    out = [root]
    if not root.exists():
        return out
    def rec(d, depth):
        if depth > maxdepth:
            return
        try:
            for x in sorted(d.iterdir()):
                if x.is_dir():
                    out.append(x); rec(x, depth + 1)
        except Exception:
            pass
    rec(root, 1)
    return out


cands = _all_dirs(INPUT)
print("[input] dirs:", [str(p) for p in cands], flush=True)
COMP = next((p for p in cands if (p / "sample_submission.csv").exists()), None)
ART = next((p for p in cands if (p / "blend.json").exists()), None)
if COMP is None or ART is None:
    raise FileNotFoundError(f"COMP={COMP} ART={ART}; dirs={[str(p) for p in cands]}")
print(f"[locate] COMP={COMP}  ART={ART}", flush=True)

feature_cols = json.load(open(ART / "feature_cols.json"))
# Prefer the 5-model blend (LGBx3 + XGB + CatBoost, OOF 11.821) when present;
# fall back to the 4-model blend (OOF 11.885) otherwise.
blend_path = ART / "blend_catboost.json"
if not blend_path.exists():
    blend_path = ART / "blend.json"
print(f"[blend] using {blend_path.name}", flush=True)
blend = json.load(open(blend_path))
keys = blend["keys"]
coefs = np.asarray(blend["ridge_coef"], dtype=np.float64)
print(f"[artifacts] {len(feature_cols)} feats; blend keys={keys} coefs={coefs.round(3)}", flush=True)

all_h = list(COMP.rglob("*__horizontal_well.csv"))
train_paths = sorted(set(f for f in all_h if f.parent.name == "train"))
test_paths = sorted(set(f for f in all_h if f.parent.name == "test"))
print(f"[wells] train={len(train_paths)} test={len(test_paths)}", flush=True)

print(">> rebuild imputers from competition train/", flush=True)
form = FormationPlaneKNN(train_paths)
row = RowKNN(train_paths)
print(f"   plane wells={len(form.df)}  row pts={len(row.ancc):,}  maxself={row.maxself}", flush=True)

print(">> build test features", flush=True)
test_df = build_dataset(test_paths, form, row, is_train=False, label="test")
print(f"   test shape: {test_df.shape}", flush=True)
X = test_df[feature_cols]
Xv = X.values

N_SPLITS = 5
fam_pred = {}
for k in keys:
    preds = np.zeros(len(test_df), dtype=np.float64)
    if k.startswith("lgb_"):
        seed = k.split("_")[1]
        for fold in range(N_SPLITS):
            b = lgb.Booster(model_file=str(ART / f"lgb_seed{seed}_fold{fold}.txt"))
            preds += b.predict(X) / N_SPLITS
    elif k.startswith("cat_"):
        seed = k.split("_")[1]
        for fold in range(N_SPLITS):
            b = CatBoostRegressor()
            b.load_model(str(ART / f"cat_seed{seed}_fold{fold}.cbm"))
            preds += b.predict(Xv) / N_SPLITS
    else:  # xgb_<seed>
        seed = k.split("_")[1]
        dte = xgb.DMatrix(Xv)
        for fold in range(N_SPLITS):
            b = xgb.Booster(); b.load_model(str(ART / f"xgb_seed{seed}_fold{fold}.json"))
            it = (0, int(b.best_iteration) + 1)
            preds += b.predict(dte, iteration_range=it) / N_SPLITS
    fam_pred[k] = preds
    print(f"   {k}: mean drift={preds.mean():.3f}", flush=True)

drift = np.zeros(len(test_df), dtype=np.float64)
for c, k in zip(coefs, keys):
    drift += c * fam_pred[k]

tvt = test_df["last_known_tvt"].to_numpy(np.float64) + drift
sub = pd.DataFrame({"id": test_df["prediction_id"], "tvt": tvt})
out = Path("/kaggle/working/submission.csv")
sub.to_csv(out, index=False)
print(f">> wrote {len(sub)} rows -> {out}", flush=True)
print(sub.head(), flush=True); print(sub.tail(), flush=True)
print("=== KERNEL DONE ===", flush=True)
