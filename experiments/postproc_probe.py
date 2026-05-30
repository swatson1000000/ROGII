"""Post-processing probe on the cached konbu OOF (LB 11.921 / OOF 11.885).

Tests the frontier-recipe postproc stack (plan.md sec 8) on predicted DRIFT:
  1. Savitzky-Golay per-well smoothing of the drift trajectory (window, order 3)
  2. global drift shrinkage   pred *= alpha
  3. PS fade-in ramp          pred *= (1 - exp(-s/tau)),  s = rows since PS

Order of ops: SG-smooth -> shrink -> fade.  Shrink & fade are pointwise linear,
so for a fixed SG window we precompute the smoothed series once and sweep
(alpha, tau) by cheap vectorized RMSE.

Reports, for the full stack and each component alone:
  - baseline RMSE (no postproc)            -> must reproduce ~11.885
  - FULL-OOF tuned RMSE (optimistic ceiling: tune & score on same rows)
  - NESTED honest RMSE (GroupKFold-5 by WELL: tune on 4 groups, score held-out,
    rotate, concatenate) -> this is the number that predicts the LB.

A gain that survives the nested number is real; a gain only in the full-OOF
number is overfit and must be ignored.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

OOF = "/tmp/konbu_oof.csv"
SG_WINDOWS = [0, 7, 11, 17, 25, 41, 61]      # 0 = no smoothing
SG_ORDER = 3
ALPHAS = np.round(np.arange(0.80, 1.061, 0.01), 3)
TAUS = [0, 10, 20, 50, 100, 200]             # 0 = no fade
N_FOLDS = 5
SEED = 42


def rmse(err):
    return float(np.sqrt(np.mean(err * err)))


def load():
    df = pd.read_csv(OOF, usecols=["well", "row_idx", "target", "oof_pred"])
    df = df.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    # rows since PS, per well
    df["s"] = df["row_idx"] - df.groupby("well")["row_idx"].transform("min")
    # contiguous group ids + segment boundaries for fast per-well ops
    codes, _ = pd.factorize(df["well"], sort=False)
    df["g"] = codes
    return df


def sg_smooth_per_well(pred, group_starts, group_lens, window, order):
    """Apply savgol per well segment; window auto-clamped to odd <= seg length."""
    if window <= 0:
        return pred
    out = pred.copy()
    for st, ln in zip(group_starts, group_lens):
        if ln < order + 2:
            continue
        w = min(window, ln if ln % 2 == 1 else ln - 1)
        if w <= order:
            continue
        if w % 2 == 0:
            w -= 1
        if w <= order:
            continue
        seg = pred[st:st + ln]
        out[st:st + ln] = savgol_filter(seg, w, order)
    return out


def precompute_sg(pred, g, windows):
    """Return {window: smoothed_array} computed once."""
    # segment starts/lens for contiguous g
    change = np.flatnonzero(np.diff(g)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(g)]))
    lens = ends - starts
    cache = {}
    for w in windows:
        cache[w] = sg_smooth_per_well(pred, starts, ends - starts, w, SG_ORDER)
    return cache, lens


def best_params(target, sg_cache, s, alphas, taus, idx):
    """Grid search over (window, alpha, tau) on rows `idx`; return (params, rmse)."""
    tgt = target[idx]
    s_idx = s[idx]
    ramps = {tau: (1.0 - np.exp(-s_idx / tau)) if tau > 0 else np.ones_like(s_idx, dtype=np.float64)
             for tau in taus}
    best = (None, np.inf)
    for w, smoothed in sg_cache.items():
        base = smoothed[idx]
        for alpha in alphas:
            ba = base * alpha
            for tau in taus:
                err = tgt - ba * ramps[tau]
                r = rmse(err)
                if r < best[1]:
                    best = ((w, float(alpha), tau), r)
    return best


def apply_params(target, sg_cache, s, params, idx):
    w, alpha, tau = params
    base = sg_cache[w][idx] * alpha
    ramp = (1.0 - np.exp(-s[idx] / tau)) if tau > 0 else 1.0
    return target[idx] - base * ramp


def nested_rmse(df, sg_cache, alphas, taus):
    """GroupKFold-5 by well: tune on 4 groups, score held-out, concatenate."""
    target = df["target"].to_numpy(np.float64)
    s = df["s"].to_numpy(np.float64)
    wells = df["well"].to_numpy()
    uniq = pd.unique(wells)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(uniq))
    fold_of_well = {uniq[i]: perm[j] % N_FOLDS for j, i in enumerate(range(len(uniq)))}
    well_fold = np.array([fold_of_well[w] for w in wells])
    err_all = np.empty(len(target))
    chosen = []
    for f in range(N_FOLDS):
        tr = np.flatnonzero(well_fold != f)
        te = np.flatnonzero(well_fold == f)
        params, _ = best_params(target, sg_cache, s, alphas, taus, tr)
        err_all[te] = apply_params(target, sg_cache, s, params, te)
        chosen.append(params)
    return rmse(err_all), chosen


def main():
    print(f"Loading {OOF} ...", flush=True)
    df = load()
    n = len(df)
    target = df["target"].to_numpy(np.float64)
    pred = df["oof_pred"].to_numpy(np.float64)
    s = df["s"].to_numpy(np.float64)
    g = df["g"].to_numpy()
    print(f"  rows={n:,}  wells={df['well'].nunique()}", flush=True)

    base_rmse = rmse(target - pred)
    print(f"\nBASELINE (no postproc) RMSE = {base_rmse:.5f}   (expect ~11.885)\n", flush=True)

    print("Precomputing Savitzky-Golay smoothings ...", flush=True)
    sg_cache, _ = precompute_sg(pred, g, SG_WINDOWS)
    all_idx = np.arange(n)

    def report(name, windows, alphas, taus):
        full_p, full_r = best_params(target, sg_cache, s, alphas, taus, all_idx)
        sub_cache = {w: sg_cache[w] for w in windows}
        nested_r, chosen = nested_rmse(df, sub_cache, alphas, taus)
        print(f"--- {name} ---", flush=True)
        print(f"  full-OOF tuned : RMSE={full_r:.5f}  d={full_r-base_rmse:+.5f}  params(win,alpha,tau)={full_p[0]}", flush=True)
        print(f"  NESTED honest  : RMSE={nested_r:.5f}  d={nested_r-base_rmse:+.5f}", flush=True)
        print(f"  per-fold params: {chosen}\n", flush=True)
        return nested_r

    # component ablations (restrict the grid; sg_cache already restricted via windows arg)
    report("Savitzky-Golay only", SG_WINDOWS, np.array([1.0]), [0])
    report("Shrinkage only", [0], ALPHAS, [0])
    report("PS-fade only", [0], np.array([1.0]), TAUS)
    report("Shrink + fade", [0], ALPHAS, TAUS)
    full = report("FULL STACK (SG + shrink + fade)", SG_WINDOWS, ALPHAS, TAUS)

    print("=" * 60, flush=True)
    print(f"VERDICT: baseline {base_rmse:.5f} -> full-stack nested {full:.5f} "
          f"({full-base_rmse:+.5f} ft)", flush=True)
    print("Gate: nested gain must be clearly negative (improvement) to be worth "
          "wiring into the inference kernel.", flush=True)


if __name__ == "__main__":
    main()
