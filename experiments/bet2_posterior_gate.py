"""BET 2 — cheap gate: does a MOTION-COUPLED calibrated GR-match posterior carry signal
the frontier stack (OOF 10.356) hasn't already captured?

The build only keeps PF point-estimate + std and the beam ARGMAX path; the tw_diff/
target-distance family samples the GR-match likelihood over an offset grid INDEPENDENTLY
per row (no motion coupling). Bet 2's genuinely-new sliver = a per-row posterior MARGINAL
from a sequential forward-backward over a TVT-offset grid (motion-coupled + calibrated),
and especially its SHAPE: entropy, calibrated std, peak mass, second-mode separation —
"where is the alignment uncertain / multimodal", which point estimates throw away.

Minimal HMM (sum-product forward-backward) per well over the eval zone:
  state s    : TVT on a grid, last_tvt +/- 80 ft, step 1 ft (161 states)
  emission   : exp(-0.5*((GR_h[i] - typewell_GR(s))/sigma)^2); uniform where GR missing
  transition : Gaussian (smoothness) centered on s + expected_drift*dMD (motion model)
  -> gamma_i(s) = normalized marginal posterior over TVT at eval row i.

Posterior-shape features per row: post_std, post_entropy, post_max (peak mass),
post_modegap (TVT gap between top-2 local modes), and post_mean_drift (the HMM's own
point estimate, included only to contrast shape-only vs +point).

Gate (same residual-extractability probe as the dead self-anchor lever): reconstruct the
frontier blended OOF residual; fit a shallow GBM residual ~ posterior-shape feats under
GroupKFold-by-well; does it drop RMSE materially below 10.356?
  * shape-only adds ~0  -> the calibrated posterior is redundant; Bet 2 incremental claim dead.
  * shape-only adds signal -> the uncertainty/multimodality is the new lever; escalate to a
    full base(221)-vs-+posterior retrain, gated BLOCK-HOLDOUT OOF (Bet 3).

CAVEAT (fail-loud): (1) residual-fit is optimistic (no interaction cost, GKF-overfit risk);
(2) this is a fresh minimal HMM, NOT the build's tuned PF -- but it directly realizes the
motion-coupled smoothed marginal the build lacks. A flat null here, given the build already
exposes the independent offset grid + PF std, is treated as dead.

Run: nohup python -u experiments/bet2_posterior_gate.py \
       > log/bet2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
MAT = ROOT / "data/processed/frontier_seeded/train_feats.parquet"
MDL = ROOT / "models/frontier"

HW = 80.0      # state half-width (ft) around last_tvt
DS = 1.0       # state step (ft)
STRIDE = 5     # eval-row stride for the HMM chain (speed)
SIGMA_GR = 20.0
TRANS_STD = 3.0
PRIOR_STD = 5.0


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def build_T(nstate, shift_states):
    """Gaussian transition matrix T[s_prev, s_next] ~ N(s_next; s_prev + shift, TRANS_STD)."""
    idx = np.arange(nstate)
    d = (idx[None, :] - idx[:, None] - shift_states)        # (nstate, nstate)
    T = np.exp(-0.5 * (d * DS / TRANS_STD) ** 2)
    T /= T.sum(1, keepdims=True) + 1e-30
    return T


def posterior_feats(p):
    hw = pd.read_csv(p, usecols=["MD", "Z", "GR", "TVT_input"])
    ti = hw["TVT_input"].to_numpy(np.float64)
    mask = np.isnan(ti)
    if not mask.any():
        return None
    ms = int(mask.argmax())
    if ms < 20:
        return None
    md = hw["MD"].to_numpy(np.float64)
    gr = hw["GR"].to_numpy(np.float64)
    last_tvt = ti[ms - 1]

    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    twf = RAW / f"{wid}__typewell.csv"
    if not twf.exists():
        return None
    tw = pd.read_csv(twf).sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(np.float64)
    tw_gr = tw["GR"].to_numpy(np.float64)
    ok = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if ok.sum() < 20:
        return None
    tw_tvt, tw_gr = tw_tvt[ok], tw_gr[ok]

    # eval-zone chain (strided)
    ev = np.arange(ms, len(hw), STRIDE)
    if len(ev) < 10:
        return None
    g = gr[ev]
    nchain = len(ev)

    # state grid + typewell GR at each state
    states = last_tvt + np.arange(-HW, HW + DS / 2, DS)
    nstate = len(states)
    tw_at_state = np.interp(states, tw_tvt, tw_gr)            # (nstate,)

    # emission (nchain, nstate); uniform where GR missing
    miss = ~np.isfinite(g)
    gg = np.where(miss, 0.0, g)
    E = np.exp(-0.5 * ((gg[:, None] - tw_at_state[None, :]) / SIGMA_GR) ** 2)
    E[miss, :] = 1.0
    E += 1e-12

    # motion: expected drift rate from prefix TVT vs MD (last 200 known rows)
    a = max(0, ms - 200)
    pm, pt = md[a:ms], ti[a:ms]
    drate = np.polyfit(pm, pt, 1)[0] if len(pm) > 5 and np.ptp(pm) > 0 else 0.0
    dmd = float(np.median(np.diff(md[ev]))) if nchain > 1 else 1.0
    shift = (drate * dmd) / DS
    T = build_T(nstate, shift)
    Tt = T.T.copy()

    # forward
    alpha = np.empty((nchain, nstate))
    pr = np.exp(-0.5 * ((states - last_tvt) / PRIOR_STD) ** 2)
    a0 = pr * E[0]; alpha[0] = a0 / (a0.sum() + 1e-30)
    for i in range(1, nchain):
        ai = (alpha[i - 1] @ T) * E[i]
        alpha[i] = ai / (ai.sum() + 1e-30)
    # backward
    beta = np.empty((nchain, nstate))
    beta[-1] = 1.0 / nstate
    for i in range(nchain - 2, -1, -1):
        bi = (beta[i + 1] * E[i + 1]) @ Tt
        beta[i] = bi / (bi.sum() + 1e-30)
    gamma = alpha * beta
    gamma /= gamma.sum(1, keepdims=True) + 1e-30

    # posterior-shape features
    mean = gamma @ states
    var = gamma @ (states ** 2) - mean ** 2
    pstd = np.sqrt(np.clip(var, 0, None))
    ent = -(gamma * np.log(gamma + 1e-30)).sum(1)
    pmax = gamma.max(1)
    # second-mode gap: distance between top-2 LOCAL maxima of the marginal
    modegap = np.zeros(nchain)
    lt = np.zeros(nchain) + miss * 1.0  # gr_missing flag carried along
    interior = gamma[:, 1:-1]
    ispeak = (gamma[:, 1:-1] > gamma[:, :-2]) & (gamma[:, 1:-1] > gamma[:, 2:])
    for i in range(nchain):
        pk = np.flatnonzero(ispeak[i]) + 1
        if len(pk) >= 2:
            h = gamma[i, pk]
            top2 = pk[np.argsort(h)[-2:]]
            modegap[i] = abs(states[top2[0]] - states[top2[1]])

    return pd.DataFrame({
        "id": [f"{wid}_{r}" for r in ev],
        "p_std": pstd, "p_ent": ent, "p_max": pmax, "p_modegap": modegap,
        "p_grmiss": lt, "p_mean_drift": mean - last_tvt,
    })


def fit_residual(X, yv, groups, label):
    oof = np.zeros(len(yv))
    gkf = GroupKFold(n_splits=5)
    params = dict(objective="regression", metric="rmse", num_leaves=31, learning_rate=0.05,
                  bagging_fraction=0.8, bagging_freq=1, min_child_samples=200, verbose=-1)
    iters = []
    for tr, va in gkf.split(X, yv, groups):
        dtr = lgb.Dataset(X[tr], yv[tr]); dva = lgb.Dataset(X[va], yv[va], reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        oof[va] = bst.predict(X[va], num_iteration=bst.best_iteration)
        iters.append(bst.best_iteration)
    new = rmse(yv - oof); base = rmse(yv)
    print(f"  [{label}] base {base:.4f} -> {new:.4f}  (gain {base-new:+.4f})  best_iters {iters}", flush=True)
    return base - new


def main():
    print(">> frontier residual ...", flush=True)
    m = pd.read_parquet(MAT, columns=["id", "well", "target"])
    n = len(m)
    blend = json.loads((MDL / "blend_frontier.json").read_text())
    pred = np.zeros(n)
    for k, c in zip(blend["keys"], blend["ridge_coef"]):
        pred += c * np.load(MDL / f"oof_{k}.npy").astype(np.float64).ravel()
    m = m.reset_index(drop=True)
    m["_res"] = m["target"].to_numpy(np.float64) - pred
    print(f"   blend OOF RMSE = {rmse(m['_res']):.4f}", flush=True)

    paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    print(f">> HMM posterior over {len(paths)} wells (stride {STRIDE}) ...", flush=True)
    feats = []
    for i, p in enumerate(paths):
        try:
            f = posterior_feats(p)
        except Exception as e:
            print(f"   WARN {os.path.basename(p)}: {e}", flush=True); f = None
        if f is not None:
            feats.append(f)
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{len(paths)}", flush=True)
    F = pd.concat(feats, ignore_index=True)
    print(f"   posterior rows {len(F):,}", flush=True)

    d = m.merge(F, on="id", how="inner")
    print(f"   scored on {len(d):,} eval rows (stride subset)", flush=True)
    shape_cols = ["p_std", "p_ent", "p_max", "p_modegap", "p_grmiss"]
    yv = d["_res"].to_numpy(np.float64); groups = d["well"].to_numpy()
    print(f"   corr(p_mean_drift, target-on-subset proxy via residual context):", flush=True)
    for c in shape_cols + ["p_mean_drift"]:
        print(f"     corr({c}, residual) = {np.corrcoef(d[c], yv)[0,1]:+.4f}", flush=True)

    print("\n=== BET 2 GATE VERDICT (residual-extractability, GKF-by-well) ===", flush=True)
    g_shape = fit_residual(d[shape_cols].to_numpy(np.float32), yv, groups, "shape-only")
    g_all = fit_residual(d[shape_cols + ["p_mean_drift"]].to_numpy(np.float32), yv, groups, "shape+mean")
    print("", flush=True)
    if g_shape < 0.02 and g_all < 0.03:
        print("  >> NULL: motion-coupled posterior (incl. its shape) adds ~nothing beyond the", flush=True)
        print("     221 frontier feats. Bet 2's calibrated-posterior lever is REDUNDANT -- dead.", flush=True)
    elif g_shape < 0.06:
        print("  >> MARGINAL: a sliver from posterior shape, but residual-fit is optimistic.", flush=True)
        print("     Likely washes in a joint retrain; weigh vs the cost of building a tuned SMC.", flush=True)
    else:
        print("  >> SIGNAL in posterior SHAPE: the uncertainty/multimodality is genuinely new.", flush=True)
        print("     ESCALATE to base(221)-vs-+posterior retrain, gated BLOCK-HOLDOUT OOF (Bet 3).", flush=True)
    print("BET2 DONE", flush=True)


if __name__ == "__main__":
    main()
