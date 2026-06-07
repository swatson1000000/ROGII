"""Cheapest de-alias check, CORRECTED proxy: HMM VITERBI (MAP) path, not the marginal mean.

pf_dealias_check.py used the HMM marginal posterior MEAN -> it averages between aliases when
the posterior is multimodal (the exact aliasing case) and blew up numerically (max 11083 ft,
non-physical). A particle filter does NOT emit a marginal mean; it tracks a temporally-
consistent PATH. The right de-aliased proxy is the Viterbi MAP path over the same HMM grid:
the motion (transition) model forces one consistent track, killing the cross-alias averaging.
Done in log-space -> numerically stable (no underflow).

Same logic as before: the Viterbi path uses the FULL GR trace (forward+backward dynamic
programming) -> strictly more info than the notebooks' causal PF. If this de-aliased MAP track
is a GOOD standalone estimator and helps at the output level, BUILD the real 150-seed PF. If
even it is bad, that's now a STRONG signal against the PF lever (more-informed, properly
de-aliased, still can't localize).

Reports: standalone RMSE + tail of the MAP-path drift vs target (null ~15.9, frontier 10.356),
and the best output-blend weight (1-w)*frontier + w*MAP vs 10.356 (notebooks use 0.7).

Run: cd ROOT && nohup python -u -m experiments.pf_dealias_check2 \
       > log/pf_dealias2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import json
import os
import numpy as np
import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
MAT = ROOT / "data/processed/frontier_seeded/train_feats.parquet"
MDL = ROOT / "models/frontier"

HW = 80.0
DS = 1.0
STRIDE = 5
SIGMA_GR = 20.0
TRANS_STD = 3.0
PRIOR_STD = 5.0


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def stats(tag, err):
    ae = np.abs(err)
    print(f"  {tag:28s} RMSE={rmse(err):7.3f}  med={np.median(ae):6.2f}  "
          f"p90={np.percentile(ae,90):6.2f}  p99={np.percentile(ae,99):7.2f}  max={ae.max():7.1f}", flush=True)


def viterbi_path(p):
    hw = pd.read_csv(p, usecols=["MD", "Z", "GR", "TVT_input"])
    ti = hw["TVT_input"].to_numpy(np.float64)
    mask = np.isnan(ti)
    if not mask.any():
        return None
    ms = int(mask.argmax())
    if ms < 20:
        return None
    md = hw["MD"].to_numpy(np.float64); z = hw["Z"].to_numpy(np.float64); gr = hw["GR"].to_numpy(np.float64)
    last_tvt = ti[ms - 1]
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    twf = RAW / f"{wid}__typewell.csv"
    if not twf.exists():
        return None
    tw = pd.read_csv(twf).sort_values("TVT")
    tt, tg = tw["TVT"].to_numpy(np.float64), tw["GR"].to_numpy(np.float64)
    ok = np.isfinite(tt) & np.isfinite(tg)
    if ok.sum() < 20:
        return None
    tt, tg = tt[ok], tg[ok]

    ev = np.arange(ms, len(hw), STRIDE)
    if len(ev) < 10:
        return None
    g = gr[ev]; nchain = len(ev)
    states = last_tvt + np.arange(-HW, HW + DS / 2, DS)
    nstate = len(states)
    tw_at_state = np.interp(states, tt, tg)

    # log emission (uniform where GR missing)
    miss = ~np.isfinite(g)
    gg = np.where(miss, 0.0, g)
    logE = -0.5 * ((gg[:, None] - tw_at_state[None, :]) / SIGMA_GR) ** 2
    logE[miss, :] = 0.0

    # log transition with motion drift (normalized rows, then log)
    a = max(0, ms - 200); pm, pt = md[a:ms], ti[a:ms]
    drate = np.polyfit(pm, pt, 1)[0] if len(pm) > 5 and np.ptp(pm) > 0 else 0.0
    dmd = float(np.median(np.diff(md[ev]))) if nchain > 1 else 1.0
    shift = (drate * dmd) / DS
    idx = np.arange(nstate)
    dd = (idx[None, :] - idx[:, None] - shift)
    T = np.exp(-0.5 * (dd * DS / TRANS_STD) ** 2)
    T /= T.sum(1, keepdims=True) + 1e-30
    logT = np.log(T + 1e-30)

    # Viterbi (log-space, stable)
    delta = -0.5 * ((states - last_tvt) / PRIOR_STD) ** 2 + logE[0]
    back = np.empty((nchain, nstate), np.int32)
    for i in range(1, nchain):
        cand = delta[:, None] + logT          # (prev, next)
        amax = cand.argmax(0)
        back[i] = amax
        delta = cand[amax, np.arange(nstate)] + logE[i]
    path = np.empty(nchain, np.int32)
    path[-1] = int(delta.argmax())
    for i in range(nchain - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    map_tvt = states[path]
    return pd.DataFrame({"id": [f"{wid}_{r}" for r in ev], "v_drift": map_tvt - last_tvt})


def main():
    m = pd.read_parquet(MAT, columns=["id", "well", "target"])
    n = len(m)
    blend = json.loads((MDL / "blend_frontier.json").read_text())
    pred = np.zeros(n)
    for k, c in zip(blend["keys"], blend["ridge_coef"]):
        pred += c * np.load(MDL / f"oof_{k}.npy").astype(np.float64).ravel()
    m = m.reset_index(drop=True); m["_pred"] = pred
    print(f">> frontier blend OOF RMSE (full) = {rmse(m['target']-m['_pred']):.4f}", flush=True)

    paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    print(f">> Viterbi MAP over {len(paths)} wells (stride {STRIDE}) ...", flush=True)
    feats = []
    for i, p in enumerate(paths):
        try:
            f = viterbi_path(p)
        except Exception as e:
            print(f"   WARN {os.path.basename(p)}: {e}", flush=True); f = None
        if f is not None:
            feats.append(f)
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{len(paths)}", flush=True)
    F = pd.concat(feats, ignore_index=True)

    d = m.merge(F, on="id", how="inner")
    tgt = d["target"].to_numpy(np.float64); vt = d["v_drift"].to_numpy(np.float64); fr = d["_pred"].to_numpy(np.float64)
    print(f"\n>> scored on {len(d):,} strided eval rows (frontier-on-subset RMSE = {rmse(tgt-fr):.4f})", flush=True)

    print("\n=== (1) DE-ALIASING: Viterbi-MAP standalone quality ===", flush=True)
    stats("null (drift=0)", tgt)
    stats("Viterbi-MAP standalone", tgt - vt)
    stats("frontier stack (subset)", tgt - fr)

    print("\n=== (2) OUTPUT-BLEND VALUE ===", flush=True)
    base = rmse(tgt - fr); best_w, best_r = 0.0, base
    for w in np.linspace(0, 1, 21):
        r = rmse(tgt - ((1 - w) * fr + w * vt))
        if w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            tag = "  <= notebooks use 0.7" if abs(w - 0.7) < 1e-9 else ""
            print(f"  w={w:.2f}  RMSE={r:.4f}  ({r-base:+.4f}){tag}", flush=True)
        if r < best_r:
            best_r, best_w = r, w

    print("\n=== VERDICT ===", flush=True)
    print(f"  Viterbi-MAP standalone RMSE = {rmse(tgt-vt):.3f}  (null {rmse(tgt):.2f}, frontier {base:.3f})", flush=True)
    print(f"  best blend w* = {best_w:.2f} -> {best_r:.4f}  (gain {base-best_r:+.4f})", flush=True)
    sa = rmse(tgt - vt)
    if best_w < 0.05 or base - best_r < 0.02:
        print("  >> NO output-blend payoff even from the de-aliased MAP track (more info than a", flush=True)
        print("     causal PF). STRONG-ish signal the PF-dominant-blend lever won't transfer to OUR", flush=True)
        print("     stack as-is; the notebooks' 0.7 is likely public-tuned. Don't build blindly.", flush=True)
        if sa > 14:
            print("     AND the MAP track is ~null-level standalone -> de-aliasing did NOT rescue the", flush=True)
            print("     GR point-estimate here (untuned sigma/trans caveat still applies).", flush=True)
    elif base - best_r < 0.10:
        print("  >> SMALL payoff; nowhere near 0.7. Their tuned PF may beat my proxy -> build real PF", flush=True)
        print("     to check, but expect a modest optimal weight on our OOF, not 0.7.", flush=True)
    else:
        print("  >> REAL payoff: de-aliased track adds materially at output level -> 'GR point-est", flush=True)
        print("     dead' was an under-ensembling false negative. BUILD the real 150-seed PF + blend.", flush=True)
    print("PFDEALIAS2 DONE", flush=True)


if __name__ == "__main__":
    main()
