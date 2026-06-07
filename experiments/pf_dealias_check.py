"""Cheapest first cut on the "seed-ensembled PF de-aliases" claim (the live direction).

The public-frontier notebooks trust a 150-seed likelihood-weighted PF at 0.7 of the OUTPUT.
That contradicts our finding that GR point-estimates are aliased/dead. Before building the
real tuned PF, test the crux cheaply with a PROXY we already have:

  Bet-2's HMM forward-BACKWARD posterior mean (experiments/bet2_posterior_gate.posterior_feats)
  is a de-aliased GR-match TVT estimate that uses the FULL GR trace (incl. future GR). The
  notebooks' PF is causal (forward-only). So the HMM smoother has STRICTLY MORE information
  than their PF -> if even this de-aliased smoother mean is a bad standalone estimator, a
  causal PF cannot be better; if it's good, the PF lever is worth building for real.

Two questions, both cheap (reuse cached frontier OOF; no GBM retrain):
  1. DE-ALIASING: standalone RMSE of the HMM-mean drift vs true target, with tail (p90/p99).
     null (predict last_known, drift=0) ~ 15.9. frontier stack = 10.356. Where does it land,
     and is the heavy alias tail controlled?
  2. OUTPUT-BLEND VALUE: best w in (1-w)*frontier_pred + w*HMM_mean vs the target. Does any
     w>0 beat 10.356? The notebooks use w=0.7; if our optimal w~0 the 0.7 is public-overfit
     (or their PF >> our proxy -> escalate). If optimal w is substantial AND beats 10.356,
     the output-blend lever is real.

ASYMMETRY (how to read it): PASS (helps) -> build the real tuned 150-seed PF. FAIL (w~0) ->
weaker, because my HMM uses fixed sigma/trans (untuned) vs their adaptive multi-scale PF;
escalate to the real PF before killing. But a FAIL of the more-informed smoother is meaningful.

Run: nohup python -u experiments/pf_dealias_check.py \
       > log/pf_dealias_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import json
import os
import numpy as np
import pandas as pd

import experiments.bet2_posterior_gate as b2   # reuse posterior_feats + the HMM

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
MAT = ROOT / "data/processed/frontier_seeded/train_feats.parquet"
MDL = ROOT / "models/frontier"


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def stats(tag, err):
    ae = np.abs(err)
    print(f"  {tag:28s} RMSE={rmse(err):7.3f}  med={np.median(ae):6.2f}  "
          f"p90={np.percentile(ae,90):6.2f}  p99={np.percentile(ae,99):7.2f}  max={ae.max():7.1f}", flush=True)


def main():
    # frontier blend OOF pred + target
    m = pd.read_parquet(MAT, columns=["id", "well", "target"])
    n = len(m)
    blend = json.loads((MDL / "blend_frontier.json").read_text())
    pred = np.zeros(n)
    for k, c in zip(blend["keys"], blend["ridge_coef"]):
        pred += c * np.load(MDL / f"oof_{k}.npy").astype(np.float64).ravel()
    m = m.reset_index(drop=True)
    m["_pred"] = pred
    print(f">> frontier blend OOF RMSE (full) = {rmse(m['target']-m['_pred']):.4f}", flush=True)

    # HMM posterior-mean drift per strided eval row (reuse Bet-2 machinery)
    paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    print(f">> HMM smoother over {len(paths)} wells (stride {b2.STRIDE}) ...", flush=True)
    feats = []
    for i, p in enumerate(paths):
        try:
            f = b2.posterior_feats(p)
        except Exception as e:
            print(f"   WARN {os.path.basename(p)}: {e}", flush=True); f = None
        if f is not None:
            feats.append(f[["id", "p_mean_drift"]])
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{len(paths)}", flush=True)
    F = pd.concat(feats, ignore_index=True)

    d = m.merge(F, on="id", how="inner")
    tgt = d["target"].to_numpy(np.float64)
    pf = d["p_mean_drift"].to_numpy(np.float64)
    fr = d["_pred"].to_numpy(np.float64)
    print(f"\n>> scored on {len(d):,} strided eval rows (frontier-on-subset RMSE = {rmse(tgt-fr):.4f})", flush=True)

    print("\n=== (1) DE-ALIASING: standalone estimator quality ===", flush=True)
    stats("null (drift=0/last_known)", tgt - 0.0)
    stats("HMM-mean standalone", tgt - pf)
    stats("frontier stack (subset)", tgt - fr)

    print("\n=== (2) OUTPUT-BLEND VALUE: (1-w)*frontier + w*HMM ===", flush=True)
    base = rmse(tgt - fr)
    best_w, best_r = 0.0, base
    for w in np.linspace(0.0, 1.0, 21):
        r = rmse(tgt - ((1 - w) * fr + w * pf))
        flag = "  <= notebooks use 0.7" if abs(w - 0.7) < 1e-9 else ""
        if w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            print(f"  w={w:.2f}  RMSE={r:.4f}  ({r-base:+.4f}){flag}", flush=True)
        if r < best_r:
            best_r, best_w = r, w

    print("\n=== VERDICT ===", flush=True)
    print(f"  best blend weight w* = {best_w:.2f}  ->  RMSE {best_r:.4f}  (vs frontier {base:.4f}, gain {base-best_r:+.4f})", flush=True)
    if best_w < 0.05 or base - best_r < 0.02:
        print("  >> NO de-alias payoff from the smoother proxy: optimal blend ~ignores it.", flush=True)
        print("     WEAK kill (proxy is untuned vs their adaptive multi-scale PF) -> if pursuing,", flush=True)
        print("     escalate to the REAL 150-seed likelihood-weighted PF before declaring dead.", flush=True)
    elif base - best_r < 0.10:
        print("  >> SMALL payoff: de-aliasing helps a little at the output level, but nowhere near", flush=True)
        print("     a 0.7 weight. The notebooks' 0.7 is likely public-overfit OR their tuned PF", flush=True)
        print("     >> my proxy. Build the real PF to find out; expect <0.7 optimal on our OOF.", flush=True)
    else:
        print("  >> REAL de-alias payoff: a de-aliased GR-match estimate adds materially at the", flush=True)
        print("     OUTPUT level -> our 'GR point-estimate dead' was an under-ensembling false", flush=True)
        print("     negative. BUILD the real tuned 150-seed PF + output blend; gate vs 10.356.", flush=True)
    print("PFDEALIAS DONE", flush=True)


if __name__ == "__main__":
    main()
