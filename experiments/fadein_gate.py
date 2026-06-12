"""CAND A gate: exponential drift fade-in (+ shrinkage) post-process on the banked nouk blend.

Transform (ravaghi apply_pp / fle3n Engine-B "warmup"):
    f = alpha * (1 - exp(-max(md_since, 0) / tau))
applied to drift, then tvt = last_known_tvt + d * f.

Two application modes, both gridded:
  FINAL : fade the final blended drift   d_blend * f
  GBMARM: fade the GBM arm only          0.43*(d_gbm*f) + 0.57*d_pf   (fle3n Engine-B shape)

Protocol: identical loading/alignment to weights_axis_scan.py (self-checks reproduce
10.3232 nouk stack OOF). Production blend w=0.57 FIXED (the LB vertex). Params (tau, alpha)
selected OUT-OF-FOLD over the same RandomState(42) 5-fold well split; in-sample best
reported only as an optimistic bound.

PRE-REGISTERED BAR (set before running): OOF-selected fade must improve the w=0.57
blended OOF by >= 0.02 ft with stable per-fold params, else CAND A is NULL and closes.

Extra readout (exploratory, NOT the gate): per-well Savitzky-Golay smoothing of the
blended drift (fle3n Engine-B ships window~17 order 3).

Run: nohup python -u experiments/fadein_gate.py > log/fadein_gate_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
KEYS = ["lgb_42", "lgb_7", "lgb_123", "cat_42", "cat_7", "cat_123"]
W_PF = 0.57
rmse = lambda e: float(np.sqrt(np.mean(e * e)))

print(">> load frontier_ens train matrix...", flush=True)
df = pd.read_parquet(ROOT / "data/processed/frontier_ens/train_feats.parquet",
                     columns=["well", "id", "last_known_tvt", "target", "md_since"])
y = df["target"].to_numpy(np.float64)
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)

X = np.column_stack([np.load(ROOT / "models/frontier_ens_nouk" / f"oof_{k}.npy").astype(np.float64)
                     for k in KEYS])
assert len(X) == len(y)
r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(X, y)
g_res_all = X @ r.coef_ - y
print(f"   nouk stack OOF = {rmse(g_res_all):.4f}  [self-check 10.3232]", flush=True)
assert abs(rmse(g_res_all) - 10.3232) < 0.01

# --- PF residual, aligned per-well (verbatim protocol from weights_axis_scan.py) ---
print(">> align PF (128-seed scale-12) residuals...", flush=True)
paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells = []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue
    pf_wells.append(wid)
grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}


def good(r):
    return r is not None and not (isinstance(r[0], str) and r[0] == "ERR")


res = joblib.load(ROOT / "models/frontier/pf_real_results.pkl")
rg = [r for r in res if good(r)]
assert len(rg) == len(pf_wells)
pos_l, pres_l = [], []
for i, wid in enumerate(pf_wells):
    truth, pf = rg[i]
    sub = grp.get(wid)
    if sub is None or len(sub) != len(truth):
        continue
    if not np.allclose(sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64),
                       truth, atol=1e-3):
        continue
    pos_l.append(sub.index.to_numpy())
    pres_l.append(pf["pf_scale_12"] - truth)
pos = np.concatenate(pos_l)
p_res = np.concatenate(pres_l)
print(f"   PF rows aligned: {len(pos)} / {len(df)}", flush=True)

yp = y[pos]
g_res = g_res_all[pos]
md = df["md_since"].to_numpy(np.float64)[pos]
wells_p = df["well"].to_numpy()[pos]
print(f"   md_since on eval rows: min {md.min():.1f}  p50 {np.median(md):.0f}  max {md.max():.0f}"
      f"  (<85 ft: {(md < 85).mean()*100:.1f}% of rows)", flush=True)
md = np.maximum(md, 0.0)

# drift predictions
d_gbm = yp + g_res           # GBM stack predicted drift
d_pf = yp + p_res            # PF predicted drift
d_blend = (1 - W_PF) * d_gbm + W_PF * d_pf
base = rmse(d_blend - yp)
print(f"\n   BASELINE blended OOF @ w={W_PF} = {base:.4f}", flush=True)

# production 5-fold well split
rng = np.random.RandomState(42)
uw = df["well"].unique().copy()
rng.shuffle(uw)
fo = {w: i % 5 for i, w in enumerate(uw)}
fold = np.array([fo[w] for w in wells_p])

TAUS = [5.0, 10.0, 20.0, 40.0, 60.0, 85.0, 120.0, 170.0, 250.0, 400.0, 700.0]
ALPHAS = [0.85, 0.90, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05]


def fade(tau, alpha):
    return alpha * (1.0 - np.exp(-md / tau))


def apply_mode(mode, tau, alpha):
    f = fade(tau, alpha)
    if mode == "FINAL":
        return d_blend * f
    return (1 - W_PF) * (d_gbm * f) + W_PF * d_pf  # GBMARM


for mode in ("FINAL", "GBMARM"):
    print(f"\n=== mode {mode}: in-sample grid (optimistic bound) ===", flush=True)
    grid = {(t, a): rmse(apply_mode(mode, t, a) - yp) for t in TAUS for a in ALPHAS}
    best = min(grid, key=grid.get)
    print(f"   in-sample best: tau={best[0]:.0f} alpha={best[1]} -> {grid[best]:.4f} "
          f"({grid[best] - base:+.4f} vs base)", flush=True)
    rav = rmse(apply_mode(mode, 85.0, 1.0) - yp)
    print(f"   ravaghi point (tau=85, alpha=1.0)  -> {rav:.4f} ({rav - base:+.4f})", flush=True)
    # pure shrinkage reference: alpha sweep at tau->0 (f ~= alpha everywhere)
    shr = {a: rmse(apply_mode(mode, 1e-6, a) - yp) for a in ALPHAS}
    bs = min(shr, key=shr.get)
    print(f"   pure-shrinkage best: alpha={bs} -> {shr[bs]:.4f} ({shr[bs] - base:+.4f})", flush=True)

    # out-of-fold selection (the gated number)
    oof_pred = np.empty(len(yp))
    chosen = []
    for fnum in range(5):
        tr, va = fold != fnum, fold == fnum
        sc = {(t, a): rmse((apply_mode(mode, t, a) - yp)[tr]) for t in TAUS for a in ALPHAS}
        t_, a_ = min(sc, key=sc.get)
        chosen.append((t_, a_))
        oof_pred[va] = apply_mode(mode, t_, a_)[va]
    o = rmse(oof_pred - yp)
    print(f"   >>> OUT-OF-FOLD OOF = {o:.4f}  ({o - base:+.4f} vs base {base:.4f})", flush=True)
    print(f"   per-fold chosen (tau, alpha): {chosen}", flush=True)
    print(f"   GATE (>= -0.02 required): {'PASS' if o - base <= -0.02 else 'NULL'}", flush=True)

# --- exploratory: per-well Savitzky-Golay on the blended drift (NOT the gate) ---
print("\n=== exploratory: per-well Savitzky-Golay (window 17, order 3) on blended drift ===", flush=True)
sg = d_blend.copy()
start = 0
for arr in pres_l:
    n = len(arr)
    seg = d_blend[start:start + n]
    if n >= 17:
        sg[start:start + n] = savgol_filter(seg, 17, 3)
    start += n
s = rmse(sg - yp)
print(f"   SG OOF = {s:.4f} ({s - base:+.4f} vs base)", flush=True)
for w in (5, 9, 33, 51):
    sg2 = d_blend.copy()
    start = 0
    for arr in pres_l:
        n = len(arr)
        if n >= w:
            sg2[start:start + n] = savgol_filter(d_blend[start:start + n], w, min(3, w - 2))
        start += n
    s2 = rmse(sg2 - yp)
    print(f"   SG window {w}: {s2:.4f} ({s2 - base:+.4f})", flush=True)

print("\nFADEIN GATE DONE", flush=True)
