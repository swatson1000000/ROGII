"""Which standalone TVT estimators are orthogonal enough to output-blend on top of the
CURRENT 8.269 blend (0.56*GBM + 0.44*PF) the way the PF was on top of the GBM?

For every candidate estimator column in the frontier-222 matrix we compute:
  - standalone OOF RMSE (how accurate it is alone)
  - corr of its residual with the CURRENT BLEND residual (orthogonality to 8.269, not just GBM)
  - closed-form + OUT-OF-FOLD optimal 2-way blend gain on top of the current blend
  - (context) the same gain on top of GBM-only -- what it WOULD have bought pre-PF

The dividend needs BOTH near-blend accuracy AND low corr-to-blend. This ranks candidates by
the honest out-of-fold gain. Reuses pf_output_blend.py's residual-space alignment verbatim.

Spaces: abs estimators (~11500) vs delta estimators (~tens) auto-detected by median magnitude.
"""
from pathlib import Path
import glob, os, json
import numpy as np
import pandas as pd
import joblib

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
FR = ROOT / "data/processed/frontier_seeded"
MODELS = ROOT / "models/frontier"
PF_KEY = "pf_scale_12"
W_PF = 0.44  # production output-blend weight (final = 0.56*GBM + 0.44*PF)

# candidate standalone TVT estimators, grouped by mechanism
GR_TRACKERS = [
    "pf_ancc_delta", "pf_z_delta",
    "beam_cons_d", "beam_loose_d", "beam_vcons_d", "beam_sm5_d", "beam_vloose_d",
    "beam_mid_d", "beam_stiff_d", "beam_mean_d", "beam_med_d",
    "sc8_d", "sc15_d", "sc25_d", "sc_cons_d", "sc_ens_d", "hyb_d",
    "dtw_ens_d", "dtw_stoch_mean_d", "dtw_r20_d", "dtw_r50_d", "dtw_r100_d", "dtw_r200_d",
]
GEOMETRY = [
    "tvtF_ANCC", "tvtFw_ANCC", "tvtF50_ANCC",
    "tvtF_ASTNU", "tvtF_ASTNL", "tvtF_EGFDU", "tvtF_EGFDL", "tvtF_BUDA",
    "form_mean_d", "tvt_dense_d", "tvt_densew_d", "tvt_dense50_d",
]
CANDS = GR_TRACKERS + GEOMETRY
GROUP = {**{c: "GR-track" for c in GR_TRACKERS}, **{c: "geometry" for c in GEOMETRY}}

# ---------------------------------------------------------------- load + align (from pf_output_blend.py)
print(">> loading frontier OOF + candidate columns...", flush=True)
cols = ["well", "id", "last_known_tvt", "target"] + CANDS
df = pd.read_parquet(FR / "train_feats.parquet", columns=cols)
blend = json.load(open(MODELS / "blend_frontier.json"))
keys, coef = blend["keys"], np.array(blend["ridge_coef"], np.float64)
oofs = np.column_stack([np.load(MODELS / f"oof_{k}.npy").astype(np.float64) for k in keys])
blended_drift = oofs @ coef
target = df["target"].to_numpy(np.float64)
lastk = df["last_known_tvt"].to_numpy(np.float64)
gbm_resid = blended_drift - target            # GBM_abs - true_tvt
true_tvt = target + lastk
print(f"   GBM blended OOF RMSE = {np.sqrt(np.mean(gbm_resid**2)):.4f}  (expect 10.3556)", flush=True)
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)

print(">> replaying PF well order...", flush=True)
paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells = []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue
    pf_wells.append(wid)
results = joblib.load(MODELS / "pf_real_results.pkl")
def is_err(r): return r is not None and isinstance(r[0], str) and r[0] == "ERR"
good = [r for r in results if r is not None and not is_err(r)]
assert len(good) == len(pf_wells), "replay count mismatch -> skip logic drifted; aborting"

gbm_keep, pf_keep = [], []
n_ok = n_bad = 0
grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}
for wid, (pf_truth, preds) in zip(pf_wells, good):
    sub = grp.get(wid)
    if sub is None or len(sub) != len(pf_truth):
        n_bad += 1; continue
    t_tvt = sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64)
    if not np.allclose(t_tvt, pf_truth, atol=1e-3, rtol=0):
        n_bad += 1; continue
    n_ok += 1
    gbm_keep.append(sub.index.to_numpy())
    pf_keep.append(preds[PF_KEY].astype(np.float64) - pf_truth)   # PF_abs - true_tvt
pos = np.concatenate(gbm_keep)                                     # row positions, PF-aligned
print(f"   aligned wells ok={n_ok} bad={n_bad}  rows={len(pos):,} ({len(pos)/len(df)*100:.1f}%)", flush=True)

g_res = gbm_resid[pos]
p_res = np.concatenate(pf_keep)
blend_res = (1 - W_PF) * g_res + W_PF * p_res                      # CURRENT 8.269 blend residual
true_a = true_tvt[pos]
lastk_a = lastk[pos]
print(f"   current blend OOF RMSE = {np.sqrt(np.mean(blend_res**2)):.4f}  (expect ~9.17)", flush=True)
print(f"   GBM-only      OOF RMSE = {np.sqrt(np.mean(g_res**2)):.4f}", flush=True)

# fold map (GKF-5 seed42, as in training) for out-of-fold weights
rng = np.random.RandomState(42)
uw = df["well"].unique().copy(); rng.shuffle(uw)
fold_of = {w: i % 5 for i, w in enumerate(uw)}
fold = np.array([fold_of[w] for w in df["well"].to_numpy()[pos]])

def oof_blend_gain(base_res, cand_res, mask):
    """out-of-fold optimal 2-way blend of base_res with cand_res over mask. returns (base_rmse, gain, mean_w)."""
    b, c, fl = base_res[mask], cand_res[mask], fold[mask]
    base_rmse = np.sqrt(np.mean(b ** 2))
    out = b.copy(); ws = []
    for f in range(5):
        tr, va = fl != f, fl == f
        if tr.sum() < 100 or va.sum() == 0:
            continue
        d = b[tr] - c[tr]
        denom = np.dot(d, d)
        wf = float(np.dot(b[tr], d) / denom) if denom > 0 else 0.0
        out[va] = (1 - wf) * b[va] + wf * c[va]
        ws.append(wf)
    rmse = np.sqrt(np.mean(out ** 2))
    return base_rmse, base_rmse - rmse, float(np.mean(ws)) if ws else 0.0

# ---------------------------------------------------------------- per-candidate scan
print("\n=== CANDIDATE SCAN (residual space, PF-aligned eval rows) ===", flush=True)
hdr = f"{'candidate':<18}{'grp':<10}{'cov%':>6}{'stand':>8}{'corrB':>7}{'wB':>6}{'gainB':>8}{'corrG':>7}{'gainG':>8}"
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
rows = []
for c in CANDS:
    col = df[c].to_numpy(np.float64)[pos]
    valid = np.isfinite(col)
    cov = valid.mean() * 100
    med = np.nanmedian(np.abs(col[valid])) if valid.any() else 0.0
    est_abs = col if med > 500 else lastk_a + col           # abs vs delta auto-detect
    c_res = est_abs - true_a
    # candidate-specific mask = finite candidate (blend/gbm residuals are always finite)
    m = valid & np.isfinite(c_res)
    if m.sum() < 1000:
        print(f"{c:<18}{GROUP[c]:<10}{cov:>6.1f}  -- too few valid rows --", flush=True)
        continue
    stand = np.sqrt(np.mean(c_res[m] ** 2))
    corrB = np.corrcoef(c_res[m], blend_res[m])[0, 1]
    corrG = np.corrcoef(c_res[m], g_res[m])[0, 1]
    baseB, gainB, wB = oof_blend_gain(blend_res, c_res, m)
    baseG, gainG, wG = oof_blend_gain(g_res, c_res, m)
    rows.append((c, GROUP[c], cov, stand, corrB, wB, gainB, corrG, gainG))
    print(f"{c:<18}{GROUP[c]:<10}{cov:>6.1f}{stand:>8.2f}{corrB:>7.3f}{wB:>6.2f}{gainB:>+8.3f}{corrG:>7.3f}{gainG:>+8.3f}", flush=True)

# ---------------------------------------------------------------- ranked verdict
rows.sort(key=lambda r: -r[6])  # by gain-on-blend
print("\n=== TOP CANDIDATES BY OUT-OF-FOLD GAIN ON TOP OF THE 8.269 BLEND ===", flush=True)
print(f"{'candidate':<18}{'grp':<10}{'gainB':>8}{'gainG(pre-PF)':>15}", flush=True)
for c, grp_, cov, stand, corrB, wB, gainB, corrG, gainG in rows[:8]:
    print(f"{c:<18}{grp_:<10}{gainB:>+8.3f}{gainG:>+15.3f}", flush=True)

best = rows[0] if rows else None
print("\n=== VERDICT ===", flush=True)
if best is None:
    print("  no valid candidates.", flush=True)
else:
    c, grp_, cov, stand, corrB, wB, gainB, corrG, gainG = best
    print(f"  best on-top-of-blend: {c} ({grp_}) gain={gainB:+.3f} ft  (corr-to-blend {corrB:.3f}, stand {stand:.2f})", flush=True)
    if gainB >= 0.10:
        print(f"  BANKABLE: a 2nd orthogonal estimator exists -> measure as PF-style output member, gate, submit.", flush=True)
    elif gainB >= 0.03:
        print(f"  MARGINAL: below the ~0.23 OOF<->LB resolution -> free fold-in at best, not a standalone sub.", flush=True)
    else:
        print(f"  NULL: the averaging axis is spent -- every candidate is redundant-with-PF or absorbed-by-GBM.", flush=True)
        print(f"  (note gainG = what each WOULD have bought pre-PF; if those are large but gainB ~0, PF already", flush=True)
        print(f"   captured that orthogonality -> route/select per-well, don't average a 4th member.)", flush=True)
print("SCAN DONE", flush=True)
