"""GATE: standalone OOF of the REAL 150-seed likelihood-weighted multi-scale PF.

The live lever (3 public notebooks at ~8.2) trusts this PF at 0.7 of the OUTPUT. Our cheap
checks said: single-seed PF (real likelihood) = 14.37 standalone, output-blends at only w=0.08,
naive 2-seed averaging is shallow (aliasing is systematic). The ONE untested mechanism is
likelihood-WEIGHTED multi-scale seed SELECTION (re-weight seeds by accumulated GR loglik ->
keep right-alias seeds). This gate measures whether THAT takes the PF from ~14 to competitive
with the GBM (~10).

PF code copied VERBATIM from /tmp/rogii_new/ravaghi_.../wellbore-geology-prediction-ridge.py
(run_particle_filter L85-161, run_pf_lik_ensemble_scales L180-196). No leak: the PF uses only
GR (full trace, observed at inference) + TVT_input prefix + Z/MD; TVT (true) is used ONLY to
score. Per-well filter, no training -> standalone RMSE over all train eval rows IS the OOF.

Decision (per PICK_UP): best ensembled-PF standalone OOF
  <= ~10  -> de-aliasing via likelihood-selection is real -> BUILD output-blend, gate vs 10.356.
  ~13-14  -> selection didn't beat single-seed -> notebooks' 0.7 is their-weaker-GBM/public-tuned
             -> DROP the lever, move to hardening 10.122.
Also reports pf_mean (naive 128-avg) vs the likelihood-weighted scales -> isolates whether
WEIGHTING (selection) beats AVERAGING (variance reduction).

Run: cd ROOT && nohup python -u experiments/pf_real_gate.py \
       > log/pf_real_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
SCALES = (3.0, 5.0, 8.0, 12.0)
N_SEEDS = 128
N_PART = 500


# ---- VERBATIM from ravaghi source (run_particle_filter, ensemble) ----
def run_particle_filter(hw, tw, n_particles=500, seed=42):
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0
    last = kn.iloc[-1]
    last_tvt = float(last['TVT_input']); last_Z = float(last['Z']); last_MD = float(last['MD'])
    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values); dz = np.diff(tail['Z'].values); dm = np.diff(tail['MD'].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + 2.0 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5
    md_v = ev['MD'].values.astype(float); z_v = ev['Z'].values.astype(float)
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]
    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD; log_lik = 0.0
    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N
        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]
    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble_scales(hw, tw, scales=SCALES, n_particles=500, n_seeds=128):
    preds = []; liks = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p); liks.append(ll)
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks); liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = np.exp(liks_n / float(scale)); weights /= weights.sum()
        out[f'pf_scale_{scale:g}'] = (weights[:, None] * pred_arr).sum(0)
    out['pf_mean'] = pred_arr.mean(0)
    return out
# ---- end verbatim ----


def process(path):
    wid = os.path.basename(path).replace("__horizontal_well.csv", "")
    try:
        hw = pd.read_csv(path)
        tw = pd.read_csv(RAW / f"{wid}__typewell.csv")
        if 'TVT' not in hw or hw['TVT_input'].isna().sum() == 0 or hw['TVT_input'].notna().sum() < 20:
            return None
        ev_mask = hw['TVT_input'].isna().values
        true_tvt = hw['TVT'].values.astype(float)
        if not np.isfinite(true_tvt[ev_mask]).all():
            return None
        out = run_pf_lik_ensemble_scales(hw, tw, n_particles=N_PART, n_seeds=N_SEEDS)
        keys = list(out.keys())
        preds = {k: out[k][ev_mask] for k in keys}
        return (true_tvt[ev_mask], preds)
    except Exception as e:
        return ('ERR', f"{wid}: {e}")


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def main():
    paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    print(f">> {len(paths)} wells; {N_SEEDS} seeds x {N_PART} particles, parallel ...", flush=True)
    results = Parallel(n_jobs=14, verbose=5)(delayed(process)(p) for p in paths)
    import joblib
    save = ROOT / "models/frontier/pf_real_results.pkl"
    joblib.dump(results, save)
    print(f">> saved raw results -> {save}", flush=True)
    def is_err(r):
        return r is not None and isinstance(r[0], str) and r[0] == 'ERR'
    errs = [r[1] for r in results if is_err(r)]
    for e in errs[:10]:
        print(f"   ERR {e}", flush=True)
    good = [r for r in results if r is not None and not is_err(r)]
    print(f">> {len(good)} wells scored ({len(errs)} errors)", flush=True)

    keys = list(good[0][1].keys())
    truth = np.concatenate([g[0] for g in good])
    print(f"\n=== STANDALONE OOF (eval rows = {len(truth):,}) — RMSE of TVT_pred vs TVT_true ===", flush=True)
    print(f"  {'null (last_known)':22s} 15.910   {'frontier GBM stack':22s} 10.356   {'single-seed PF':22s} 14.37 (cached)", flush=True)
    best_k, best_r = None, 1e9
    for k in keys:
        pred = np.concatenate([g[1][k] for g in good])
        r = rmse(pred - truth)
        tag = "  <- naive 128-avg" if k == 'pf_mean' else "  (likelihood-weighted)"
        print(f"  {k:22s} RMSE={r:8.3f}{tag}", flush=True)
        if k != 'pf_mean' and r < best_r:
            best_r, best_k = r, k
    pm = rmse(np.concatenate([g[1]['pf_mean'] for g in good]) - truth)

    print("\n=== GATE VERDICT ===", flush=True)
    print(f"  naive 128-seed average (pf_mean) = {pm:.3f}  vs single-seed 14.37  -> averaging gain {14.37-pm:+.2f}", flush=True)
    print(f"  best likelihood-weighted scale = {best_k} -> {best_r:.3f}  -> selection-over-averaging {pm-best_r:+.2f}", flush=True)
    if best_r <= 10.5:
        print("  >> PASS: ensembled PF is competitive with the GBM. De-aliasing via likelihood", flush=True)
        print("     selection is REAL -> BUILD the output-blend (re-fit weight on OUR OOF) + gate vs 10.356.", flush=True)
    elif best_r <= 12.5:
        print("  >> PARTIAL: better than single-seed but not GBM-competitive. A 0.7 weight will NOT", flush=True)
        print("     transfer to our stack; at best a small output-blend (~0.1). Marginal lever.", flush=True)
    else:
        print("  >> FAIL: even the likelihood-weighted multi-scale ensemble stays ~single-seed level.", flush=True)
        print("     De-aliasing did NOT deliver on our data. The notebooks' 0.7 reflects a weaker GBM", flush=True)
        print("     or public-tuning, not transferable signal. DROP the PF-dominant lever -> harden 10.122.", flush=True)
    print("PFREAL DONE", flush=True)


if __name__ == "__main__":
    main()
