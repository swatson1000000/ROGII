"""GATE: standalone OOF of the 14-config beam ensemble + its output-blend gain on the 8.269 blend.

The analog of pf_real_gate.py, for the OTHER half of the public ravaghi/pilkwang pipeline.
blend_candidate_scan.py showed the EXISTING 7-config beam columns (beam_mean_d etc.) are ~15.8 ft
standalone and buy ~0 on the blend. But that's the SIMPLE-MEAN, single-config-derived version --
the PF was likewise worthless single-seed (14.37) and only became +1.18 after the 128-seed
LIKELIHOOD-WEIGHTED ensemble. This builds the un-built artifact and tests the same way.

Two ensembles, both reported (mirrors pf_mean vs pf_scale_*):
  beam_mean        = ravaghi's run_beam_ensemble VERBATIM (plain mean of 14 configs)
  beam_lik_{scale} = the 14 tracks softmax-weighted by a COMMON GR-emission loglik (the PF's
                     gs-normalized residual, scales {3,5,8,12}) -> the de-aliased version with a shot

beam_search + BEAM_CONFIGS copied VERBATIM from
/tmp/rogii_new/ravaghi_.../wellbore-geology-prediction-ridge.py (L50-65, L199-284).
No leak: beam + likelihood use only GR (full trace, observed) + typewell + TVT_input prefix; true
TVT is used ONLY to score. Per-well, no training -> standalone RMSE over all train eval rows IS OOF.

Alignment to the GBM OOF + PF pkl reuses blend_candidate_scan.py's verified sorted(glob)+skip
replay (same process() filter -> same 773-well order). Refuses to blend any well that fails the
per-well true-TVT check.

Run: cd ROOT && nohup python -u experiments/beam_real_gate.py \
       > log/beam_real_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os, json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from joblib import Parallel, delayed

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
FR = ROOT / "data/processed/frontier_seeded"
MODELS = ROOT / "models/frontier"
SCALES = (3.0, 5.0, 8.0, 12.0)
PF_KEY = "pf_scale_12"
W_PF = 0.44

# ---- VERBATIM from ravaghi source (BEAM_CONFIGS L50-65, beam_search L199-260) ----
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10,  8.0,  64.0, 2),
    ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5),
    (20,  4.0,  36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2),
    (10, 50.0, 400.0, 0),
]


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])
    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()
    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC = mc * np.array([2., 1., 0., 1., 2.])
    bidx = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.
    bn = 1
    result = np.zeros(n)
    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)
        gr_e = (gv - tw_gr[ci])**2 / es
        tot = bcost[:bn, None] + gr_e + MC[None, :]
        tot = np.where(valid, tot, np.inf)
        ni_f = ni.flatten(); tot_f = tot.flatten(); vf = valid.flatten()
        ni_f = ni_f[vf]; tot_f = tot_f[vf]
        order = np.argsort(tot_f)
        ni_s = ni_f[order]; tot_s = tot_f[order]
        _, first = np.unique(ni_s, return_index=True)
        ni_u = ni_s[first]; tot_u = tot_s[first]
        kept = min(bs, len(ni_u))
        top = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top = top[np.argsort(tot_u[top])]
        bidx[:kept] = ni_u[top]; bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1]; bcost[kept:] = np.inf
        bn = kept
        result[step] = tw_tvt[bidx[0]]
    return result
# ---- end verbatim ----


def run_beam_ensemble_full(hw, tw):
    """Return (beam_mean, {beam_lik_scale: track}, track_for_each_config_unused).
    Builds all 14 tracks; combines by plain mean (ravaghi) AND by common-likelihood softmax."""
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    out_init = hw['TVT_input'].values.astype(float).copy()
    if len(ev) == 0:
        return out_init.copy(), {f'beam_lik_{s:g}': out_init.copy() for s in SCALES}

    last_tvt = float(kn.iloc[-1]['TVT_input'])
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    gr_all = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    hgr = gr_all[ev.index]

    # common GR-emission noise scale gs (PF's, from the known zone)
    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))

    tracks = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
              for (bs, mc, es, r) in BEAM_CONFIGS]
    track_arr = np.stack(tracks, 0)                          # (14, n_eval)

    # common loglik per config: -0.5 * sum((hgr - tw_gr@track)^2 / gs^2), capped like the PF
    liks = np.empty(len(tracks))
    for j, trk in enumerate(tracks):
        eg = np.interp(trk, tw_tvt, tw_gr)
        d = (hgr - eg) / gs
        liks[j] = -0.5 * float(np.sum(np.minimum(d**2, 600.)))
    liks_n = liks - liks.max()

    beam_mean = track_arr.mean(0)
    lik_out = {}
    for scale in SCALES:
        wt = np.exp(liks_n / float(scale)); wt /= wt.sum()
        lik_out[f'beam_lik_{scale:g}'] = (wt[:, None] * track_arr).sum(0)

    out_mean = out_init.copy(); out_mean[list(ev.index)] = beam_mean
    out_lik = {}
    for k, v in lik_out.items():
        o = out_init.copy(); o[list(ev.index)] = v; out_lik[k] = o
    return out_mean, out_lik


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
        out_mean, out_lik = run_beam_ensemble_full(hw, tw)
        preds = {'beam_mean': out_mean[ev_mask]}
        for k, v in out_lik.items():
            preds[k] = v[ev_mask]
        return (true_tvt[ev_mask], preds)
    except Exception as e:
        return ('ERR', f"{wid}: {e}")


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def main():
    paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    print(f">> {len(paths)} wells; 14-config beam ensemble, parallel ...", flush=True)
    results = Parallel(n_jobs=14, verbose=5)(delayed(process)(p) for p in paths)
    import joblib
    save = MODELS / "beam_real_results.pkl"
    joblib.dump(results, save)
    print(f">> saved raw results -> {save}", flush=True)

    def is_err(r): return r is not None and isinstance(r[0], str) and r[0] == 'ERR'
    errs = [r[1] for r in results if is_err(r)]
    for e in errs[:10]:
        print(f"   ERR {e}", flush=True)
    good = [r for r in results if r is not None and not is_err(r)]
    print(f">> {len(good)} wells scored ({len(errs)} errors)", flush=True)

    keys = list(good[0][1].keys())
    truth = np.concatenate([g[0] for g in good])
    print(f"\n=== STANDALONE OOF (eval rows = {len(truth):,}) — RMSE of TVT_pred vs TVT_true ===", flush=True)
    print("  anchors:  null 15.910 | frontier GBM 10.356 | 128-seed PF 10.993 | current blend 9.169", flush=True)
    best_k, best_r = None, 1e9
    for k in keys:
        pred = np.concatenate([g[1][k] for g in good])
        r = rmse(pred - truth)
        tag = "  <- plain mean (ravaghi)" if k == 'beam_mean' else "  (likelihood-weighted)"
        print(f"  {k:18s} RMSE={r:8.3f}{tag}", flush=True)
        if k != 'beam_mean' and r < best_r:
            best_r, best_k = r, k
    bm = rmse(np.concatenate([g[1]['beam_mean'] for g in good]) - truth)
    print(f"\n  plain-mean beam = {bm:.3f}  |  best likelihood-weighted = {best_k} -> {best_r:.3f}"
          f"  (selection-over-averaging {bm-best_r:+.2f})", flush=True)

    # ---------------- output-blend test vs the CURRENT 8.269 blend ----------------
    print("\n=== OUTPUT-BLEND TEST vs the 8.269 blend (alignment from blend_candidate_scan) ===", flush=True)
    df = pd.read_parquet(FR / "train_feats.parquet", columns=["well", "id", "last_known_tvt", "target"])
    blend = json.load(open(MODELS / "blend_frontier.json"))
    coef = np.array(blend["ridge_coef"], np.float64)
    oofs = np.column_stack([np.load(MODELS / f"oof_{k}.npy").astype(np.float64) for k in blend["keys"]])
    blended_drift = oofs @ coef
    target = df["target"].to_numpy(np.float64); lastk = df["last_known_tvt"].to_numpy(np.float64)
    gbm_resid = blended_drift - target
    true_tvt_all = target + lastk
    df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)

    # replay PF well order (identical skip logic) -> map results to wells
    pf_wells = []
    for p in paths:
        wid = os.path.basename(p).replace("__horizontal_well.csv", "")
        hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input"))
        if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
            continue
        pf_wells.append(wid)
    pf_results = joblib.load(MODELS / "pf_real_results.pkl")
    pf_good = [r for r in pf_results if r is not None and not is_err(r)]
    assert len(pf_good) == len(pf_wells) == len(good), "well-order mismatch -> abort blend test"

    grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}
    gbm_keep, pf_keep, beam_keep = [], [], {k: [] for k in keys if k != 'beam_mean'}
    beam_keep['beam_mean'] = []
    n_ok = n_bad = 0
    for wid, (pf_truth, pf_preds), (bm_truth, bm_preds) in zip(pf_wells, pf_good, good):
        sub = grp.get(wid)
        if sub is None or len(sub) != len(pf_truth):
            n_bad += 1; continue
        t_tvt = sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64)
        if not (np.allclose(t_tvt, pf_truth, atol=1e-3) and np.allclose(t_tvt, bm_truth, atol=1e-3)):
            n_bad += 1; continue
        n_ok += 1
        gbm_keep.append(sub.index.to_numpy())
        pf_keep.append(pf_preds[PF_KEY].astype(np.float64) - pf_truth)
        for k in beam_keep:
            beam_keep[k].append(bm_preds[k].astype(np.float64) - bm_truth)
    pos = np.concatenate(gbm_keep)
    g_res = gbm_resid[pos]; p_res = np.concatenate(pf_keep)
    blend_res = (1 - W_PF) * g_res + W_PF * p_res
    fold = None
    rng = np.random.RandomState(42)
    uw = df["well"].unique().copy(); rng.shuffle(uw)
    fold_of = {w: i % 5 for i, w in enumerate(uw)}
    fold = np.array([fold_of[w] for w in df["well"].to_numpy()[pos]])
    print(f"   aligned wells ok={n_ok} bad={n_bad}; blend OOF={rmse(blend_res):.4f}", flush=True)

    def oof_gain(base, cand):
        base_rmse = rmse(base); out = base.copy(); ws = []
        for f in range(5):
            tr, va = fold != f, fold == f
            d = base[tr] - cand[tr]; den = np.dot(d, d)
            wf = float(np.dot(base[tr], d) / den) if den > 0 else 0.0
            out[va] = (1 - wf) * base[va] + wf * cand[va]; ws.append(wf)
        return base_rmse, base_rmse - rmse(out), float(np.mean(ws))

    print(f"\n  {'beam variant':18s}{'stand':>8}{'corrBlend':>10}{'wBlend':>8}{'gainBlend':>10}{'corrGBM':>9}{'gainGBM':>9}", flush=True)
    rows = []
    for k in ['beam_mean'] + [x for x in keys if x != 'beam_mean']:
        c_res = np.concatenate(beam_keep[k])
        stand = rmse(c_res)
        corrB = np.corrcoef(c_res, blend_res)[0, 1]
        corrG = np.corrcoef(c_res, g_res)[0, 1]
        _, gB, wB = oof_gain(blend_res, c_res)
        _, gG, _ = oof_gain(g_res, c_res)
        rows.append((k, gB))
        print(f"  {k:18s}{stand:>8.2f}{corrB:>10.3f}{wB:>8.2f}{gB:>+10.3f}{corrG:>9.3f}{gG:>+9.3f}", flush=True)

    rows.sort(key=lambda r: -r[1])
    bestk, bestg = rows[0]
    print("\n=== VERDICT ===", flush=True)
    print(f"  best beam-on-blend: {bestk}  gain={bestg:+.3f} ft", flush=True)
    if bestg >= 0.10:
        print("  BANKABLE: the beam ensemble IS a 2nd orthogonal output member -> add to the kernel as a", flush=True)
        print("  3-way blend (GBM + PF + beam), re-fit weights on OOF, gate, submit. Re-apply per-well seeding NA (beam is deterministic).", flush=True)
    elif bestg >= 0.03:
        print("  MARGINAL: below the ~0.23 OOF<->LB resolution -> free fold-in at best, not a standalone sub.", flush=True)
    else:
        print("  NULL: beam is redundant with the PF already in the blend (same sequential GR-tracker family).", flush=True)
        print("  Confirms the public recipe bundling PF+beam into ONE block. Averaging axis stays spent.", flush=True)
        print("  -> the live lever is the per-well SELECTOR (route PF/GBM by well), not a 3rd average member.", flush=True)
    print("BEAMREAL DONE", flush=True)


if __name__ == "__main__":
    main()
