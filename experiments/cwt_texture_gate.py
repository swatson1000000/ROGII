"""#4 (research_geosteering_6ft.md §6.4): CWT/DWT detail-band GR-texture features + cheap LGB gate.

The 222-feat build's NCC (sc8/15/25) aligns eval GR vs the well's own known-prefix GR with per-window
amplitude normalization. #4's claim: a SCALE-ISOLATED detail band (CWT/DWT detail coeffs) is "a different
similarity surface than raw-amplitude NCC" -> potentially orthogonal. Doc rates it <=0.1 ft / likely
redundant; this is the "one cheap shot as a feature" it recommends.

Detail band = high-pass (gr - rolling_mean(gr, W_TREND)) -> isolates fine stratigraphic texture, drops the
broad level the raw NCC already uses. Then the SAME multi_scale_ncc alignment on the detail band.

4 new features (leak-safe: eval GR observed at inference; ktvt/kgr = known prefix only -> mirrors sc*_d):
    dwt_ncc_d    detail-band NCC TVT estimate (softmax-ensembled over scales) minus last_tvt
    dwt_ncc_sc   its match score (confidence)
    dwt_vs_sc    disagreement vs the existing raw sc15_d (orthogonality signal)
    gr_detail_std per-row local texture amplitude (rolling std of the detail band)

GATE (same as dip): single-LGB GKF-5 seed42, base-222 vs +4. delta <= -0.05 -> real, proceed to retrain;
~flat (|d|<0.05) or regress -> absorbed/redundant, STOP (would only dilute through the 0.57 PF blend).

Run: nohup python -u experiments/cwt_texture_gate.py > log/cwt_gate_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
FR = ROOT / "data/processed/frontier_seeded"
OUTDIR = ROOT / "data/processed/frontier_cwt"; OUTDIR.mkdir(parents=True, exist_ok=True)
FEAT_OUT = ROOT / "data/processed/cwt_feats.parquet"
NEW_COLS = ["dwt_ncc_d", "dwt_ncc_sc", "gr_detail_std"]      # dwt_vs_sc added post-merge (needs sc15_d)
W_TREND = 31                                                  # high-pass trend window (detail = gr - trend)
N_SPLITS, SPLIT_SEED = 5, 42
LGB_PARAMS = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
    min_child_samples=10, min_child_weight=0.5, n_estimators=5000, n_jobs=-1,
    reg_alpha=2.03, reg_lambda=87.28, subsample=0.645, subsample_freq=1,
    colsample_bytree=0.821, objective="regression", metric="rmse", verbose=-1,
    device_type="cuda", max_bin=255, random_state=42)


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    """VERBATIM from rogii_frontier_inference.py L604-625 (align hgr windows vs known-zone kgr)."""
    out = []
    for hw in hws:
        win = 2 * hw + 1; nk = len(kgr); nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32); M = len(sts)
        if M == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode='edge')
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1); scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3. * scores); sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return sc_ens, np.max(scores, 1).astype(np.float32)


def detail(g, w=W_TREND):
    s = pd.Series(g)
    s = s.interpolate(limit_direction="both").fillna(float(np.nanmean(g)) if np.isfinite(np.nanmean(g)) else 0.0)
    trend = s.rolling(w, center=True, min_periods=1).mean()
    return (s - trend).values.astype(np.float32)


def one_well(wid, hp, ids, ri):
    h = pd.read_csv(hp, usecols=lambda c: c in ("GR", "TVT_input", "TVT"))
    kn = h["TVT_input"].notna().to_numpy()
    if kn.sum() < 40:
        n = len(ri)
        return pd.DataFrame({"id": ids, "dwt_ncc_d": np.zeros(n, np.float32),
                             "dwt_ncc_sc": np.zeros(n, np.float32), "gr_detail_std": np.zeros(n, np.float32)})
    gr = h["GR"].to_numpy(np.float64)
    gr_det = detail(gr)                                   # detail band over full trajectory
    ktvt = h["TVT_input"].to_numpy(np.float64)[kn]
    last_tvt = float(ktvt[-1])
    kdet = gr_det[kn]                                     # known-zone detail band
    hdet = gr_det[ri]                                     # eval-row detail band
    tvt_est, score = multi_scale_ncc(kdet, ktvt.astype(np.float32), hdet)
    # local texture amplitude at each eval row
    det_std = pd.Series(gr_det).rolling(15, center=True, min_periods=1).std().fillna(0.).values[ri].astype(np.float32)
    return pd.DataFrame({"id": ids,
                         "dwt_ncc_d": (tvt_est - np.float32(last_tvt)).astype(np.float32),
                         "dwt_ncc_sc": score,
                         "gr_detail_std": det_std})


def build_feats(tf, raw_dir, label, n_jobs=12):
    tf = tf.copy(); tf["_ri"] = tf["id"].str.rsplit("_", n=1).str[-1].astype(int)
    paths = {os.path.basename(p).replace("__horizontal_well.csv", ""): p
             for p in glob.glob(str(raw_dir / "*__horizontal_well.csv"))}
    grp = {w: g for w, g in tf.groupby("well", sort=False)}
    wells = [w for w in tf["well"].unique() if w in paths]
    print(f"   [{label}] {len(wells)} wells, computing detail-band NCC...", flush=True)
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(one_well)(w, paths[w], grp[w]["id"].to_numpy(), grp[w]["_ri"].to_numpy()) for w in wells)
    out = pd.concat(res, ignore_index=True)
    print(f"   [{label}] {len(out):,} rows", flush=True)
    return out


def run_oof(tr, cols, y, splits, tag):
    oof = np.zeros(len(tr), np.float32)
    for fold, (a, b) in enumerate(splits):
        dtr = lgb.Dataset(tr.iloc[a][cols], label=y[a])
        dva = lgb.Dataset(tr.iloc[b][cols], label=y[b], reference=dtr)
        m = lgb.train(LGB_PARAMS, dtr, valid_sets=[dva], num_boost_round=5000,
                      callbacks=[lgb.early_stopping(125, verbose=False)])
        oof[b] = m.predict(tr.iloc[b][cols], num_iteration=m.best_iteration)
        print(f"   [{tag}] fold {fold}: {np.sqrt(np.mean((oof[b]-y[b])**2)):.4f} it={m.best_iteration}", flush=True)
    r = float(np.sqrt(np.mean((oof - y) ** 2)))
    print(f"   [{tag}] OOF = {r:.4f}", flush=True)
    return r, oof


def main():
    print(">> load train_feats + build CWT-texture feats...", flush=True)
    tr = pd.read_parquet(FR / "train_feats.parquet")
    cwt_tr = build_feats(tr[["well", "id"]], RAW, "train")
    cwt_tr.to_parquet(FEAT_OUT)
    tr = tr.merge(cwt_tr, on="id", how="left")
    tr["dwt_vs_sc"] = (tr["dwt_ncc_d"] - tr["sc15_d"]).astype(np.float32)
    NEW = NEW_COLS + ["dwt_vs_sc"]
    for c in NEW:
        tr[c] = tr[c].fillna(0.0).astype(np.float32)
    print(f"   merged {tr.shape}", flush=True)
    tr.to_parquet(OUTDIR / "train_feats.parquet")

    base_cols = [c for c in tr.columns if c not in {"well", "id", "target"} and c not in NEW]
    cwt_cols = base_cols + NEW
    y = tr["target"].to_numpy(np.float32)
    rng = np.random.RandomState(SPLIT_SEED); uw = tr["well"].unique().copy(); rng.shuffle(uw)
    fold_of = {w: i % N_SPLITS for i, w in enumerate(uw)}
    wf = tr["well"].map(fold_of).to_numpy()
    splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]

    print(f">> base ({len(base_cols)} feats) ...", flush=True)
    r_base, oof_base = run_oof(tr, base_cols, y, splits, "base")
    print(f">> +cwt ({len(cwt_cols)} feats) ...", flush=True)
    r_cwt, _ = run_oof(tr, cwt_cols, y, splits, "+cwt")

    resid = oof_base - y
    print("\n=== feature diagnostics (corr with target / base-OOF residual) ===", flush=True)
    for c in NEW:
        v = tr[c].to_numpy(np.float32)
        ct = float(np.corrcoef(v, y)[0, 1]) if np.std(v) > 0 else 0.0
        cr = float(np.corrcoef(v, resid)[0, 1]) if np.std(v) > 0 else 0.0
        print(f"  {c:16s} std={np.std(v):.4g}  corr(target)={ct:+.4f}  corr(residual)={cr:+.4f}", flush=True)

    d = r_cwt - r_base
    print("\n=== CHEAP CWT-TEXTURE GATE VERDICT ===", flush=True)
    print(f"  base LGB-222 OOF = {r_base:.4f}   +cwt LGB-{len(cwt_cols)} OOF = {r_cwt:.4f}   delta = {d:+.4f}", flush=True)
    if d <= -0.05:
        print("  >> REAL SIGNAL: detail-band texture is orthogonal -> proceed to full retrain.", flush=True)
    elif d <= 0.003:
        print("  >> ~FLAT: detail-band NCC is REDUNDANT with raw NCC/DTW (as the doc predicted). STOP.", flush=True)
    else:
        print("  >> REGRESSION: texture feats HURT the OOF. STOP.", flush=True)
    print("CWT GATE DONE", flush=True)


if __name__ == "__main__":
    main()
