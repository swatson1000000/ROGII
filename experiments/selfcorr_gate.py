"""pilkwang selfcorr_* (prefix GR self-correlation) + cheap LGB gate on the nouk-231 base.

THE ONE feature family flagged portable in plan.md (2026-06-06 "portable secondary gates") that was
NEVER gated: within-well analog lookup -- match each eval row's windowed GR signature against the
well's OWN LABELED PREFIX windows and read off that prefix row's TVT. NOT typewell matching (all 5
existing matcher families slide vs the typewell), NOT spatial/density-coupled (single-well, own data
only -- the PF-transfer-friendly category). Code ported VERBATIM from
/tmp/rogii_new/pilkwang_rogii-eda-target-free-alignment-for-tvt (gr_window_signature +
selfcorr_prefix_tvt_features; hw=15, stride=3, 5-dim signature, NN top-2).

5 new features (drift-space delta like every *_d in the build; absolute selfcorr_tvt NOT used):
    selfcorr_d        selfcorr_tvt - last_known_tvt (the drift estimate)
    selfcorr_score    exp(-dist/2.5) match confidence
    selfcorr_trust    score * prefix-length confidence
    selfcorr_top2gap  NN2-NN1 distance gap (ambiguity)
    selfcorr_vs_sc    disagreement vs existing sc15_d (orthogonality signal)

GATE (pre-registered, same protocol/folds/params as super28/lever gates -> comparable):
  delta <= -0.05            -> real -> full 6-model retrain + stack OOF gate
  -0.05 < delta <= -0.02    -> marginal: proceed only if stack OOF confirms <= -0.03
  > -0.02                   -> dead, STOP (don't rationalize; aliasing within-well is the likely mode)

Run: nohup python -u experiments/selfcorr_gate.py > log/selfcorr_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
FEAT_OUT = ROOT / "data/processed/selfcorr_feats.parquet"
UK = ["tvt_uk_d", "uk_ancc", "uk_vs_dense"]
NEW_COLS = ["selfcorr_d", "selfcorr_score", "selfcorr_trust", "selfcorr_top2gap"]  # _vs_sc post-merge
N_SPLITS, SPLIT_SEED = 5, 42
LGB_PARAMS = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
    min_child_samples=10, min_child_weight=0.5, n_estimators=5000, n_jobs=-1,
    reg_alpha=2.03, reg_lambda=87.28, subsample=0.645, subsample_freq=1,
    colsample_bytree=0.821, objective="regression", metric="rmse", verbose=-1,
    device_type="cuda", max_bin=255, random_state=42)


# ---------- VERBATIM pilkwang (rogii-eda-target-free-alignment-for-tvt.py L1786+) ----------
def gr_window_signature(values, half_window=15):
    values = pd.Series(values, dtype='float64').interpolate(limit_direction='both')
    fallback = float(values.dropna().median()) if values.notna().any() else 0.0
    values = values.fillna(fallback)
    window = max(3, int(half_window) * 2 + 1)
    roll = values.rolling(window, center=True, min_periods=max(3, window // 3))
    mean = roll.mean()
    std = roll.std().fillna(0.0)
    rng = (roll.max() - roll.min()).fillna(0.0)
    grad = values.diff().rolling(window, center=True, min_periods=max(3, window // 3)).mean().fillna(0.0)
    center = values
    sig = np.column_stack([mean.to_numpy(dtype=float), std.to_numpy(dtype=float),
                           rng.to_numpy(dtype=float), grad.to_numpy(dtype=float),
                           center.to_numpy(dtype=float)])
    return sig.astype(np.float32)


def selfcorr_prefix_tvt_features(prefix_gr, prefix_tvt, tail_gr, last_known_tvt,
                                 half_window=15, stride=3):
    n_tail = len(tail_gr)
    empty = {
        'selfcorr_tvt': np.full(n_tail, np.nan, dtype=np.float32),
        'selfcorr_delta': np.full(n_tail, np.nan, dtype=np.float32),
        'selfcorr_score': np.full(n_tail, np.nan, dtype=np.float32),
        'selfcorr_trust': np.full(n_tail, np.nan, dtype=np.float32),
        'selfcorr_top2_gap': np.full(n_tail, np.nan, dtype=np.float32),
    }
    prefix_gr = np.asarray(prefix_gr, dtype=float)
    prefix_tvt = np.asarray(prefix_tvt, dtype=float)
    tail_gr = np.asarray(tail_gr, dtype=float)
    valid_prefix = np.isfinite(prefix_gr) & np.isfinite(prefix_tvt)
    if n_tail == 0 or valid_prefix.sum() < max(30, half_window * 2):
        return empty

    prefix_sig_all = gr_window_signature(prefix_gr, half_window=half_window)
    tail_sig = gr_window_signature(tail_gr, half_window=half_window)
    candidates = np.flatnonzero(valid_prefix)
    candidates = candidates[::max(1, int(stride))]
    if len(candidates) < 5:
        return empty
    prefix_sig = prefix_sig_all[candidates]
    prefix_tvt_candidates = prefix_tvt[candidates]
    good = np.isfinite(prefix_sig).all(axis=1) & np.isfinite(prefix_tvt_candidates)
    if good.sum() < 5:
        return empty
    prefix_sig = prefix_sig[good]
    prefix_tvt_candidates = prefix_tvt_candidates[good]

    center = np.nanmedian(prefix_sig, axis=0)
    scale = np.nanstd(prefix_sig, axis=0)
    scale = np.where(~np.isfinite(scale) | (scale < 1e-6), 1.0, scale)
    prefix_z = (prefix_sig - center) / scale
    tail_z = (tail_sig - center) / scale
    finite_tail = np.isfinite(tail_z).all(axis=1)

    dist = np.full((n_tail, min(2, len(prefix_z))), np.nan, dtype=np.float32)
    nn_idx = np.full((n_tail, min(2, len(prefix_z))), -1, dtype=int)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(2, len(prefix_z)), algorithm='auto')
    nn.fit(prefix_z)
    d, idx = nn.kneighbors(tail_z[finite_tail])
    dist[finite_tail, :d.shape[1]] = d.astype(np.float32)
    nn_idx[finite_tail, :idx.shape[1]] = idx

    best_valid = nn_idx[:, 0] >= 0
    sc_tvt = np.full(n_tail, np.nan, dtype=np.float32)
    sc_tvt[best_valid] = prefix_tvt_candidates[nn_idx[best_valid, 0]].astype(np.float32)
    best_dist = dist[:, 0]
    score = np.exp(-np.clip(best_dist, 0.0, 20.0) / 2.5).astype(np.float32)
    score[~best_valid] = np.nan
    if dist.shape[1] >= 2:
        top2_gap = (dist[:, 1] - dist[:, 0]).astype(np.float32)
        top2_gap[~np.isfinite(top2_gap)] = np.nan
    else:
        top2_gap = np.full(n_tail, np.nan, dtype=np.float32)
    prefix_len_conf = np.clip(valid_prefix.sum() / 250.0, 0.0, 1.0)
    trust = np.clip(score * prefix_len_conf, 0.0, 1.0).astype(np.float32)
    return {
        'selfcorr_tvt': sc_tvt,
        'selfcorr_delta': sc_tvt - float(last_known_tvt),
        'selfcorr_score': score,
        'selfcorr_trust': trust,
        'selfcorr_top2_gap': top2_gap,
    }
# ---------- end verbatim ----------


def one_well(wid, hp, ids, ri):
    h = pd.read_csv(hp, usecols=lambda c: c in ("GR", "TVT_input"))
    kn = h["TVT_input"].notna().to_numpy()
    n = len(ri)
    zero = pd.DataFrame({"id": ids, "selfcorr_d": np.zeros(n, np.float32),
                         "selfcorr_score": np.zeros(n, np.float32),
                         "selfcorr_trust": np.zeros(n, np.float32),
                         "selfcorr_top2gap": np.zeros(n, np.float32)})
    if kn.sum() < 40:
        return zero
    gr = h["GR"].to_numpy(np.float64)
    ktvt = h["TVT_input"].to_numpy(np.float64)[kn]
    f = selfcorr_prefix_tvt_features(gr[kn], ktvt, gr[ri], float(ktvt[-1]))
    return pd.DataFrame({"id": ids,
                         "selfcorr_d": f["selfcorr_delta"],
                         "selfcorr_score": f["selfcorr_score"],
                         "selfcorr_trust": f["selfcorr_trust"],
                         "selfcorr_top2gap": f["selfcorr_top2_gap"]})


def main():
    print(">> load nouk-231 matrix (frontier_ens minus UK)...", flush=True)
    tr = pd.read_parquet(ROOT / "data/processed/frontier_ens/train_feats.parquet")
    tr = tr.drop(columns=[c for c in UK if c in tr.columns])

    tf = tr[["well", "id"]].copy()
    tf["_ri"] = tf["id"].str.rsplit("_", n=1).str[-1].astype(int)
    paths = {os.path.basename(p).replace("__horizontal_well.csv", ""): p
             for p in glob.glob(str(RAW / "*__horizontal_well.csv"))}
    grp = {w: g for w, g in tf.groupby("well", sort=False)}
    wells = [w for w in tf["well"].unique() if w in paths]
    print(f">> building selfcorr feats over {len(wells)} wells...", flush=True)
    res = Parallel(n_jobs=12, backend="loky")(
        delayed(one_well)(w, paths[w], grp[w]["id"].to_numpy(), grp[w]["_ri"].to_numpy()) for w in wells)
    sc = pd.concat(res, ignore_index=True)
    sc.to_parquet(FEAT_OUT)
    print(f"   {len(sc):,} rows -> {FEAT_OUT}", flush=True)

    tr = tr.merge(sc, on="id", how="left")
    tr["selfcorr_vs_sc"] = (tr["selfcorr_d"] - tr["sc15_d"]).astype(np.float32)
    NEW = NEW_COLS + ["selfcorr_vs_sc"]
    nanrate = tr[NEW].isna().mean()
    print(f"   NaN rates pre-fill: {nanrate.round(4).to_dict()}", flush=True)
    for c in NEW:
        tr[c] = tr[c].fillna(0.0).astype(np.float32)

    # standalone quality of the raw estimator (context, NOT the gate)
    y = tr["target"].to_numpy(np.float32)
    v = tr["selfcorr_d"].to_numpy(np.float32)
    print(f"   selfcorr_d standalone RMSE vs target = {np.sqrt(np.mean((v - y) ** 2)):.3f} (null≈15.9)", flush=True)

    base = [c for c in tr.columns if c not in {"well", "id", "target"} and c not in NEW]
    assert len(base) == 231, f"base is {len(base)}"
    rng = np.random.RandomState(SPLIT_SEED); uw = tr["well"].unique().copy(); rng.shuffle(uw)
    fold_of = {w: i % N_SPLITS for i, w in enumerate(uw)}
    wf = tr["well"].map(fold_of).to_numpy()
    splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]

    def run(cols, tag):
        oof = np.zeros(len(tr), np.float32)
        for fold, (a, b) in enumerate(splits):
            dtr = lgb.Dataset(tr.iloc[a][cols], label=y[a])
            dva = lgb.Dataset(tr.iloc[b][cols], label=y[b], reference=dtr)
            m = lgb.train(LGB_PARAMS, dtr, valid_sets=[dva], num_boost_round=5000,
                          callbacks=[lgb.early_stopping(125, verbose=False)])
            oof[b] = m.predict(tr.iloc[b][cols], num_iteration=m.best_iteration)
        r = float(np.sqrt(np.mean((oof - y) ** 2)))
        print(f"   [{tag:10s}] OOF = {r:.4f}  ({len(cols)} feats)", flush=True)
        return r, oof

    print(">> gate: base231 vs +selfcorr5 ...", flush=True)
    r_base, oof_base = run(base, "base231")
    r_sc, _ = run(base + NEW, "+selfcorr5")

    resid = oof_base - y
    print("\n=== feature diagnostics ===", flush=True)
    for c in NEW:
        v = tr[c].to_numpy(np.float32)
        ct = float(np.corrcoef(v, y)[0, 1]) if np.std(v) > 0 else 0.0
        cr = float(np.corrcoef(v, resid)[0, 1]) if np.std(v) > 0 else 0.0
        print(f"  {c:16s} corr(target)={ct:+.4f}  corr(residual)={cr:+.4f}", flush=True)

    d = r_sc - r_base
    print("\n=== SELFCORR GATE VERDICT ===", flush=True)
    print(f"  base231 {r_base:.4f}   +selfcorr5 {r_sc:.4f}   delta = {d:+.4f}", flush=True)
    if d <= -0.05:
        print("  >> REAL: proceed to full 6-model retrain + stack OOF gate.", flush=True)
    elif d <= -0.02:
        print("  >> MARGINAL: proceed only if stack OOF confirms <= -0.03.", flush=True)
    else:
        print("  >> DEAD: within-well analog lookup is redundant/aliased on this base. STOP.", flush=True)
    print("SELFCORR GATE DONE", flush=True)


if __name__ == "__main__":
    main()
