"""Re-audit probe #1: is multi-scale NCC dead, or was Phase-2 just the wrong frame?

Phase 2 killed NCC as a linear blend / drift-correlation (14.979->14.973). This tests
it the way the winning recipe uses every signal: as weak GBM FEATURES on top of the
strong konbu 78-feature base. Same method as the Phase-4 GR ablation.

Builds 9 multi-scale NCC features per hidden row (per-scale drift+score, softmax blend,
cross-scale disagreement), merges into the cached konbu matrix on prediction_id, and
compares single GPU-LGB OOF: 78 vs 78+NCC (same shuffled GroupKFold-5).
"""
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor

RAW = Path("/home/swatson/work/kaggle/ROGII/data/raw/train")
ART = Path("/home/swatson/work/kaggle/ROGII/data/processed/konbu")
N_SPLITS, SPLIT_SEED = 5, 42
HWS = (8, 15, 25)
LGB = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89, min_child_samples=10,
           min_child_weight=0.5, n_estimators=5000, reg_alpha=2.03, reg_lambda=87.28,
           subsample=0.645, subsample_freq=1, colsample_bytree=0.821, objective="regression",
           metric="rmse", verbose=-1, device_type="cuda", max_bin=255, seed=42)


def fill(a):
    return pd.Series(a).interpolate(limit_direction="both").bfill().ffill().to_numpy()


def ms_ncc_features(kgr, ktvt, hgr, anchor, hws=HWS, stride=3, temp=3.0):
    """Return per-row (blend_drift, scale_drifts[nh,S], scale_scores[nh,S])."""
    nh, nk = len(hgr), len(kgr)
    tvts, scores = [], []
    for hw in hws:
        win = 2 * hw + 1
        if nk < win + 1 or nh == 0:
            tvts.append(np.full(nh, anchor, np.float32)); scores.append(np.zeros(nh, np.float32)); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().to_numpy().astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().to_numpy().astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        ctr = sts + hw
        Cm = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]]
        Cn = (Cm - Cm.mean(1, keepdims=True)) / (Cm.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]]
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        tvts.append(ktvt[np.clip(ctr[best], 0, nk - 1)].astype(np.float32))
        scores.append(ncc.max(1).astype(np.float32))
    T = np.stack(tvts, 1); S = np.stack(scores, 1)            # (nh, n_scales)
    sw = np.exp(temp * S); sw /= sw.sum(1, keepdims=True) + 1e-9
    blend = (T * sw).sum(1).astype(np.float32)
    return blend - anchor, (T - anchor).astype(np.float32), S


def build_one(wid):
    p = RAW / f"{wid}__horizontal_well.csv"
    h = pd.read_csv(p)
    if "TVT_input" not in h.columns:
        return None
    mask = h["TVT_input"].isna().to_numpy()
    if not mask.any():
        return None
    ms = int(np.flatnonzero(mask)[0])
    if ms == 0:
        return None
    gr = fill(h["GR"].to_numpy())
    kgr = gr[:ms].astype(np.float32)
    ktvt = h["TVT_input"].to_numpy()[:ms].astype(np.float32)
    hgr = gr[ms:].astype(np.float32)
    anchor = float(h["TVT_input"].iloc[ms - 1])
    blend, sd, sc = ms_ncc_features(kgr, ktvt, hgr, anchor)
    hidden_idx = h.index[ms:]
    out = pd.DataFrame({"prediction_id": [f"{wid}_{i}" for i in hidden_idx]})
    out["ncc_blend_drift"] = blend
    for j, hw in enumerate(HWS):
        out[f"ncc{hw}_drift"] = sd[:, j]
        out[f"ncc{hw}_score"] = sc[:, j]
    out["ncc_score_max"] = sc.max(1)
    out["ncc_drift_std"] = sd.std(1)
    return out


def main():
    df = pd.read_parquet(ART / "train_feats.parquet")
    base_feats = [c for c in df.columns if c not in {"well", "prediction_id", "target"}]
    wids = sorted({pid.rsplit("_", 1)[0] for pid in df["prediction_id"]})
    print(f"base matrix: {df.shape}, {len(base_feats)} feats, {len(wids)} wells", flush=True)

    print(">> build multi-scale NCC features (parallel)", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=14) as ex:
        for i, r in enumerate(ex.map(build_one, wids, chunksize=4)):
            if r is not None:
                parts.append(r)
            if (i + 1) % 150 == 0:
                print(f"   {i+1}/{len(wids)}", flush=True)
    ncc = pd.concat(parts, ignore_index=True)
    ncc_feats = [c for c in ncc.columns if c != "prediction_id"]
    print(f"   ncc features: {ncc_feats}", flush=True)

    m = df.merge(ncc, on="prediction_id", how="left")
    miss = m[ncc_feats[0]].isna().sum()
    print(f"   merged: {m.shape}; rows missing NCC = {miss}", flush=True)
    m[ncc_feats] = m[ncc_feats].fillna(0.0)

    y = m["target"].to_numpy()
    rng = np.random.RandomState(SPLIT_SEED)
    w = m["well"].unique().copy(); rng.shuffle(w)
    fo = {x: i % N_SPLITS for i, x in enumerate(w)}
    wf = m["well"].map(fo).to_numpy()
    splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]

    def oof(cols, tag):
        o = np.zeros(len(m), np.float32)
        for tr, va in splits:
            dtr = lgb.Dataset(m.iloc[tr][cols], label=y[tr])
            dva = lgb.Dataset(m.iloc[va][cols], label=y[va], reference=dtr)
            mod = lgb.train(LGB, dtr, valid_sets=[dva], num_boost_round=LGB["n_estimators"],
                            callbacks=[lgb.early_stopping(125, verbose=False)])
            o[va] = mod.predict(m.iloc[va][cols], num_iteration=mod.best_iteration)
        r = float(np.sqrt(np.mean((o - y) ** 2)))
        print(f"[{tag}] OOF rmse={r:.4f} (#feats={len(cols)})", flush=True)
        return r

    base = oof(base_feats, "BASE 78")
    withn = oof(base_feats + ncc_feats, "BASE+NCC")
    print(f"\n>>> multi-scale NCC features contribute: {base - withn:+.4f} ft to OOF", flush=True)
    print(">>> (positive = NCC helps = Phase-2 'NCC dead' was the wrong frame)", flush=True)


if __name__ == "__main__":
    main()
