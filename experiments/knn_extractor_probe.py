"""Extractability probe: KNN-SEEDED local GR extractor (plan sec 5, Phase-4 move).

Hypothesis: the existing beam/tw_diff GR-match features are seeded on
last_known_tvt (drift=0, prefix anchor), which drifts increasingly wrong downhole.
Seeding the typewell-GR comparison on the SPATIAL KNN estimate (fk_tvt_formula,
the -Z+ANCC+b_well plane-KNN TVT) instead places the comparison window near the
true TVT (~+-12 ft), where GR is sharp (oracle corr +0.72, ~+-4 ft). The GBM should
then be able to refine the ~12 ft KNN estimate within that tight window.

This builds, per hidden row:
  - knn_diff_<o> = gr - tw_gr(knn_abs + o)  for a fine offset grid (KNN-seeded)
  - knn_refined_drift = argmin_o |gr - tw_gr(knn_abs+o)| over o in +-20 ft (the
    GR-refined correction to the KNN seed) and knn_refined_cost (match quality)
  - knn_seed_resid = |gr - tw_gr(knn_abs)|  (o=0 match quality)

then compares single GPU-LGB OOF (78 base vs 78 + these), same shuffled GKF-5 as
ncc_feature_ablation / phase4_reexam (seed 42). GATE: a clearly negative delta
(> ~0.05 ft improvement) justifies the full build (LGBx3+XGB stack + kernel).
A null repeats the NCC outcome (redundant) and we move to CatBoost only.
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor

ROOT = Path("/home/swatson/work/kaggle/ROGII")
ART = ROOT / "data/processed/konbu"
N_SPLITS, SPLIT_SEED = 5, 42
OFF_GRID = np.arange(-20.0, 20.01, 0.5, dtype=np.float32)          # refine search
DIFF_OFFS = np.array([-20, -15, -10, -7, -5, -3, 0, 3, 5, 7, 10, 15, 20], dtype=np.float32)
LGB = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89, min_child_samples=10,
           min_child_weight=0.5, n_estimators=5000, reg_alpha=2.03, reg_lambda=87.28,
           subsample=0.645, subsample_freq=1, colsample_bytree=0.821, objective="regression",
           metric="rmse", verbose=-1, device_type="cuda", max_bin=255, seed=42)

# globals set in main(), inherited by workers via fork
_SEED_IS_ABS = None
_TW_DIR = None
_TW_PAT = None


def _find_typewell(wid):
    for pat in (f"{wid}__typewell.csv", f"{wid}_typewell.csv"):
        p = _TW_DIR / pat
        if p.exists():
            return p
    hits = glob.glob(str(_TW_DIR / f"{wid}*typewell*.csv"))
    return Path(hits[0]) if hits else None


def build_one(args):
    wid, sub = args  # sub: DataFrame for this well, cols [prediction_id,row_idx,gr,knn_abs]
    tp = _find_typewell(wid)
    if tp is None:
        return None
    t = pd.read_csv(tp)
    if "TVT" not in t.columns or "GR" not in t.columns:
        return None
    tw = t[["TVT", "GR"]].dropna().sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(dtype=np.float32)
    tw_gr = tw["GR"].to_numpy(dtype=np.float32)
    if len(tw_tvt) < 8:
        return None

    sub = sub.sort_values("row_idx")
    gr = sub["gr"].to_numpy(dtype=np.float32)
    knn_abs = sub["knn_abs"].to_numpy(dtype=np.float32)
    gr = np.where(np.isnan(gr), np.float32(np.nanmean(tw_gr)), gr)

    out = pd.DataFrame({"prediction_id": sub["prediction_id"].to_numpy()})

    # KNN-seeded offset diffs (vectorized np.interp over the fine grid)
    for o in DIFF_OFFS:
        samp = np.interp(knn_abs + o, tw_tvt, tw_gr).astype(np.float32)
        out[f"knn_diff_{int(o)}"] = (gr - samp).astype(np.float32)

    # local GR refinement: argmin_o |gr - tw_gr(knn_abs+o)|, o in OFF_GRID
    # build (nrow, len(OFF_GRID)) matrix of |gr - tw_gr(knn_abs+o)|
    grid = OFF_GRID[None, :]                                   # (1, G)
    query = knn_abs[:, None] + grid                            # (nrow, G)
    samp = np.interp(query.ravel(), tw_tvt, tw_gr).reshape(query.shape).astype(np.float32)
    cost = np.abs(gr[:, None] - samp)                          # (nrow, G)
    j = cost.argmin(axis=1)
    out["knn_refined_drift"] = OFF_GRID[j].astype(np.float32)  # correction to seed
    out["knn_refined_cost"] = cost[np.arange(len(j)), j].astype(np.float32)
    out["knn_seed_resid"] = np.abs(gr - np.interp(knn_abs, tw_tvt, tw_gr)).astype(np.float32)
    return out


def main():
    global _SEED_IS_ABS, _TW_DIR, _TW_PAT
    df = pd.read_parquet(ART / "train_feats.parquet")
    base_feats = [c for c in df.columns if c not in {"well", "prediction_id", "target"}]
    print(f"base matrix: {df.shape}, {len(base_feats)} feats", flush=True)

    # locate typewell dir
    for cand in (ROOT / "data/raw/train", ROOT / "data/raw", ROOT / "data/train"):
        if cand.exists() and glob.glob(str(cand / "*typewell*.csv")):
            _TW_DIR = cand
            break
    if _TW_DIR is None:
        raise SystemExit("typewell files not found under data/raw")
    print(f"typewell dir: {_TW_DIR}  (n={len(glob.glob(str(_TW_DIR/'*typewell*.csv')))})", flush=True)

    # seed semantics: absolute TVT (~1e4) vs drift (~0)?
    fk = df["fk_tvt_formula"].to_numpy()
    _SEED_IS_ABS = bool(np.nanmedian(np.abs(fk)) > 1000.0)
    print(f"fk_tvt_formula median|.|={np.nanmedian(np.abs(fk)):.1f} -> seed_is_abs={_SEED_IS_ABS}", flush=True)
    knn_abs = fk if _SEED_IS_ABS else (df["last_known_tvt"].to_numpy() + fk)
    df = df.assign(knn_abs=knn_abs.astype(np.float32))
    # sanity: knn_abs should correlate with true absolute TVT
    abs_true = df["last_known_tvt"].to_numpy() + df["target"].to_numpy()
    print(f"corr(knn_abs, abs_true)={np.corrcoef(df['knn_abs'], abs_true)[0,1]:.4f} "
          f"(seed imputation RMSE={np.sqrt(np.mean((df['knn_abs']-abs_true)**2)):.2f} ft)", flush=True)

    # build new features per well in parallel
    wells = df["well"].unique()
    groups = [(w, g[["prediction_id", "row_idx", "gr", "knn_abs"]].copy())
              for w, g in df.groupby("well", sort=False)]
    print(f">> building KNN-seeded GR features over {len(wells)} wells", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=14) as ex:
        for i, r in enumerate(ex.map(build_one, groups, chunksize=4)):
            if r is not None:
                parts.append(r)
            if (i + 1) % 150 == 0:
                print(f"   {i+1}/{len(groups)}", flush=True)
    new = pd.concat(parts, ignore_index=True)
    new_feats = [c for c in new.columns if c != "prediction_id"]
    print(f"   new features ({len(new_feats)}): {new_feats}", flush=True)

    m = df.merge(new, on="prediction_id", how="left")
    miss = m[new_feats[0]].isna().sum()
    print(f"   merged {m.shape}; rows missing new feats = {miss}", flush=True)
    m[new_feats] = m[new_feats].fillna(0.0)

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
    withk = oof(base_feats + new_feats, "BASE+KNNseed")
    print(f"\n>>> KNN-seeded GR extractor contributes: {base - withk:+.4f} ft to OOF", flush=True)
    print(">>> GATE: > +0.05 ft => worth the full build; ~0 => redundant (move to CatBoost)", flush=True)


if __name__ == "__main__":
    main()
