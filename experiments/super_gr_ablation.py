"""GR-matcher ablation: is the super-solution's OOF win real signal or overfit GR candy?

Backlog spec (PICK_UP_HERE): ablate full super-170 vs 170-minus-GR-matchers (drop the whole
GR-vs-typewell matching channel: PF / 7 beams / multi-scale NCC / 4 tw_diff families / gr_vs_tw
/ affine-cal / prefix-RMSE / matcher-vs-spatial disagreements; KEEP spatial backbone + trajectory
+ raw GR shape/rolling). Compare BOTH:
  - shuffled GroupKFold-5 (seed 42)  -> standard OOF + per-fold variance
  - spatial BLOCK-HOLDOUT (KMeans-5 on well centroids) -> OOD proxy (memory: single-well LOO
    over-credits density/GR-coupled estimators; the GP went 24.5 LOO -> 48 block -> LB regression)

Interpretation:
  * apparent GR gain  = OOF(ablated) - OOF(full)  under shuffled-GKF
  * OOD-robust GR gain = same delta under block-holdout
  * if the block-holdout gain is much smaller, and/or full has much higher block-holdout fold
    variance, the GR matchers are OOF candy that likely won't transfer to LB (the forum-post
    failure mode: Viterbi feats +0.46 OOF / +0.001 LB, fold var 0.44->0.87).

Same single LGB(cuda) config for every cell (relative comparison is what matters). Cheap probe,
not the gate model. Run on skynet AFTER super_train_lgb (shared GPU).
Run: nohup python -u experiments/super_gr_ablation.py > log/super_ablation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans

ROOT = Path("/home/swatson/work/kaggle/ROGII")
SP = ROOT / "data/processed/super"
RAW = ROOT / "data/raw/train"
N_SPLITS = 5

# lean-but-fair single LGB (relative full-vs-ablated comparison; not the 8000-it gate model)
LGB = dict(boosting_type="gbdt", num_leaves=127, learning_rate=0.05, min_child_samples=15,
           subsample=0.75, subsample_freq=1, colsample_bytree=0.75, reg_lambda=3.0, reg_alpha=0.05,
           objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
           device_type="cuda", max_bin=255)
N_EST, EARLY = 3000, 150

train_df = pd.read_parquet(SP / "train_feats.parquet")
all_feats = [c for c in train_df.columns if c not in {"well", "id", "target"}]
y = train_df["target"].to_numpy(np.float32)
wells = train_df["well"].to_numpy()

# ---- define the GR-matching channel to drop ----
DROP_PREFIX = ("pf_", "beam_", "sc8", "sc15", "sc25", "sc_", "tda", "tdbc", "tdsc", "tdpf", "hyb_", "signal_")
DROP_EXACT = {"gr_vs_tw", "gr_vs_slp", "cal_a", "cal_b", "pfx_rmse"}
def is_matcher(c):
    return c in DROP_EXACT or c.startswith(DROP_PREFIX)
gr_matchers = [c for c in all_feats if is_matcher(c)]
kept = [c for c in all_feats if not is_matcher(c)]
print(f">> total feats={len(all_feats)} | GR-matchers dropped={len(gr_matchers)} | kept={len(kept)}", flush=True)
print(f"   dropped: {gr_matchers}", flush=True)
print(f"   kept:    {kept}", flush=True)

# ---- two CV protocols ----
# (1) shuffled GroupKFold-5 seed 42 (matches the gate)
rng = np.random.RandomState(42)
uw = pd.unique(train_df["well"]).copy(); rng.shuffle(uw)
fold_of = {w: i % N_SPLITS for i, w in enumerate(uw)}
wf_shuf = np.array([fold_of[w] for w in wells])

# (2) spatial block-holdout: KMeans-5 on per-well median (X,Y)
cents = []
for w in uw:
    p = RAW / f"{w}__horizontal_well.csv"
    try:
        d = pd.read_csv(p, usecols=["X", "Y"]).dropna()
        cents.append((w, float(d["X"].median()), float(d["Y"].median())))
    except Exception:
        cents.append((w, np.nan, np.nan))
cdf = pd.DataFrame(cents, columns=["well", "x", "y"]).dropna()
xy = cdf[["x", "y"]].to_numpy()
xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-9)
km = KMeans(n_clusters=N_SPLITS, random_state=42, n_init=10).fit(xy)
block_of = dict(zip(cdf["well"], km.labels_))
# wells with missing centroid (rare) -> assign to block 0
wf_block = np.array([block_of.get(w, 0) for w in wells])
print(f">> block sizes (wells): {np.bincount(km.labels_).tolist()}", flush=True)


def run(feat_cols, wf, label):
    oof = np.zeros(len(train_df), np.float32)
    fold_rmse = []
    Xall = train_df[feat_cols]
    for f in range(N_SPLITS):
        tr = np.where(wf != f)[0]; va = np.where(wf == f)[0]
        dtr = lgb.Dataset(Xall.iloc[tr], label=y[tr])
        dva = lgb.Dataset(Xall.iloc[va], label=y[va], reference=dtr)
        m = lgb.train(LGB, dtr, valid_sets=[dva], num_boost_round=N_EST,
                      callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        oof[va] = m.predict(Xall.iloc[va], num_iteration=m.best_iteration)
        fr = float(np.sqrt(np.mean((oof[va] - y[va]) ** 2))); fold_rmse.append(fr)
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    fstd = float(np.std(fold_rmse))
    print(f"   [{label}] OOF={rmse:.4f}  fold_std={fstd:.4f}  folds={[round(x,3) for x in fold_rmse]}", flush=True)
    return {"oof": rmse, "fold_std": fstd, "folds": fold_rmse}


res = {}
print("\n>> shuffled GroupKFold-5:", flush=True)
res["shuf_full"] = run(all_feats, wf_shuf, "shuf full")
res["shuf_abl"] = run(kept, wf_shuf, "shuf ablated")
print("\n>> spatial block-holdout (KMeans-5):", flush=True)
res["block_full"] = run(all_feats, wf_block, "block full")
res["block_abl"] = run(kept, wf_block, "block ablated")

shuf_gain = res["shuf_abl"]["oof"] - res["shuf_full"]["oof"]
block_gain = res["block_abl"]["oof"] - res["block_full"]["oof"]
print("\n=== GR-MATCHER ABLATION SUMMARY ===", flush=True)
print(f"   shuffled-GKF : full {res['shuf_full']['oof']:.4f} (fstd {res['shuf_full']['fold_std']:.3f}) | "
      f"ablated {res['shuf_abl']['oof']:.4f} (fstd {res['shuf_abl']['fold_std']:.3f}) | "
      f"GR gain {shuf_gain:+.4f}", flush=True)
print(f"   block-holdout: full {res['block_full']['oof']:.4f} (fstd {res['block_full']['fold_std']:.3f}) | "
      f"ablated {res['block_abl']['oof']:.4f} (fstd {res['block_abl']['fold_std']:.3f}) | "
      f"GR gain {block_gain:+.4f}", flush=True)
print(f"   --> GR gain shuffled {shuf_gain:+.4f}  vs  block-holdout {block_gain:+.4f}", flush=True)
print(f"   --> full fold_std shuffled {res['shuf_full']['fold_std']:.3f} -> block {res['block_full']['fold_std']:.3f}", flush=True)
if block_gain < 0.5 * shuf_gain or res["block_full"]["fold_std"] > 2 * res["shuf_full"]["fold_std"]:
    print("   --> SIGNAL: GR-matcher gain is OOD-fragile (shrinks OOD and/or fold variance blows up) -> likely OOF candy.", flush=True)
else:
    print("   --> SIGNAL: GR-matcher gain holds under block-holdout -> real, transfer risk lower.", flush=True)
json.dump({"n_total": len(all_feats), "n_dropped": len(gr_matchers), "dropped": gr_matchers,
           "shuf_gain": shuf_gain, "block_gain": block_gain, **res},
          open(SP / "gr_ablation_summary.json", "w"), indent=2)
print("=== SUPER ABLATION DONE ===", flush=True)
