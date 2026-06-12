"""Addendum to weights_axis_scan: 12-model positive-Ridge (6 nouk + 6 super) + PF, out-of-fold.

Question: does pooling at the MODEL level (one 12-column positive-Ridge stack, then scalar PF blend)
beat the stack-of-stacks 3-way (9.1222)? The 12-model shape is the cleaner production architecture
(same kernel blend code, just 12 models + a 2nd feature matrix).

Also reports the out-of-fold-ridge convention next to the production full-OOF-ridge convention.

Run: nohup python -u experiments/weights_axis_scan2.py > log/weights_scan2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import Ridge

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
KEYS = ["lgb_42", "lgb_7", "lgb_123", "cat_42", "cat_7", "cat_123"]
rmse = lambda e: float(np.sqrt(np.mean(e * e)))

df = pd.read_parquet(ROOT / "data/processed/frontier_ens/train_feats.parquet",
                     columns=["well", "id", "last_known_tvt", "target"])
y = df["target"].to_numpy(np.float64)
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)

Xn = np.column_stack([np.load(ROOT / "models/frontier_ens_nouk" / f"oof_{k}.npy").astype(np.float64) for k in KEYS])
sup = pd.read_parquet(ROOT / "data/processed/super/train_feats.parquet", columns=["id"])
Xs_own = np.column_stack([np.load(ROOT / "models/super" / f"oof_{k}.npy").astype(np.float64) for k in KEYS])
sup = pd.concat([sup, pd.DataFrame(Xs_own, columns=[f"s_{k}" for k in KEYS])], axis=1)
m = df.merge(sup, on="id", how="left")
Xs = m[[f"s_{k}" for k in KEYS]].to_numpy(np.float64)
assert not np.isnan(Xs).any()
X12 = np.hstack([Xn, Xs])

# PF alignment (verbatim protocol from bet5_pf_blend.py)
paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells = []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue
    pf_wells.append(wid)
grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}
res = joblib.load(ROOT / "models/frontier/pf_real_results.pkl")
rg = [r for r in res if r is not None and not (isinstance(r[0], str) and r[0] == "ERR")]
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
pos = np.concatenate(pos_l); p_res = np.concatenate(pres_l)

rng = np.random.RandomState(42); uw = df["well"].unique().copy(); rng.shuffle(uw)
fo = {w: i % 5 for i, w in enumerate(uw)}
fold_all = df["well"].map(fo).to_numpy()
fold = fold_all[pos]

# full-OOF ridge (production convention) on 12 models
r12 = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(X12, y)
res12 = X12 @ r12.coef_ - y
print(f"12-model stack (full-OOF ridge): OOF = {rmse(res12):.4f}", flush=True)
print(f"   coef = {dict(zip(['n_'+k for k in KEYS]+['s_'+k for k in KEYS], np.round(r12.coef_,3)))}", flush=True)
print(f"   (nouk 6-model was 10.3232; super 10.3708)", flush=True)

# out-of-fold ridge on 12 models (honest convention)
res12_of = np.empty(len(y))
for f in range(5):
    tr, va = fold_all != f, fold_all == f
    rr = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(X12[tr], y[tr])
    res12_of[va] = X12[va] @ rr.coef_ - y[va]
print(f"12-model stack (out-of-fold ridge): OOF = {rmse(res12_of):.4f}", flush=True)


def blend2_oof(g, p, tag):
    gw = np.empty(len(g))
    for f in range(5):
        tr, va = fold != f, fold == f
        d = g[tr] - p[tr]
        gw[va] = float(np.dot(g[tr], d) / np.dot(d, d))
    r = rmse((1 - gw) * g + gw * p)
    print(f"   [{tag:28s}] OOF = {r:.4f}  w = {gw.mean():.3f}", flush=True)
    return r


print("\n=== +PF blend (the LB predictor) ===", flush=True)
blend2_oof(res12[pos], p_res, "12-model(full) + PF")
blend2_oof(res12_of[pos], p_res, "12-model(out-of-fold) + PF")
print(f"\n   reference: nouk+PF = 9.1621, 3-way nouk+PF+super = 9.1222", flush=True)
print("WEIGHTS SCAN 2 DONE", flush=True)
