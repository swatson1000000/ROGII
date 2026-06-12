"""frontier_super 6-model stack blend + PF-blend LB predictor, against the pre-registered gate.

Stack: positive-Ridge over 6 OOF columns (production convention, same as ens_nouk_blend.py);
self-checks the banked nouk stack reproduces 10.3232. Writes models/frontier_super/blend_frontier.json.
Then the LB predictor: out-of-fold scalar-w PF blend of each stack (protocol of bet5_pf_blend.py);
banked reference nouk+PF = 9.1621 (weights_axis_scan).

PRE-REGISTERED GATE (decided before this ran, do NOT rationalize after):
  stack gain <= -0.03 vs 10.3232 (the no-UK transfer size) AND PF-blended gain < 0
      -> proceed: kernel surgery (port 28 cols) + ONE submission with a pre-registered LB gate
  stack gain in (-0.03, 0) -> marginal, stop and reassess (don't ship a wash)
  stack gain >= 0          -> dead, stop

Run: nohup python -u experiments/ens_super_blend.py > log/ens_super_blend_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os, json
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import Ridge

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
KEYS = ["lgb_42", "lgb_7", "lgb_123", "cat_42", "cat_7", "cat_123"]
SEED_OF = {k: (k.split("_")[0], int(k.split("_")[1])) for k in KEYS}
rmse = lambda e: float(np.sqrt(np.mean(e * e)))

df = pd.read_parquet(ROOT / "data/processed/frontier_ens/train_feats.parquet",
                     columns=["well", "id", "last_known_tvt", "target"])
y = df["target"].to_numpy(np.float64)
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)


def stack(model_dir):
    d = ROOT / "models" / model_dir
    X = np.column_stack([np.load(d / f"oof_{SEED_OF[k][0]}_{SEED_OF[k][1]}.npy").astype(np.float64) for k in KEYS])
    assert len(X) == len(y)
    r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(X, y)
    res = X @ r.coef_ - y
    return rmse(res), r.coef_.tolist(), res


nouk_rmse, _, res_nouk = stack("frontier_ens_nouk")
print(f"   frontier_ens_nouk (231): stack OOF = {nouk_rmse:.4f}  [self-check 10.3232]", flush=True)
assert abs(nouk_rmse - 10.3232) < 0.01

sup_rmse, sup_coef, res_sup = stack("frontier_super")
print(f"   frontier_super    (259): stack OOF = {sup_rmse:.4f}  coef={[round(c,3) for c in sup_coef]}", flush=True)
d_stack = sup_rmse - nouk_rmse
print(f"   stack delta = {d_stack:+.4f}  (pre-registered bar: <= -0.03)", flush=True)

# cat3-only ablation (user question: is the LGB side worth keeping?)
dC = ROOT / "models/frontier_super"
Xc = np.column_stack([np.load(dC / f"oof_cat_{s}.npy").astype(np.float64) for s in [42, 7, 123]])
rc = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(Xc, y)
res_cat3 = Xc @ rc.coef_ - y
print(f"   frontier_super cat3-only: stack OOF = {rmse(res_cat3):.4f}  coef={[round(c,3) for c in rc.coef_.tolist()]}"
      f"  (vs 6-model {sup_rmse:.4f}; if >= 6-model -0.01, LGB diversity is real)", flush=True)
json.dump({"keys": KEYS, "ridge_coef": sup_coef, "oof": sup_rmse},
          open(ROOT / "models/frontier_super/blend_frontier.json", "w"), indent=2)

# --- PF blend (LB predictor), verbatim alignment protocol ---
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
fold = np.array([fo[w] for w in df["well"].to_numpy()[pos]])


def blend2_oof(g_all, tag):
    g = g_all[pos]; gw = np.empty(len(g))
    for f in range(5):
        tr, va = fold != f, fold == f
        d = g[tr] - p_res[tr]
        gw[va] = float(np.dot(g[tr], d) / np.dot(d, d))
    r = rmse((1 - gw) * g + gw * p_res)
    print(f"   [{tag:14s}+ PF] OOF = {r:.4f}  w = {gw.mean():.3f}", flush=True)
    return r


print("\n=== PF-blended (the LB predictor) ===", flush=True)
b_nouk = blend2_oof(res_nouk, "nouk(231)  ")
b_sup = blend2_oof(res_sup, "super(259) ")
b_cat3 = blend2_oof(res_cat3, "cat3-only  ")
d_blend = b_sup - b_nouk
print(f"   PF-blended delta = {d_blend:+.4f}  (nouk->LB was ~1:1 at the stack level)", flush=True)

print("\n=== PRE-REGISTERED GATE VERDICT ===", flush=True)
if d_stack <= -0.03 and d_blend < 0:
    print(f"  >> PROCEED: stack {d_stack:+.4f} <= -0.03 and PF-blended {d_blend:+.4f} < 0.", flush=True)
    print("     Next: kernel surgery (port the 28 cols), bit-exact validation, ONE submission,", flush=True)
    print("     with an LB gate registered BEFORE submitting.", flush=True)
elif d_stack < 0:
    print(f"  >> MARGINAL: stack {d_stack:+.4f} in (-0.03, 0). Stop and reassess -- don't ship a wash.", flush=True)
else:
    print(f"  >> DEAD: stack {d_stack:+.4f} >= 0. The cheap-LGB gain did not survive the 6-model stack. STOP.", flush=True)
print("ENS SUPER BLEND DONE", flush=True)
