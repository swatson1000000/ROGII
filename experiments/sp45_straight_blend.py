"""Straight output-blend (production path): final = (1-w)*GBM + w*PF_scale12, w re-fit out-of-fold.
Compare sp45 (spread 4.5) scale-12 PF vs the banked spread-2.0 scale-12 PF (OOF blend 9.1686)."""
from pathlib import Path
import glob, os, json
import numpy as np, pandas as pd, joblib

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"; FR = ROOT / "data/processed/frontier_seeded"; MODELS = ROOT / "models/frontier"
rmse = lambda e: float(np.sqrt(np.mean(e * e)))

df = pd.read_parquet(FR / "train_feats.parquet", columns=["well", "id", "last_known_tvt", "target"])
blend = json.load(open(MODELS / "blend_frontier.json"))
coef = np.array(blend["ridge_coef"], np.float64)
oofs = np.column_stack([np.load(MODELS / f"oof_{k}.npy").astype(np.float64) for k in blend["keys"]])
gbm_resid = (oofs @ coef) - df["target"].to_numpy(np.float64)
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)

paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells = []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input", "Z"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue
    pf_wells.append(wid)

grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}

def good(r): return r is not None and not (isinstance(r[0], str) and r[0] == "ERR")

def blend_oof(pkl_name, scale_key="pf_scale_12"):
    res = joblib.load(MODELS / pkl_name)
    rg = [r for r in res if good(r)]
    assert len(rg) == len(pf_wells)
    pos_l, pres_l = [], []
    for i, wid in enumerate(pf_wells):
        truth, pf_by_scale = rg[i]
        sub = grp.get(wid)
        if sub is None or len(sub) != len(truth): continue
        if not np.allclose(sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64), truth, atol=1e-3):
            continue
        pos_l.append(sub.index.to_numpy())
        pres_l.append(pf_by_scale[scale_key] - truth)
    pos = np.concatenate(pos_l); p_res = np.concatenate(pres_l); g_res = gbm_resid[pos]
    rng = np.random.RandomState(42); uw = df["well"].unique().copy(); rng.shuffle(uw)
    fold_of = {w: i % 5 for i, w in enumerate(uw)}
    fold = np.array([fold_of[w] for w in df["well"].to_numpy()[pos]])
    gw = np.empty(len(g_res))
    for f in range(5):
        tr, va = fold != f, fold == f
        d = g_res[tr] - p_res[tr]; gw[va] = float(np.dot(g_res[tr], d) / np.dot(d, d))
    return rmse((1 - gw) * g_res + gw * p_res), gw.mean(), rmse(p_res)

for name, pkl in [("spread-2.0 (banked)", "pf_real_results.pkl"), ("sp45 (spread 4.5)", "pf_sp45_results.pkl")]:
    b, w, sa = blend_oof(pkl)
    print(f"  {name:22s}: standalone PF={sa:.4f}  GBM(+)PF blend={b:.4f}  (mean w_pf={w:.3f})", flush=True)
print("SP45STRAIGHT DONE", flush=True)
