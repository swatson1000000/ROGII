"""Weights axis scan on the banked nouk-231 stack (all offline, 0 submissions).

Three questions, all on out-of-fold weights (same well-fold protocol as production):
  (a) PF blend weight re-fit on the NOUK stack -- the plan.md contingency: kernel v10 ships
      w=0.57, the vertex fit on the OLD frontier-222 GBM. If the nouk stack's OOF-optimal w
      shifted vs the 222 stack's, the LB vertex likely shifted by ~the same delta.
  (b) The SUPER 6-model stack (170-feat sidegrade build, OOF ~10.37) as a 2nd GBM output-blend
      member -- never tested (the "averaging axis spent" scan covered single columns + beam only).
  (c) 3-way nouk + PF + super (and 4-way with the old 222 stack) out-of-fold.

Self-checks: base-222 stack reproduces 10.3556; nouk reproduces 10.3232; 222+PF reproduces ~9.1732.

Run: nohup python -u experiments/weights_axis_scan.py > log/weights_scan_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import Ridge

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
KEYS = ["lgb_42", "lgb_7", "lgb_123", "cat_42", "cat_7", "cat_123"]
rmse = lambda e: float(np.sqrt(np.mean(e * e)))

print(">> load frontier_ens train matrix (row order of all frontier-family OOF arrays)...", flush=True)
df = pd.read_parquet(ROOT / "data/processed/frontier_ens/train_feats.parquet",
                     columns=["well", "id", "last_known_tvt", "target"])
y = df["target"].to_numpy(np.float64)
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)


def stack_resid(mdir, check=None):
    X = np.column_stack([np.load(ROOT / "models" / mdir / f"oof_{k}.npy").astype(np.float64) for k in KEYS])
    assert len(X) == len(y), f"{mdir}: {len(X)} vs {len(y)}"
    r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(X, y)
    res = X @ r.coef_ - y
    print(f"   {mdir:22s} stack OOF = {rmse(res):.4f}" + (f"  [self-check {check}]" if check else ""), flush=True)
    if check is not None:
        assert abs(rmse(res) - check) < 0.01, f"{mdir} mismatch"
    return res


res_nouk = stack_resid("frontier_ens_nouk", check=10.3232)
res_222 = stack_resid("frontier", check=10.3556)

# --- super stack: different parquet -> align by id ---
print(">> align super stack OOF by id...", flush=True)
sup = pd.read_parquet(ROOT / "data/processed/super/train_feats.parquet", columns=["id", "target"])
Xs = np.column_stack([np.load(ROOT / "models/super" / f"oof_{k}.npy").astype(np.float64) for k in KEYS])
assert len(Xs) == len(sup)
rs = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(Xs, sup["target"].to_numpy(np.float64))
sup_res_own = Xs @ rs.coef_ - sup["target"].to_numpy(np.float64)
print(f"   super (own rows)       stack OOF = {rmse(sup_res_own):.4f}  [blend6 raw was 10.371]", flush=True)
sup["sres"] = sup_res_own
m = df.merge(sup, on="id", how="left", suffixes=("", "_sup"))
cov = m["sres"].notna().mean()
tgt_ok = np.nanmax(np.abs(m["target_sup"].to_numpy(np.float64) - y)) if cov == 1.0 else \
    np.nanmax(np.abs((m["target_sup"] - df["target"]).to_numpy(np.float64)))
print(f"   coverage on frontier rows = {cov:.4f}, max|target diff| = {tgt_ok:.4g}", flush=True)
res_sup = m["sres"].to_numpy(np.float64)  # NaN where uncovered

# --- PF residual, aligned per-well (verbatim protocol from bet5_pf_blend.py) ---
print(">> align PF (128-seed scale-12) residuals...", flush=True)
paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells = []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue
    pf_wells.append(wid)
grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}


def good(r):
    return r is not None and not (isinstance(r[0], str) and r[0] == "ERR")


res = joblib.load(ROOT / "models/frontier/pf_real_results.pkl")
rg = [r for r in res if good(r)]
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
pos = np.concatenate(pos_l)
p_res = np.concatenate(pres_l)
print(f"   PF rows aligned: {len(pos)} / {len(df)}", flush=True)

# restrict every member to the PF-covered rows (the production blend domain)
g_nouk, g_222 = res_nouk[pos], res_222[pos]
g_sup = res_sup[pos]
sup_ok = ~np.isnan(g_sup)
print(f"   super coverage on PF rows: {sup_ok.mean():.4f}", flush=True)

rng = np.random.RandomState(42)
uw = df["well"].unique().copy()
rng.shuffle(uw)
fo = {w: i % 5 for i, w in enumerate(uw)}
fold = np.array([fo[w] for w in df["well"].to_numpy()[pos]])

print("\n=== member standalone RMSE (PF-covered rows) + residual corr matrix ===", flush=True)
members = {"nouk": g_nouk, "g222": g_222, "PF": p_res}
if sup_ok.all():
    members["super"] = g_sup
else:
    members["super"] = np.where(sup_ok, g_sup, 0.0)
    print("   ⚠ super has uncovered rows -> zero-filled (interpret super numbers with care)", flush=True)
for k, v in members.items():
    print(f"   {k:6s} {rmse(v):.4f}", flush=True)
names = list(members)
C = np.corrcoef(np.vstack([members[k] for k in names]))
print("   corr: " + "  ".join(f"{a}-{b}={C[i, j]:.3f}" for i, a in enumerate(names)
                              for j, b in enumerate(names) if i < j), flush=True)


def blend2_oof(g, p, tag, ref=None):
    """out-of-fold scalar w for (1-w)*g + w*p"""
    gw = np.empty(len(g))
    for f in range(5):
        tr, va = fold != f, fold == f
        d = g[tr] - p[tr]
        gw[va] = float(np.dot(g[tr], d) / np.dot(d, d))
    r = rmse((1 - gw) * g + gw * p)
    extra = f"  (delta vs {ref[0]}: {r - ref[1]:+.4f})" if ref else ""
    print(f"   [{tag:24s}] OOF = {r:.4f}  w = {gw.mean():.3f} (per-fold {gw.min():.3f}-{gw.max():.3f}){extra}", flush=True)
    return r, gw.mean()


def blendk_oof(gs, tag, ref=None):
    """out-of-fold sum-to-1 least-squares weights over k members; gs = list of residual arrays"""
    G = np.vstack(gs).T  # rows x k
    pred = np.empty(len(G))
    ws = []
    for f in range(5):
        tr, va = fold != f, fold == f
        D = G[tr, 1:] - G[tr, :1]  # r = g0 + A @ coefs, A = (gi - g0)
        coefs, *_ = np.linalg.lstsq(D, -G[tr, 0], rcond=None)
        pred[va] = G[va, 0] + (G[va, 1:] - G[va, :1]) @ coefs
        w = np.concatenate([[1 - coefs.sum()], coefs])
        ws.append(w)
    wm = np.mean(ws, axis=0)
    r = rmse(pred)
    extra = f"  (delta vs {ref[0]}: {r - ref[1]:+.4f})" if ref else ""
    print(f"   [{tag:24s}] OOF = {r:.4f}  w = {np.round(wm, 3).tolist()}{extra}", flush=True)
    return r, wm


print("\n=== (a) PF weight re-fit: 222 vs nouk stack ===", flush=True)
b222, w222 = blend2_oof(g_222, p_res, "g222 + PF  [check ~9.1732]")
bnouk, wnouk = blend2_oof(g_nouk, p_res, "nouk + PF", ref=("g222+PF", b222))
print(f"   >> OOF-opt w shift (nouk - 222) = {wnouk - w222:+.3f}; kernel ships LB-vertex 0.57 fit on the 222 stack.", flush=True)
print(f"   >> If the LB vertex shifts ~equally, nouk LB-opt w ~= {0.57 + (wnouk - w222):.3f}", flush=True)

print("\n=== (b) super stack as a 2nd GBM member ===", flush=True)
blend2_oof(g_nouk, members["super"], "nouk + super")
b3, w3 = blendk_oof([g_nouk, p_res, members["super"]], "nouk + PF + super", ref=("nouk+PF", bnouk))

print("\n=== (c) old 222 stack as extra diversity ===", flush=True)
blendk_oof([g_nouk, p_res, g_222], "nouk + PF + g222", ref=("nouk+PF", bnouk))
b4, w4 = blendk_oof([g_nouk, p_res, members["super"], g_222], "nouk + PF + super + g222", ref=("nouk+PF", bnouk))

print("\n=== VERDICT ===", flush=True)
print(f"   banked predictor (nouk+PF, w out-of-fold) OOF = {bnouk:.4f}", flush=True)
print(f"   best 3-way (nouk+PF+super)                OOF = {b3:.4f}  ({b3 - bnouk:+.4f})", flush=True)
print(f"   best 4-way (+g222)                        OOF = {b4:.4f}  ({b4 - bnouk:+.4f})", flush=True)
print("   Gate by [[output-blend-gated-by-orthogonality]]: judge the BLEND GAIN, not standalone dominance.", flush=True)
print("   But honest bar: GBM-side OOF gains transferred 1:1 only ONCE (no-UK); PF-side favorable.", flush=True)
print("WEIGHTS AXIS SCAN DONE", flush=True)
