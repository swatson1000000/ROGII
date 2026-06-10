"""#2 (research_geosteering_6ft.md §6.2): apparent-dip / cumulative-curvature feature family + cheap LGB gate.

The 222-feat build carries only 1st-order trajectory/dip terms (instantaneous dzdmd/dxdmd/dydmd,
linear prefix slopes slp_all/slp_50/slp_z, and the LINEAR extrapolation slp_b_d_*). The doc's #2 lever
is the 2nd-order / cumulative piece that b_well + a linear slope cannot represent: path curvature
(dogleg / Q-3D tortuosity) and the apparent-dip GRADIENT (how the dip changes along the well).

5 new features (all leak-safe + NOT density-coupled -> gate on ordinary OOF, not block-holdout):
    dogleg        per-row instantaneous path curvature = angle(tangent_i, tangent_{i-1}) / dMD
    cum_dogleg    per-row cumulative |dogleg| from PS (mycarta Q-3D tortuosity; absent + additive)
    tvt_dip_grad  per-well prefix d^2(TVT_input)/d(MD)^2  (quadratic 2nd coeff over the KNOWN prefix)
    tvt_dip_grad_z per-well prefix d^2(TVT_input)/d(Z)^2
    quad_b_d      per-row dip-aware (quadratic-fit) TVT extrapolation drift vs last_tvt (clipped +-200)

LEAK CHECK: dogleg/cum_dogleg use X/Y/Z/MD only (fully observed at eval); tvt_dip_grad(_z)/quad_b_d use
TVT_input over the KNOWN prefix only (rows < PS) + observed eval MD/Z. No eval-row TVT touched -- mirrors
how the existing slp_all/slp_50 (L824-825) are built.

GATE: cheap single-LGB GKF-5 seed42 (konbu params, same folds as bet5_lgb_check). base-222 vs +5.
  delta <= -0.05 (real, beyond the ~0.003 LGB noise + the ~0.114 single-LGB gain UK showed) -> proceed to
  full 6-model retrain. ~flat (|delta|<0.05) or regress -> the dip family is absorbed by the 222; STOP
  (it would only dilute through the 0.57 PF blend, like UK did). Also prints corr(feat,target) and
  corr(feat, base-OOF residual) to see leftover-error signal even if the LGB delta is small.

Run: nohup python -u experiments/dip_curvature_gate.py > log/dip_gate_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from pathlib import Path
import glob, os
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
FR = ROOT / "data/processed/frontier_seeded"
OUTDIR = ROOT / "data/processed/frontier_dip"; OUTDIR.mkdir(parents=True, exist_ok=True)
FEAT_OUT = ROOT / "data/processed/dip_feats.parquet"
NEW_COLS = ["dogleg", "cum_dogleg", "tvt_dip_grad", "tvt_dip_grad_z", "quad_b_d"]
N_SPLITS, SPLIT_SEED = 5, 42
LGB_PARAMS = dict(boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
    min_child_samples=10, min_child_weight=0.5, n_estimators=5000, n_jobs=-1,
    reg_alpha=2.03, reg_lambda=87.28, subsample=0.645, subsample_freq=1,
    colsample_bytree=0.821, objective="regression", metric="rmse", verbose=-1,
    device_type="cuda", max_bin=255, random_state=42)


def well_dip_feats(md, x, y, z, tvt_in):
    """Per-row dip/curvature arrays for one well (full trajectory). Returns dict of len(md) arrays.
    md,x,y,z: float64 full-trajectory. tvt_in: float64 with NaN past PS (known prefix only)."""
    n = len(md)
    kn = np.isfinite(tvt_in)
    ps = int(kn.sum())  # known prefix is contiguous at the start; PS = first eval row index

    # ---- instantaneous path curvature (dogleg) over the full trajectory ----
    dmd = np.diff(md)
    dmd_safe = np.where(dmd > 1e-6, dmd, np.nan)
    tx, ty, tz = np.diff(x) / dmd_safe, np.diff(y) / dmd_safe, np.diff(z) / dmd_safe  # tangent components
    tnorm = np.sqrt(tx * tx + ty * ty + tz * tz)
    tnorm_safe = np.where(tnorm > 1e-9, tnorm, np.nan)
    ux, uy, uz = tx / tnorm_safe, ty / tnorm_safe, tz / tnorm_safe  # unit tangents, len n-1
    dot = (ux[1:] * ux[:-1] + uy[1:] * uy[:-1] + uz[1:] * uz[:-1])
    ang = np.arccos(np.clip(dot, -1.0, 1.0))                       # turn angle between consecutive tangents
    dogleg_mid = ang / np.where(dmd_safe[1:] > 1e-6, dmd_safe[1:], np.nan)  # rad/ft, len n-2
    dogleg = np.zeros(n, np.float64)
    dogleg[2:] = np.nan_to_num(dogleg_mid, nan=0.0)                # align: row i curvature uses i-2,i-1,i
    # cumulative curvature from PS forward
    cum = np.zeros(n, np.float64)
    if ps < n:
        cum[ps:] = np.cumsum(dogleg[ps:])

    # ---- prefix apparent-dip gradient (quadratic 2nd-derivative of TVT over the known prefix) ----
    grad_md = 0.0; grad_z = 0.0
    qcoef = None; md_c = 0.0
    last_tvt = float(tvt_in[kn][-1]) if ps >= 1 else 0.0
    if ps >= 10:
        mdk = md[:ps]; zk = z[:ps]; tk = tvt_in[:ps]
        md_c = float(mdk.mean())
        try:
            qcoef = np.polyfit(mdk - md_c, tk, 2)                  # [a,b,c]; a = 0.5*d2TVT/dMD2
            grad_md = float(2.0 * qcoef[0])
        except Exception:
            qcoef = None
        try:
            zc = float(zk.mean()); qz = np.polyfit(zk - zc, tk, 2)
            grad_z = float(2.0 * qz[0])
        except Exception:
            grad_z = 0.0

    # ---- per-row dip-aware (quadratic) TVT extrapolation drift ----
    quad_b_d = np.zeros(n, np.float64)
    if qcoef is not None:
        pred = np.polyval(qcoef, md - md_c)
        quad_b_d = np.clip(pred - last_tvt, -200.0, 200.0)
    else:  # too-short prefix: fall back to flat (drift 0)
        quad_b_d[:] = 0.0

    return {
        "dogleg": dogleg.astype(np.float32),
        "cum_dogleg": cum.astype(np.float32),
        "tvt_dip_grad": np.full(n, np.float32(grad_md)),
        "tvt_dip_grad_z": np.full(n, np.float32(grad_z)),
        "quad_b_d": quad_b_d.astype(np.float32),
    }


def build_feats(tf, raw_dir, label):
    """Compute NEW_COLS for the eval rows present in tf (aligned by _ri). Returns DataFrame[id + NEW_COLS]."""
    tf = tf.copy()
    tf["_ri"] = tf["id"].str.rsplit("_", n=1).str[-1].astype(int)
    paths = {os.path.basename(p).replace("__horizontal_well.csv", ""): p
             for p in glob.glob(str(raw_dir / "*__horizontal_well.csv"))}
    grp = {w: g for w, g in tf.groupby("well", sort=False)}
    wells = list(tf["well"].unique())
    recs = {"id": [], **{c: [] for c in NEW_COLS}}
    miss = 0
    for k, wid in enumerate(wells):
        g = grp[wid]; p = paths.get(wid)
        if p is None:
            miss += len(g); continue
        h = pd.read_csv(p, usecols=lambda c: c in ("MD", "X", "Y", "Z", "TVT_input"))
        md = h["MD"].to_numpy(np.float64); x = h["X"].to_numpy(np.float64)
        y = h["Y"].to_numpy(np.float64); z = h["Z"].to_numpy(np.float64)
        tin = h["TVT_input"].to_numpy(np.float64)
        f = well_dip_feats(md, x, y, z, tin)
        ri = g["_ri"].to_numpy()
        recs["id"].append(g["id"].to_numpy())
        for c in NEW_COLS:
            recs[c].append(f[c][ri])
        if (k + 1) % 200 == 0:
            print(f"   [{label}] {k+1}/{len(wells)}", flush=True)
    out = pd.DataFrame({"id": np.concatenate(recs["id"]),
                        **{c: np.concatenate(recs[c]) for c in NEW_COLS}})
    print(f"   [{label}] {len(out):,} rows, {miss} unmatched", flush=True)
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
    print(">> load train_feats + build dip feats...", flush=True)
    tr = pd.read_parquet(FR / "train_feats.parquet")
    dip_tr = build_feats(tr[["well", "id"]], RAW, "train")
    dip_tr.to_parquet(FEAT_OUT)
    print(f">> saved dip feats -> {FEAT_OUT}", flush=True)
    tr = tr.merge(dip_tr, on="id", how="left")
    assert all(c in tr.columns for c in NEW_COLS), "dip merge failed"
    nbad = int(tr[NEW_COLS].isna().any(axis=1).sum())
    for c in NEW_COLS:
        tr[c] = tr[c].fillna(0.0).astype(np.float32)
    print(f"   merged {tr.shape}; {nbad} rows had a NaN (filled 0)", flush=True)
    tr.to_parquet(OUTDIR / "train_feats.parquet")
    # also build for the 3-well test matrix (consistency for a later retrain)
    te = pd.read_parquet(FR / "test_feats.parquet")
    dip_te = build_feats(te[["well", "id"]], ROOT / "data/raw/test", "test")
    te = te.merge(dip_te, on="id", how="left")
    for c in NEW_COLS:
        te[c] = te[c].fillna(0.0).astype(np.float32)
    te.to_parquet(OUTDIR / "test_feats.parquet")
    print(f"   test merged {te.shape}", flush=True)

    base_cols = [c for c in tr.columns if c not in {"well", "id", "target"} and c not in NEW_COLS]
    dip_cols = base_cols + NEW_COLS
    y = tr["target"].to_numpy(np.float32)
    rng = np.random.RandomState(SPLIT_SEED); uw = tr["well"].unique().copy(); rng.shuffle(uw)
    fold_of = {w: i % N_SPLITS for i, w in enumerate(uw)}
    wf = tr["well"].map(fold_of).to_numpy()
    splits = [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(N_SPLITS)]

    print(f">> base ({len(base_cols)} feats) ...", flush=True)
    r_base, oof_base = run_oof(tr, base_cols, y, splits, "base")
    print(f">> +dip ({len(dip_cols)} feats) ...", flush=True)
    r_dip, oof_dip = run_oof(tr, dip_cols, y, splits, "+dip")
    np.save(ROOT / "models/frontier/oof_lgb42_dipcheck.npy", oof_dip)

    resid = oof_base - y
    print("\n=== feature diagnostics (corr with target / base-OOF residual) ===", flush=True)
    for c in NEW_COLS:
        v = tr[c].to_numpy(np.float32)
        ct = float(np.corrcoef(v, y)[0, 1]) if np.std(v) > 0 else 0.0
        cr = float(np.corrcoef(v, resid)[0, 1]) if np.std(v) > 0 else 0.0
        print(f"  {c:16s} std={np.std(v):.4g}  corr(target)={ct:+.4f}  corr(residual)={cr:+.4f}", flush=True)

    d = r_dip - r_base
    print("\n=== CHEAP DIP-CURVATURE GATE VERDICT ===", flush=True)
    print(f"  base LGB-222 OOF = {r_base:.4f}   +dip LGB-227 OOF = {r_dip:.4f}   delta = {d:+.4f}", flush=True)
    if d <= -0.05:
        print("  >> REAL SIGNAL (beyond LGB noise): proceed to full 6-model retrain + PF-blend + LB.", flush=True)
    elif d <= 0.003:
        print("  >> ~FLAT: the dip family is largely ABSORBED by the 222. A feature this small would only "
              "dilute through the 0.57 PF blend (like UK: -0.069 stack -> +0.060 LB). STOP -- not worth a retrain.", flush=True)
    else:
        print("  >> REGRESSION: dip feats HURT the interpolation OOF. STOP.", flush=True)
    print("DIP GATE DONE", flush=True)


if __name__ == "__main__":
    main()
