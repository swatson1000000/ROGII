"""STAGE 1 (free, cached): does the PUBLISHED sel15 selector + hold mechanism improve our PF arm,
and does the selector-arm OUTPUT-BLENDED with our GBM beat our current 9.17 OOF?

The 7.776 notebooks route the PF's INTERNAL config per well-bin (scale/beam/hold by n_eval,z_span)
-- a DIFFERENT selector than the GBM-vs-PF weight we gated null. The selector + hold are
deterministic (fixed thresholds, no fitting), so we can apply them to our cached multi-scale PF
(models/frontier/pf_real_results.pkl, scales 3/5/8/12, spread 2.0) + 14-config beam
(beam_real_results.pkl) with ZERO re-compute. This isolates the selector+hold gain BEFORE spending
57 min on the sp45 (spread 4.5) PF re-run.

selector constants + functions VERBATIM from /tmp/rogii_0608/lightningv08_lb-7-776-rogii-ridge-sp.
Caveat: thresholds/variant map are PUBLIC-tuned (possibly on the leak); this is a first read, not
the final tuning. We IGNORE tvt_from_contacts (3-well leak). OOF over 773 train wells is honest.
"""
from pathlib import Path
import glob, os, json
import numpy as np
import pandas as pd
import joblib

ROOT = Path("/home/swatson/work/kaggle/ROGII")
RAW = ROOT / "data/raw/train"
FR = ROOT / "data/processed/frontier_seeded"
MODELS = ROOT / "models/frontier"

# ---- VERBATIM selector (lightningv08 7.776) ----
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2', 1: 'pf_scale_3_hold_0.15', 2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15', 4: 'pf_scale_5_beam_0.05_hold_0.05', 5: 'pf_scale_12_beam_0.2_hold_0.05',
}
SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'

def selector_well_code(n_eval, z_span):
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side='right'))
    code = n_bin + 2 * z_bin
    return code, SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)

def parse_selector_variant(name):
    parts = name.split('_'); scale = float(parts[2]); bw = hw_ = 0.0
    if 'beam' in parts: bw = float(parts[parts.index('beam') + 1])
    if 'hold' in parts: hw_ = float(parts[parts.index('hold') + 1])
    return scale, bw, hw_

def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    scale, bw, hw_ = parse_selector_variant(name)
    base = pf_by_scale.get(f'pf_scale_{scale:g}')
    pred = (1.0 - bw) * base + bw * tvt_beam
    pred = (1.0 - hw_) * pred + hw_ * last_known_tvt
    return pred
# ---- end verbatim ----

rmse = lambda e: float(np.sqrt(np.mean(e * e)))

print(">> load GBM oof + align...", flush=True)
df = pd.read_parquet(FR / "train_feats.parquet", columns=["well", "id", "last_known_tvt", "target"])
blend = json.load(open(MODELS / "blend_frontier.json"))
coef = np.array(blend["ridge_coef"], np.float64)
oofs = np.column_stack([np.load(MODELS / f"oof_{k}.npy").astype(np.float64) for k in blend["keys"]])
gbm_resid = (oofs @ coef) - df["target"].to_numpy(np.float64)
df["_ri"] = df["id"].str.rsplit("_", n=1).str[-1].astype(int)

paths = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
pf_wells, well_neval, well_zspan, well_lastk = [], [], [], []
for p in paths:
    wid = os.path.basename(p).replace("__horizontal_well.csv", "")
    hw = pd.read_csv(p, usecols=lambda c: c in ("TVT", "TVT_input", "Z"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 20:
        continue
    ev = hw["TVT_input"].isna().values
    z = hw["Z"].values.astype(float)[ev]
    pf_wells.append(wid)
    well_neval.append(int(ev.sum()))
    well_zspan.append(float(np.nanmax(z) - np.nanmin(z)) if len(z) else 0.0)
    well_lastk.append(float(hw["TVT_input"].dropna().iloc[-1]))

pf_res = joblib.load(MODELS / "pf_real_results.pkl")
bm_res = joblib.load(MODELS / "beam_real_results.pkl")
def good(r): return r is not None and not (isinstance(r[0], str) and r[0] == "ERR")
pf_good = [r for r in pf_res if good(r)]
bm_good = [r for r in bm_res if good(r)]
assert len(pf_good) == len(bm_good) == len(pf_wells), "alignment mismatch"

grp = {w: g.sort_values("_ri") for w, g in df.groupby("well", sort=False)}
sel_res, code_hist = [], {}
gbm_keep = []
ref = {k: [] for k in ['pf_scale_8', 'pf_scale_12', 'global', 'selector']}
truth_all = []
for i, wid in enumerate(pf_wells):
    truth, pf_by_scale = pf_good[i]
    tb = bm_good[i][1]['beam_mean']
    lk = well_lastk[i]
    code, variant = selector_well_code(well_neval[i], well_zspan[i])
    code_hist[code] = code_hist.get(code, 0) + 1
    sel = apply_selector_variant(variant, pf_by_scale, tb, lk)
    glob_ = apply_selector_variant(SELECTOR_GLOBAL_VARIANT, pf_by_scale, tb, lk)
    sub = grp.get(wid)
    ok = sub is not None and len(sub) == len(truth) and np.allclose(
        sub["target"].to_numpy(np.float64) + sub["last_known_tvt"].to_numpy(np.float64), truth, atol=1e-3)
    if not ok:
        continue
    truth_all.append(truth)
    ref['pf_scale_8'].append(pf_by_scale['pf_scale_8'] - truth)
    ref['pf_scale_12'].append(pf_by_scale['pf_scale_12'] - truth)
    ref['global'].append(glob_ - truth)
    ref['selector'].append(sel - truth)
    gbm_keep.append((sub.index.to_numpy(), sel - truth))

print(f"   selector bin histogram (code:count): {dict(sorted(code_hist.items()))}", flush=True)
print("\n=== STANDALONE OOF (eval rows) ===", flush=True)
print("  anchors: null 15.910 | GBM 10.356 | our scale-12 PF 10.993 | current GBM+PF blend 9.17", flush=True)
for k in ['pf_scale_12', 'pf_scale_8', 'global', 'selector']:
    print(f"  {k:12s} RMSE = {rmse(np.concatenate(ref[k])):.4f}", flush=True)

# ---- output-blend the SELECTOR arm with the GBM (re-fit w out-of-fold) ----
pos = np.concatenate([idx for idx, _ in gbm_keep])
s_res = np.concatenate([r for _, r in gbm_keep])
g_res = gbm_resid[pos]
rng = np.random.RandomState(42)
uw = df["well"].unique().copy(); rng.shuffle(uw)
fold_of = {w: i % 5 for i, w in enumerate(uw)}
fold = np.array([fold_of[w] for w in df["well"].to_numpy()[pos]])
gw = np.empty(len(g_res))
for f in range(5):
    tr, va = fold != f, fold == f
    d = g_res[tr] - s_res[tr]; gw[va] = float(np.dot(g_res[tr], d) / np.dot(d, d))
blend_rmse = rmse((1 - gw) * g_res + gw * s_res)
print("\n=== SELECTOR-ARM (+) GBM OUTPUT-BLEND (out-of-fold w) ===", flush=True)
print(f"  GBM-only OOF            = {rmse(g_res):.4f}", flush=True)
print(f"  GBM (+) selector-arm    = {blend_rmse:.4f}   (mean w_sel={gw.mean():.3f})", flush=True)
print(f"  current GBM (+) scale12 = 9.1686 (banked path)", flush=True)
gain = 9.1686 - blend_rmse
print("\n=== VERDICT (spread-2.0 cache; sp45 would only add) ===", flush=True)
if gain >= 0.05:
    print(f"  PROMISING: selector arm blend OOF {blend_rmse:.4f} beats 9.1686 by {gain:+.4f} even at spread 2.0", flush=True)
    print("  -> run the sp45 (spread 4.5) PF re-run + full reproduce; expect more.", flush=True)
elif gain >= 0.0:
    print(f"  MARGINAL at spread 2.0 ({gain:+.4f}); sp45 may push it over -> worth the PF re-run.", flush=True)
else:
    print(f"  selector arm does NOT beat our blend at spread 2.0 ({gain:+.4f}); sp45 is the swing factor", flush=True)
    print("  -> the gain (if any) lives in sp45, not the selector/hold; re-run PF with spread 4.5 to check.", flush=True)
print("SELARM DONE", flush=True)
