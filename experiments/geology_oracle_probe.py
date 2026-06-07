"""Geology-layer lever: oracle-ceiling + extractability probe (gate BEFORE building features).

The typewell gives (TVT, GR, Geology). Geology is the stratigraphic layer label at each typewell
TVT depth -> it defines, per layer, a TVT band [lo,hi]. The typewell TVT is in the SAME frame as
the horizontal-well TVT (verified). So any TVT estimate can be mapped -> layer -> band.

We test, over eval rows (>= PS) of all train wells:
  (A) ORACLE CEILING: if we knew the TRUE layer, how well does the layer-band-CENTER predict TVT?
      And does CLAMPING the frontier-quality estimate to the true band reduce error? (Headroom test.)
  (B) EXTRACTABILITY: how often does mapping an OBSERVED estimate (last_known_TVT, the geologist's
      anchor) -> layer match the true layer? A cheap proxy for "can we predict the layer at all."
  (C) REDUNDANCY: is the layer already implied by last_known_TVT? (If last_known already lands in the
      right band almost always, the layer feature adds ~nothing on top of what the model has.)

Null over eval rows = predict last_known_TVT. Our banked frontier is ~10.1 LB / 10.356 OOF.
"""
from pathlib import Path
import glob, os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = Path("/home/swatson/work/kaggle/ROGII")
HWF = sorted(glob.glob(str(ROOT / "data/raw/train/*__horizontal_well.csv")))


def layer_bands(tw):
    """layer -> (lo, hi) TVT band from the typewell."""
    g = tw.dropna(subset=["Geology", "TVT"])
    bands = {}
    for lay, sub in g.groupby("Geology"):
        bands[lay] = (float(sub.TVT.min()), float(sub.TVT.max()))
    return bands


def tvt_to_layer(tw_sorted_tvt, tw_sorted_lay, tvt):
    """nearest-typewell-row layer for a TVT value (vectorized via searchsorted)."""
    idx = np.searchsorted(tw_sorted_tvt, tvt)
    idx = np.clip(idx, 0, len(tw_sorted_tvt) - 1)
    # snap to nearer of idx-1/idx
    left = np.clip(idx - 1, 0, len(tw_sorted_tvt) - 1)
    pick_left = np.abs(tw_sorted_tvt[left] - tvt) < np.abs(tw_sorted_tvt[idx] - tvt)
    idx = np.where(pick_left, left, idx)
    return tw_sorted_lay[idx]


def one_well(f):
    wid = os.path.basename(f).split("__")[0]
    hw = pd.read_csv(f)
    tw = pd.read_csv(f.replace("__horizontal_well", "__typewell"))
    tw = tw.dropna(subset=["TVT", "Geology"]).sort_values("TVT")
    if len(tw) < 5 or hw["TVT_input"].isna().sum() == 0:
        return None
    twt = tw.TVT.to_numpy(np.float64); twl = tw.Geology.to_numpy(object)
    bands = layer_bands(tw)
    # eval rows = where TVT_input is NaN but TVT (truth) known (train wells)
    ev = hw[hw["TVT_input"].isna() & hw["TVT"].notna()].copy()
    if len(ev) == 0:
        return None
    last_known = hw.loc[hw["TVT_input"].notna(), "TVT_input"]
    if len(last_known) == 0:
        return None
    lk = float(last_known.iloc[-1])
    true_tvt = ev.TVT.to_numpy(np.float64)
    n = len(true_tvt)

    true_lay = tvt_to_layer(twt, twl, true_tvt)
    lk_lay = tvt_to_layer(twt, twl, np.full(n, lk))
    # band center / width for the TRUE layer (oracle) and for the lk-implied layer (observed)
    def band_arr(lays):
        lo = np.array([bands.get(l, (np.nan, np.nan))[0] for l in lays])
        hi = np.array([bands.get(l, (np.nan, np.nan))[1] for l in lays])
        return lo, hi
    olo, ohi = band_arr(true_lay)
    ocenter = (olo + ohi) / 2
    owidth = ohi - olo
    # clamp lk to the TRUE band (oracle headroom: does layer correct the anchor?)
    lk_clamped = np.clip(np.full(n, lk), olo, ohi)

    return dict(
        wid=wid, n=n,
        true_tvt=true_tvt, lk=np.full(n, lk),
        ocenter=ocenter, owidth=owidth, lk_clamped=lk_clamped,
        layer_match=(true_lay == lk_lay).astype(np.float64),
    )


def main():
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = [r for r in ex.map(one_well, HWF) if r is not None]
    print(f">> wells with eval rows: {len(res)}", flush=True)
    cat = lambda k: np.concatenate([r[k] for r in res])
    true_tvt = cat("true_tvt"); lk = cat("lk")
    ocenter = cat("ocenter"); owidth = cat("owidth"); lk_clamped = cat("lk_clamped")
    layer_match = cat("layer_match")
    rmse = lambda p: float(np.sqrt(np.nanmean((true_tvt - p) ** 2)))

    print(f"\n=== eval rows: {len(true_tvt):,} ===", flush=True)
    print(f"NULL (predict last_known_TVT)        RMSE = {rmse(lk):.3f}", flush=True)
    print(f"ORACLE layer-band CENTER             RMSE = {rmse(ocenter):.3f}", flush=True)
    print(f"ORACLE clamp(last_known -> true band) RMSE = {rmse(lk_clamped):.3f}", flush=True)
    print(f"\nlayer band width: median={np.nanmedian(owidth):.1f} ft  p25={np.nanpercentile(owidth,25):.1f}  p75={np.nanpercentile(owidth,75):.1f}", flush=True)
    print(f"EXTRACTABILITY: last_known-implied layer == true layer: {layer_match.mean()*100:.1f}% of eval rows", flush=True)
    print("\n(banked frontier ~10.1 LB / 10.356 OOF; null ~15.9. Oracle below ~10 => headroom; clamp<<null => layer corrects anchor.)", flush=True)


if __name__ == "__main__":
    main()
