"""6-model positive-Ridge stack blend for frontier_ens, vs base-222 and frontier_uk baselines.

The DECISIVE gate number: does the 234-feat (base-222 + 12 lever feats) 6-model stack OOF beat the
base-222 6-model stack OOF, under the IDENTICAL positive-Ridge fit? The single-LGB gate said
+ALL12 -0.128 vs base; this tests whether that survives the full LGBx3+CatBoostx3 stack (where UK
got absorbed/inverted at the LB last time).

Self-check: reproduces frontier_uk's banked stack OOF (blend_frontier.json = 10.2864) to confirm
the Ridge config matches what shipped.

Run: python -u experiments/ens_blend.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path("/home/swatson/work/kaggle/ROGII")
KEYS = ["lgb_42", "lgb_7", "lgb_123", "cat_42", "cat_7", "cat_123"]
SEED_OF = {"lgb_42": ("lgb", 42), "lgb_7": ("lgb", 7), "lgb_123": ("lgb", 123),
           "cat_42": ("cat", 42), "cat_7": ("cat", 7), "cat_123": ("cat", 123)}

# target — identical GKF seed42 row order across all three dirs (left-merge preserves order)
y = pd.read_parquet(ROOT / "data/processed/frontier_ens/train_feats.parquet",
                    columns=["target"])["target"].to_numpy(np.float32)


def stack_oof(model_dir):
    d = ROOT / "models" / model_dir
    oofs = []
    for k in KEYS:
        fam, seed = SEED_OF[k]
        oofs.append(np.load(d / f"oof_{fam}_{seed}.npy"))
    OOF = np.column_stack(oofs)
    assert len(OOF) == len(y), f"{model_dir}: {len(OOF)} rows vs y {len(y)}"
    r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(OOF, y)
    rmse = float(np.sqrt(np.mean((r.predict(OOF) - y) ** 2)))
    return rmse, r.coef_.tolist()


print("=== 6-model positive-Ridge stack OOF (pre-PF-blend) ===", flush=True)
# self-check first
uk_rmse, uk_coef = stack_oof("frontier_uk")
print(f"  frontier_uk (225): {uk_rmse:.4f}  [self-check vs banked 10.2864]", flush=True)
assert abs(uk_rmse - 10.2864) < 0.01, f"Ridge config mismatch: got {uk_rmse}"

base_rmse, base_coef = stack_oof("frontier")
print(f"  frontier    (222): {base_rmse:.4f}  <- BASELINE", flush=True)

ens_rmse, ens_coef = stack_oof("frontier_ens")
print(f"  frontier_ens(234): {ens_rmse:.4f}  coef={[round(c,3) for c in ens_coef]}", flush=True)

print("\n=== VERDICT (stack OOF deltas vs base-222) ===", flush=True)
print(f"  frontier_uk  - frontier: {uk_rmse - base_rmse:+.4f}  (UK-only; shipped, LB +0.060 REGRESSION)", flush=True)
print(f"  frontier_ens - frontier: {ens_rmse - base_rmse:+.4f}  <-- THE 12-feat joint test on the full stack", flush=True)
d = ens_rmse - base_rmse
if d <= -0.05:
    print("  >> JOINT SIGNAL SURVIVES THE STACK. Productionize-path: build a kernel that computes the", flush=True)
    print("     12 feats on hidden wells + PF blend, then ONE LB submission to settle OOF<->LB transfer.", flush=True)
elif d < 0:
    print("  >> WEAK/AMBIGUOUS: smaller than single-LGB -0.128; the stack partly absorbs it. Compare", flush=True)
    print("     vs the UK-only delta — if ~same as UK, it's the known-bad UK lever and LB will likely invert.", flush=True)
else:
    print("  >> ABSORBED: the full stack erases the joint signal. STOP -- consistent with 'levers exhausted'.", flush=True)
json.dump({"base": base_rmse, "frontier_uk": uk_rmse, "frontier_ens": ens_rmse,
           "ens_coef": ens_coef, "delta_vs_base": d}, open(ROOT / "models/frontier_ens/blend_summary.json", "w"), indent=2)
print("=== ENS BLEND DONE ===", flush=True)
