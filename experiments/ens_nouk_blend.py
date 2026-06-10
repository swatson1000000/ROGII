"""No-UK 6-model positive-Ridge stack blend; writes kernel-format blend_frontier.json.

Compares the 231-feat (base-222 + dip + cwt, NO UK) 6-model stack OOF vs base-222 and vs the
234-feat ens (which shipped LB 8.171, a regression). Self-checks the base-222 stack reproduces the
banked 10.3556. Writes models/frontier_ens_nouk/blend_frontier.json = {keys, ridge_coef, oof}.

Run: python -u experiments/ens_nouk_blend.py
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

y = pd.read_parquet(ROOT / "data/processed/frontier_ens/train_feats.parquet",
                    columns=["target"])["target"].to_numpy(np.float32)


def stack_oof(model_dir):
    d = ROOT / "models" / model_dir
    OOF = np.column_stack([np.load(d / f"oof_{SEED_OF[k][0]}_{SEED_OF[k][1]}.npy") for k in KEYS])
    assert len(OOF) == len(y), f"{model_dir}: {len(OOF)} rows vs y {len(y)}"
    r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(OOF, y)
    rmse = float(np.sqrt(np.mean((r.predict(OOF) - y) ** 2)))
    return rmse, r.coef_.tolist()


print("=== 6-model positive-Ridge stack OOF (pre-PF-blend) ===", flush=True)
base_rmse, _ = stack_oof("frontier")
print(f"  frontier      (222): {base_rmse:.4f}  [self-check vs banked 10.3556]", flush=True)
assert abs(base_rmse - 10.3556) < 0.01, f"base config mismatch: got {base_rmse}"

ens_rmse, _ = stack_oof("frontier_ens")
print(f"  frontier_ens  (234): {ens_rmse:.4f}  (shipped LB 8.171, regressed)", flush=True)

nouk_rmse, nouk_coef = stack_oof("frontier_ens_nouk")
print(f"  frontier_ens_nouk(231): {nouk_rmse:.4f}  coef={[round(c,3) for c in nouk_coef]}", flush=True)

print("\n=== VERDICT (stack OOF deltas vs base-222) ===", flush=True)
print(f"  frontier_ens      - frontier: {ens_rmse - base_rmse:+.4f}  (with UK)", flush=True)
print(f"  frontier_ens_nouk - frontier: {nouk_rmse - base_rmse:+.4f}  <-- NO-UK joint test", flush=True)

json.dump({"keys": KEYS, "ridge_coef": nouk_coef, "oof": nouk_rmse},
          open(ROOT / "models/frontier_ens_nouk/blend_frontier.json", "w"), indent=2)
print(f"\n>> wrote models/frontier_ens_nouk/blend_frontier.json (oof={nouk_rmse:.4f})", flush=True)
print("=== ENS NO-UK BLEND DONE ===", flush=True)
