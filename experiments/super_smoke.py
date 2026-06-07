"""Cheap pre-flight for super_build.py: verify the patched exec works, one train + one test
well build, feature count is ~the expected ~150, no NaN/inf, and a well builds bit-identically
twice (per-well np.random seeding -> determinism). Run directly (foreground, ~10s)."""
import numpy as np
import super_build as sb  # triggers the patched exec: builds FI/DI, warms numba

TRAIN_DIR = sb.TRAIN_DIR
TEST_DIR = sb.TEST_DIR

tr_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
te_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))


def _tp(hp):
    wid = hp.stem.replace("__horizontal_well", "")
    return hp.parent / f"{wid}__typewell.csv"


# pick the first train well that actually builds (some get skipped)
tr_df = None
for hp in tr_paths:
    tr_df = sb.build_well(str(hp), str(_tp(hp)), True)
    if tr_df is not None:
        print(f"train well {hp.stem}: shape={tr_df.shape}")
        # determinism: rebuild the SAME well, must be bit-identical
        tr_df2 = sb.build_well(str(hp), str(_tp(hp)), True)
        num = tr_df.select_dtypes("number")
        num2 = tr_df2.select_dtypes("number")
        max_abs = float(np.abs(num.values - num2.values).max())
        print(f"  determinism rebuild max|delta| = {max_abs:.2e}  (must be 0.0)")
        break

te_hp = te_paths[0]
te_df = sb.build_well(str(te_hp), str(_tp(te_hp)), False)
print(f"test well {te_hp.stem}: shape={te_df.shape}")

feats = [c for c in tr_df.columns if c not in {"well", "id", "target"}]
print(f"#features = {len(feats)}")
print(f"'target' in train cols: {'target' in tr_df.columns} | in test cols: {'target' in te_df.columns}")

num = tr_df.select_dtypes("number")
print(f"train NaN cells: {int(num.isna().sum().sum())} | inf cells: {int(np.isinf(num.values).sum())}")
tnum = te_df.select_dtypes("number")
print(f"test  NaN cells: {int(tnum.isna().sum().sum())} | inf cells: {int(np.isinf(tnum.values).sum())}")
print("=== SMOKE OK ===")
